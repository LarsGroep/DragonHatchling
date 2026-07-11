"""Self-supervised trainer for UMT-ViT (ARCHITECTURE §3.8, §7, §9 row U4).

Drives the label-free pre-training loop that ties the U2/U3 modules and the U4
loss suite together, exactly as the notebook reference's training cell does, but
as a resumable, callback-driven object:

- **Optimisation:** AdamW over the model parameters plus — in ``gradient`` SOM
  mode — the SOM neurons; a warmup-then-cosine learning-rate schedule; the DESOM
  SOM-neighbourhood ``σ`` annealed exponentially over the run
  (:func:`umtvit.models.som3d.resolve_sigma`).
- **Memory knobs (§7):** automatic mixed precision via :func:`torch.autocast` +
  :class:`torch.amp.GradScaler` (only when CUDA *and* ``train.amp``; the
  ``enabled=False`` path runs the same code on CPU), and optional gradient
  checkpointing over the encoder blocks (``train.grad_checkpoint``).
- **Per step:** two augmented views → all objective terms (geodesic gated to
  ``lambda_geodesic > 0``), voxel subsampling (``train.som_sample_voxels``) and
  SOM best-match-unit hit accumulation.
- **Per epoch:** dead-neuron revival (``model.som_revival``), SOM metrics, a
  ``history`` dict of per-step term series (in the spirit of the notebook), and a
  ``on_epoch_end(epoch, metrics, trainer)`` callback so U5/U6 can snapshot
  without subclassing.
- **Checkpointing:** :meth:`save_checkpoint` writes ``{model, som, config,
  history, optimizer, scaler, scheduler, epoch, step, rng}`` and
  :meth:`load_checkpoint` restores the *exact* training state (parameters,
  optimiser/scaler/scheduler, counters, and torch/numpy/python RNG) so
  ``train(resume_from=...)`` continues bit-for-bit on CPU with AMP off.
"""

from __future__ import annotations

import contextlib
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from umtvit.config import Config
from umtvit.losses import (
    geodesic_loss,
    monotone_centroid_loss,
    nt_xent,
    ordering_loss,
    smoothness_loss,
    total_loss,
)
from umtvit.models.model import UMTViT
from umtvit.models.som3d import Soft3DSOM, resolve_sigma

__all__ = ["Trainer"]

# AdamW weight decay. The config schema (ARCHITECTURE §4) does not carry a
# weight-decay knob, so the notebook reference's value is used as the trainer
# default; it does not affect resume determinism.
_WEIGHT_DECAY = 0.05

# Per-step history keys: the loss-term series plus the two schedule traces,
# matching the notebook reference's ``history`` dict.
_TERM_KEYS = ("ntxent", "som", "smooth", "order", "order_monotone", "geodesic")
_HISTORY_KEYS = ("total",) + _TERM_KEYS + ("sigma", "lr")

# A training data source: a ready DataLoader, a map-style dataset (wrapped into a
# DataLoader here), or a zero-arg factory returning a DataLoader.
DataSource = Union[DataLoader, Callable[[], DataLoader], Any]
EpochCallback = Callable[[int, Dict[str, Any], "Trainer"], None]


class Trainer:
    """Resumable self-supervised trainer for :class:`~umtvit.models.model.UMTViT`.

    Args:
        config: A validated :class:`~umtvit.config.Config`; every schedule and
            weight is read from it.
        model: The UMT-ViT encoder producing ``{"volume", "pooled", "proj",
            "layers"}`` per forward.
        som: The 3-D SOM consumed by the quantization term and hit/revival
            bookkeeping. In ``gradient`` update mode its neurons join the
            optimiser.
        data: Training data — a :class:`~torch.utils.data.DataLoader`, a map-style
            dataset (wrapped here with ``train.batch_size``), or a factory
            returning a DataLoader. Each batch is ``(view_a, view_b, label)``.
        on_epoch_end: Optional callback invoked as ``on_epoch_end(epoch, metrics,
            trainer)`` after each epoch (for snapshots/logging).

    The constructor seeds torch/numpy/python from ``train.seed`` so the training
    loop's randomness (voxel subsampling, any shuffling generator) is
    reproducible independently of how ``model``/``som`` were initialised.
    """

    def __init__(
        self,
        config: Config,
        model: UMTViT,
        som: Soft3DSOM,
        data: DataSource,
        on_epoch_end: Optional[EpochCallback] = None,
    ) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.som = som.to(self.device)
        self.on_epoch_end: Optional[EpochCallback] = on_epoch_end

        self._set_seed(config.train.seed)
        self.loader = self._resolve_loader(data)
        self.steps_per_epoch = max(1, len(self.loader))

        # AMP is a GPU-only path; on CPU the scaler/autocast are disabled no-ops
        # but the same call sites run (code-covered).
        self.use_amp = bool(config.train.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        # Gradient checkpointing over the encoder blocks (§7), toggled by config.
        self.model.backbone.encoder.use_checkpoint = bool(
            config.train.grad_checkpoint
        )

        self.som_sample_voxels = int(config.train.som_sample_voxels)

        # Objective weights (ARCHITECTURE §3.8); geodesic is gated on its weight.
        ls = config.loss
        self.weights: Dict[str, float] = {
            "ntxent": float(ls.lambda_ntxent),
            "som": float(ls.lambda_som),
            "smooth": float(ls.lambda_smooth),
            "order": float(ls.lambda_order),
            "order_monotone": float(ls.order_monotone),
            "geodesic": float(ls.lambda_geodesic),
        }

        # Optimiser: model params + (gradient-mode) SOM neurons.
        params: List[torch.nn.Parameter] = list(self.model.parameters())
        if self.som.update == "gradient":
            params += list(self.som.parameters())
        self.opt = torch.optim.AdamW(
            params, lr=config.train.lr, weight_decay=_WEIGHT_DECAY
        )

        # Run length: exactly one of epochs / max_steps is set (schema-enforced).
        if config.train.max_steps is not None:
            self.max_steps: Optional[int] = int(config.train.max_steps)
            self.total_steps = self.max_steps
            self.budget_epochs = math.ceil(self.max_steps / self.steps_per_epoch)
        else:
            self.max_steps = None
            self.budget_epochs = int(config.train.epochs)
            self.total_steps = self.budget_epochs * self.steps_per_epoch

        self.warmup_steps = int(config.train.warmup_steps)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_mult)

        self.sigma_start, self.sigma_end = resolve_sigma(
            ls.sigma_start, ls.sigma_end, config.model.som_grid
        )

        self.step = 0
        self.epoch = 0
        self.history: Dict[str, List[float]] = {k: [] for k in _HISTORY_KEYS}
        self.metrics_history: List[Dict[str, Any]] = []

    # ---- schedules -------------------------------------------------------- #
    def _lr_mult(self, step: int) -> float:
        """Warmup-then-cosine multiplier on the base LR (LambdaLR closure)."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _sigma_at(self, step: int) -> float:
        """Exponential DESOM neighbourhood-width anneal ``σ_start → σ_end``."""
        frac = step / max(1, self.total_steps)
        return self.sigma_start * (self.sigma_end / self.sigma_start) ** frac

    def _autocast(self):
        """Mixed-precision context on GPU, a no-op on CPU (AMP disabled)."""
        if self.use_amp:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            return torch.autocast(self.device.type, dtype=dtype)
        return contextlib.nullcontext()

    # ---- setup helpers ---------------------------------------------------- #
    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _resolve_loader(self, data: DataSource) -> DataLoader:
        """Normalise ``data`` to a DataLoader (dataset ⇒ wrapped, factory ⇒ called)."""
        if isinstance(data, DataLoader):
            return data
        if callable(data) and not hasattr(data, "__getitem__"):
            loader = data()
            if not isinstance(loader, DataLoader):
                raise TypeError(
                    "data factory must return a torch DataLoader, got "
                    f"{type(loader).__name__}"
                )
            return loader
        return DataLoader(
            data, batch_size=self.config.train.batch_size, shuffle=True
        )

    # ---- one optimisation step ------------------------------------------- #
    def _step(self, xa: Tensor, xb: Tensor, epoch_hits: Tensor) -> Tuple[
        Dict[str, float], float, Tensor
    ]:
        """Run a single paired-view optimisation step; return term floats, σ, voxels."""
        sigma = self._sigma_at(self.step)
        with self._autocast():
            oa = self.model(xa)
            ob = self.model(xb)
            volume = oa["volume"]
            vox = volume.reshape(-1, volume.shape[-1])
            idx = torch.randperm(vox.shape[0], device=vox.device)[
                : self.som_sample_voxels
            ]
            v_sub = vox[idx].float()
            # Data-driven SOM init on the very first batch's voxels.
            if self.step == 0 and self.config.model.som_init == "data":
                self.som.data_init(v_sub.detach())
            terms: Dict[str, Tensor] = {
                "ntxent": nt_xent(
                    oa["proj"], ob["proj"], self.config.loss.ntxent_temperature
                ),
                "som": self.som.loss(v_sub, sigma),
                "smooth": smoothness_loss(volume, self.config.loss.smooth_axes),
                "order": ordering_loss(volume, self.config.loss.order_fmax),
                "order_monotone": monotone_centroid_loss(volume),
            }
            # Geodesic is gated: only computed when its weight is positive, so
            # weight 0 means zero overhead and a constant-zero history term.
            if self.weights["geodesic"] > 0.0:
                terms["geodesic"] = geodesic_loss(
                    v_sub, oa["proj"].float(), ob["proj"].float()
                )
            else:
                terms["geodesic"] = volume.new_zeros(())
            total, detached = total_loss(terms, self.weights)

        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(total).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()

        with torch.no_grad():
            recent = v_sub.detach()
            epoch_hits += torch.bincount(
                self.som.bmu(recent), minlength=self.som.K
            ).float()
        return detached, sigma, recent

    # ---- epoch loop ------------------------------------------------------- #
    def _run_epoch(self) -> Dict[str, Any]:
        """Train one epoch; revive dead neurons; return the epoch metrics dict."""
        self.model.train()
        epoch_hits = torch.zeros(self.som.K, device=self.device)
        recent_vox: Optional[Tensor] = None
        for batch in self.loader:
            if self.max_steps is not None and self.step >= self.max_steps:
                break
            xa = batch[0].to(self.device)
            xb = batch[1].to(self.device)
            detached, sigma, recent_vox = self._step(xa, xb, epoch_hits)
            for key in _TERM_KEYS:
                self.history[key].append(detached[key])
            self.history["total"].append(detached["total"])
            self.history["sigma"].append(sigma)
            self.history["lr"].append(self.opt.param_groups[0]["lr"])
            self.step += 1

        revived = 0
        if self.config.model.som_revival and recent_vox is not None:
            revived = self.som.revive(epoch_hits, recent_vox)

        metrics: Dict[str, Any] = {
            "epoch": self.epoch,
            "step": self.step,
            "revived": revived,
            "loss": self.history["total"][-1] if self.history["total"] else float("nan"),
            "sigma": self.history["sigma"][-1] if self.history["sigma"] else 0.0,
            "lr": self.history["lr"][-1] if self.history["lr"] else 0.0,
        }
        if recent_vox is not None:
            metrics.update(self.som.metrics(recent_vox))
        return metrics

    def _fire_epoch_end(self, metrics: Dict[str, Any]) -> None:
        self.metrics_history.append(metrics)
        if self.on_epoch_end is not None:
            self.on_epoch_end(self.epoch, metrics, self)

    def train(
        self,
        resume_from: Optional[str] = None,
        until_epoch: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Run (or resume) training and return the per-step ``history`` dict.

        Args:
            resume_from: Optional checkpoint path; restores the exact training
                state before continuing (parameters, optimiser/scaler/scheduler,
                counters, RNG).
            until_epoch: Optional cap on the epoch this call runs *up to* (of the
                config's total epoch budget) — used to checkpoint mid-run without
                changing the schedule's total-step horizon. Defaults to the full
                budget.

        Returns:
            The ``history`` dict (per-step term/schedule series).
        """
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        stop_epoch = self.budget_epochs if until_epoch is None else min(
            int(until_epoch), self.budget_epochs
        )
        while self.epoch < stop_epoch:
            if self.max_steps is not None and self.step >= self.max_steps:
                break
            self.epoch += 1
            metrics = self._run_epoch()
            self._fire_epoch_end(metrics)
        return self.history

    # ---- checkpointing ---------------------------------------------------- #
    def save_checkpoint(self, path: str) -> str:
        """Persist the full training state to ``path`` (see class docstring)."""
        checkpoint = {
            "model": self.model.state_dict(),
            "som": self.som.state_dict(),
            "config": self.config.to_dict(),
            "history": self.history,
            "metrics_history": self.metrics_history,
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.sched.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        torch.save(checkpoint, path)
        return str(path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Restore the exact training state saved by :meth:`save_checkpoint`."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.som.load_state_dict(checkpoint["som"])
        self.history = checkpoint["history"]
        self.metrics_history = checkpoint.get("metrics_history", [])
        self.opt.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.sched.load_state_dict(checkpoint["scheduler"])
        self.epoch = int(checkpoint["epoch"])
        self.step = int(checkpoint["step"])
        rng = checkpoint.get("rng")
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        return checkpoint
