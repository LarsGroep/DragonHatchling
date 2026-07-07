"""Generic training loop, dataset- and backbone-agnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from hatchvision.hebbian.memory import HebbianFeatureMemory


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 5e-4
    device: str = ""            # "" = auto
    log_every: int = 50
    # Per-class loss weights: pass torch.Tensor([w0, w1, ...]) to upweight
    # rare classes. Use compute_class_weights() to derive from label counts.
    class_weights: Optional[torch.Tensor] = None
    # Cosine annealing: restarts LR to lr every `lr_cycle_epochs` epochs.
    # 0 = fixed LR (default). Good for fine-tuning on imbalanced medical data.
    lr_cycle_epochs: int = 0


def compute_class_weights(
    labels: Sequence[int], num_classes: int, method: str = "inv_freq"
) -> torch.Tensor:
    """Derive per-class loss weights from a sequence of integer class labels.

    ``method="inv_freq"`` (default): weight_k = N / (num_classes * count_k),
    matching scikit-learn's "balanced" mode. Use for highly imbalanced datasets
    like HAM10000 where the majority class outnumbers rare ones by 50-70x.
    """
    counts = torch.zeros(num_classes)
    for lbl in labels:
        counts[lbl] += 1
    counts = counts.clamp(min=1)
    if method == "inv_freq":
        n = counts.sum()
        return (n / (num_classes * counts)).float()
    raise ValueError(f"Unknown method {method!r}; use 'inv_freq'")


def resolve_device(requested: str = "") -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Trains any ImageClassifier; optionally feeds a Hebbian memory.

    The memory is observation-only: the trainer merely forwards each batch's
    labels to it so class-conditional statistics can accumulate.  Removing
    the memory changes nothing about optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig = TrainConfig(),
        hebbian_memory: Optional[HebbianFeatureMemory] = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.memory = hebbian_memory
        weights = config.class_weights
        if weights is not None:
            weights = weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        if config.lr_cycle_epochs > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.lr_cycle_epochs, eta_min=config.lr * 0.01
            )
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        }

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for step, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            if self.memory is not None:
                self.memory.observe_labels(labels)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            bs = labels.shape[0]
            total += bs
            loss_sum += loss.item() * bs
            correct += (logits.argmax(1) == labels).sum().item()
            if self.config.log_every and step % self.config.log_every == 0:
                print(
                    f"  step {step:4d}/{len(loader)}  "
                    f"loss {loss.item():.4f}  acc {correct / total:.3f}"
                )
        return loss_sum / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        # Validation passes should not contaminate the Hebbian statistics.
        ctx = self.memory.paused() if self.memory is not None else None
        if ctx:
            ctx.__enter__()
        try:
            total, correct, loss_sum = 0, 0, 0.0
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                bs = labels.shape[0]
                total += bs
                loss_sum += loss.item() * bs
                correct += (logits.argmax(1) == labels).sum().item()
            return loss_sum / total, correct / total
        finally:
            if ctx:
                ctx.__exit__(None, None, None)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        for epoch in range(self.config.epochs):
            print(f"epoch {epoch + 1}/{self.config.epochs}")
            tr_loss, tr_acc = self.train_epoch(train_loader)
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            msg = f"  train loss {tr_loss:.4f} acc {tr_acc:.3f}"
            if val_loader is not None:
                va_loss, va_acc = self.evaluate(val_loader)
                self.history["val_loss"].append(va_loss)
                self.history["val_acc"].append(va_acc)
                msg += f" | val loss {va_loss:.4f} acc {va_acc:.3f}"
            print(msg)
            if self.scheduler is not None:
                self.scheduler.step()
        return self.history
