"""Concept tier (§9) — k-sparse autoencoder, providers, dictionary, quality gate.

A k-sparse autoencoder (default 4096 features, k=32, layer 9) trained on
fine-tuned-model token activations yields a **concept dictionary** per
(model, dataset, layer): each feature gets top-64 exemplar patches
(image id + token idx + activation), class-conditional firing rates, and an
optional CLIP-text probe label slot (marked *suggested*; exemplars are ground
truth). A :func:`quality_gate` decides — per dataset — whether the SAE is good
enough; if it fails, callers fall back to :class:`KMeansConceptProvider` behind
the identical :class:`ConceptProvider` interface (§9, DECISION-LOG).

Import discipline
-----------------
``import vitreous.concepts`` stays free of torch and scikit-learn: the SAE
``nn.Module`` class is built lazily (module-level ``__getattr__``), and every
provider imports its heavy backend (torch / sklearn) inside the method that
needs it. Only numpy is imported eagerly (a base dependency).

TopK activation
---------------
The SAE uses the OpenAI/Gao-et-al. **TopK** activation: the ``k`` largest
pre-activations are retained (rest zeroed), giving *exactly* ``k`` nonzero
features per token and a directly controllable L0 — no L1 tuning, no
activation-threshold drift. Concept activations are these retained values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

DEFAULT_LAYER = 9
DEFAULT_N_FEATURES = 4096
DEFAULT_K = 32
DEFAULT_TOPK_EXEMPLARS = 64
DEFAULT_PACK_TOPK = 8  # per-token top-k features stored in concepts.json


# --------------------------------------------------------------------------- #
# numpy helpers (torch-free)
# --------------------------------------------------------------------------- #


def _to_numpy(x: Any) -> np.ndarray:
    """Detach + numpy-ify a torch tensor, or pass numpy/array-like through."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        x = x.numpy()
    return np.ascontiguousarray(np.asarray(x))


def layer_token_activations(trace: Any, layer: int) -> np.ndarray:
    """Extract the ``[T, D]`` token activations at timeline step ``layer`` (§9).

    ``trace.tokens`` is ``[L+1, T, D]`` (block inputs t=0..L-1 + final norm t=L),
    so ``layer=9`` selects the input to block 9 — the SAE's canonical probe site.
    """
    tokens = _to_numpy(trace.tokens).astype(np.float32)
    if tokens.ndim != 3:
        raise ValueError(f"expected trace.tokens [L+1, T, D], got {tokens.shape}")
    if not (0 <= layer < tokens.shape[0]):
        raise IndexError(f"layer {layer} out of range for {tokens.shape[0]} steps")
    return tokens[layer]


def _topk_per_row(mat: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (feature_ids ``[N,k]``, activations ``[N,k]``) — top-k per row, desc."""
    k = int(min(topk, mat.shape[1]))
    idx = np.argsort(-mat, axis=1, kind="stable")[:, :k]
    vals = np.take_along_axis(mat, idx, axis=1)
    return idx.astype(np.int64), vals.astype(np.float32)


# --------------------------------------------------------------------------- #
# dictionary dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class ExemplarRef:
    """One exemplar patch for a concept feature (§9): the ground-truth evidence."""

    image_id: str
    token_idx: int
    activation: float
    class_label: int = -1

    def to_json(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "token_idx": int(self.token_idx),
            "activation": round(float(self.activation), 6),
            "class_label": int(self.class_label),
        }


@dataclass
class ConceptFeature:
    """One dictionary feature (§9)."""

    id: int
    exemplars: List[ExemplarRef] = field(default_factory=list)
    firing_rates: List[float] = field(default_factory=list)  # per class
    coherence: float = 0.0  # top-exemplar class purity (∈ [0,1])
    # CLIP-text probe suggestion — left None; a pluggable ``label_hook`` may set
    # it. Exemplars are ground truth; labels are *suggested*, human-editable.
    suggested_label: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "exemplars": [e.to_json() for e in self.exemplars],
            "firing_rates": [round(float(r), 6) for r in self.firing_rates],
            "coherence": round(float(self.coherence), 6),
            "suggested_label": self.suggested_label,
        }


@dataclass
class ConceptDictionary:
    """A concept dictionary artifact for a (model, dataset, layer) (§9)."""

    model: str
    dataset: str
    layer: int
    provider_kind: str = "sae"
    n_concepts: int = 0
    num_classes: int = 0
    features: List[ConceptFeature] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "layer": int(self.layer),
            "provider_kind": self.provider_kind,
            "n_concepts": int(self.n_concepts),
            "num_classes": int(self.num_classes),
            "features": [f.to_json() for f in self.features],
        }


# --------------------------------------------------------------------------- #
# ConceptProvider protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class ConceptProvider(Protocol):
    """Turns token activations into per-token concept activations (§9).

    ``kind`` is ``"sae"`` or ``"kmeans"``; :meth:`encode` maps ``[N, D]``
    activations to a ``[N, C]`` concept-activation matrix (sparse for the SAE,
    centroid-similarity for k-means); :meth:`top_features_per_token` pulls a
    trace's probe-layer tokens through :meth:`encode` and returns the top-k
    concepts per token — exactly what the pack's ``concepts.json`` stores.
    """

    kind: str

    def n_concepts(self) -> int:
        ...

    def encode(self, activations: Any) -> np.ndarray:
        ...

    def top_features_per_token(
        self, trace: Any, layer: int, topk: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...


class _ProviderMixin:
    """Shared ``top_features_per_token`` / dictionary building for both providers."""

    layer: int
    kind: str

    def n_concepts(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def encode(self, activations: Any) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def top_features_per_token(
        self, trace: Any, layer: Optional[int] = None, topk: int = DEFAULT_PACK_TOPK
    ) -> Tuple[np.ndarray, np.ndarray]:
        acts = layer_token_activations(trace, self.layer if layer is None else layer)
        feats = self.encode(acts)
        return _topk_per_row(feats, topk)

    def build_dictionary(
        self,
        activations: Any,
        exemplar_refs: List[ExemplarRef],
        *,
        num_classes: int,
        model: str = "model",
        dataset: str = "dataset",
        topk_exemplars: int = DEFAULT_TOPK_EXEMPLARS,
        active_threshold: float = 0.0,
        label_hook: Optional[Callable[[int, List[ExemplarRef]], Optional[str]]] = None,
    ) -> ConceptDictionary:
        return build_concept_dictionary(
            self,
            activations,
            exemplar_refs,
            num_classes=num_classes,
            model=model,
            dataset=dataset,
            layer=self.layer,
            topk_exemplars=topk_exemplars,
            active_threshold=active_threshold,
            label_hook=label_hook,
        )


# --------------------------------------------------------------------------- #
# lazy TopK SAE class (torch imported only on first access)
# --------------------------------------------------------------------------- #

_SAE_CLASS: Optional[type] = None


def _build_sae_class() -> type:
    """Build (and cache) the torch ``KSparseAutoencoder`` class. Imports torch."""
    global _SAE_CLASS
    if _SAE_CLASS is not None:
        return _SAE_CLASS

    import torch
    from torch import nn

    class KSparseAutoencoder(nn.Module):
        """TopK sparse autoencoder over token activations (§9).

        encoder ``Linear(D -> F)`` → TopK(k) → decoder ``Linear(F -> D)``. A
        pre-encoder bias ``b_dec`` is subtracted before encoding and added back
        after decoding (Anthropic/Gao convention). ``tie_decoder=True`` ties the
        decoder to the encoder transpose (no separate ``W_dec``).
        """

        def __init__(
            self,
            d_in: int = 384,
            n_features: int = DEFAULT_N_FEATURES,
            k: int = DEFAULT_K,
            tie_decoder: bool = False,
            seed: int = 0,
        ) -> None:
            super().__init__()
            self.d_in = int(d_in)
            self.n_features = int(n_features)
            self.k = int(k)
            self.tie_decoder = bool(tie_decoder)
            g = torch.Generator().manual_seed(int(seed))

            W_enc = torch.empty(self.n_features, self.d_in)
            nn.init.kaiming_uniform_(W_enc, a=5 ** 0.5, generator=g)
            self.W_enc = nn.Parameter(W_enc)
            self.b_enc = nn.Parameter(torch.zeros(self.n_features))
            self.b_dec = nn.Parameter(torch.zeros(self.d_in))
            if self.tie_decoder:
                self.register_parameter("W_dec", None)
            else:
                # Init decoder columns as unit-norm encoder transpose.
                W_dec = W_enc.t().clone()
                W_dec = W_dec / (W_dec.norm(dim=0, keepdim=True) + 1e-8)
                self.W_dec = nn.Parameter(W_dec)  # [d_in, n_features]

        def decoder_weight(self) -> Any:
            """Return the ``[d_in, n_features]`` decoder matrix (tied or free)."""
            return self.W_enc.t() if self.tie_decoder else self.W_dec

        def preacts(self, x: Any) -> Any:
            return (x - self.b_dec) @ self.W_enc.t() + self.b_enc

        def topk(self, pre: Any) -> Any:
            k = min(self.k, pre.shape[-1])
            vals, idx = pre.topk(k, dim=-1)
            out = torch.zeros_like(pre)
            out.scatter_(-1, idx, vals)
            return out

        def encode(self, x: Any) -> Any:
            """Return the sparse ``[.., F]`` TopK concept code (exactly k nonzeros)."""
            return self.topk(self.preacts(x))

        def decode(self, feats: Any) -> Any:
            return feats @ self.decoder_weight().t() + self.b_dec

        def forward(self, x: Any):
            feats = self.encode(x)
            return self.decode(feats), feats

    _SAE_CLASS = KSparseAutoencoder
    return _SAE_CLASS


def __getattr__(name: str) -> Any:  # PEP 562 — lazy torch import for the SAE class
    if name == "KSparseAutoencoder":
        return _build_sae_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- #
# training + stats
# --------------------------------------------------------------------------- #


@dataclass
class SAEStats:
    """Training diagnostics for a trained SAE (§9)."""

    n_features: int
    k: int
    n_samples: int
    epochs: int
    dead_feature_count: int
    dead_feature_rate: float
    l0: float
    recon_loss: float
    initial_recon_loss: float
    duplicate_feature_rate: float

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        return {
            key: (round(v, 6) if isinstance(v, float) else v) for key, v in d.items()
        }


def train_sae(
    activations: Any,
    *,
    epochs: int = 100,
    k: int = DEFAULT_K,
    n_features: int = DEFAULT_N_FEATURES,
    seed: int = 0,
    lr: float = 1e-3,
    batch_size: Optional[int] = None,
    tie_decoder: bool = False,
    device: str = "cpu",
    duplicate_threshold: float = 0.9,
    verbose: bool = False,
) -> Tuple[Any, SAEStats]:
    """Train a TopK sparse autoencoder on ``activations`` ``[N, D]`` (§9).

    Pure torch, CPU-runnable on small ``N``. Returns ``(sae, stats)`` where
    ``stats`` reports the dead-feature count/rate, final & initial reconstruction
    loss, measured L0 (== ``k``), and the duplicate-feature rate (fraction of
    features whose decoder direction has cosine > ``duplicate_threshold`` with
    another feature). Feed the trained SAE to :class:`SAEConceptProvider`.
    """
    import torch

    X = torch.as_tensor(_to_numpy(activations), dtype=torch.float32, device=device)
    if X.ndim != 2:
        raise ValueError(f"expected activations [N, D], got shape {tuple(X.shape)}")
    n, d = int(X.shape[0]), int(X.shape[1])
    torch.manual_seed(seed)

    sae = _build_sae_class()(
        d_in=d, n_features=n_features, k=k, tie_decoder=tie_decoder, seed=seed
    ).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    mse = torch.nn.functional.mse_loss

    with torch.no_grad():
        initial_loss = float(mse(sae(X)[0], X).item())

    bs = int(batch_size) if batch_size else n
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(int(epochs)):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, bs):
            xb = X[perm[start : start + bs]]
            recon, _feats = sae(xb)
            loss = mse(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if verbose and (epoch % max(1, epochs // 10) == 0):
            print(f"epoch {epoch}: loss {float(loss.item()):.5f}")

    # -- diagnostics --------------------------------------------------------- #
    sae.eval()
    with torch.no_grad():
        feats = sae.encode(X)                       # [N, F]
        recon = sae.decode(feats)
        final_loss = float(mse(recon, X).item())
        nonzero = feats != 0
        active_per_feature = nonzero.sum(dim=0)     # [F]
        dead = int((active_per_feature == 0).sum().item())
        l0 = float(nonzero.sum(dim=1).float().mean().item())

        Wd = sae.decoder_weight().detach()          # [d_in, F]
        Wn = Wd / (Wd.norm(dim=0, keepdim=True) + 1e-8)
        sim = Wn.t() @ Wn                           # [F, F]
        sim.fill_diagonal_(0.0)
        max_sim = sim.max(dim=1).values
        dup_rate = float((max_sim > duplicate_threshold).float().mean().item())

    stats = SAEStats(
        n_features=int(n_features),
        k=int(k),
        n_samples=n,
        epochs=int(epochs),
        dead_feature_count=dead,
        dead_feature_rate=dead / float(n_features),
        l0=l0,
        recon_loss=final_loss,
        initial_recon_loss=initial_loss,
        duplicate_feature_rate=dup_rate,
    )
    return sae, stats


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #


@dataclass
class SAEConceptProvider(_ProviderMixin):
    """ConceptProvider whose features are trained SAE latents (§9, default)."""

    sae: Any
    layer: int = DEFAULT_LAYER
    kind: str = "sae"

    def n_concepts(self) -> int:
        return int(self.sae.n_features)

    def encode(self, activations: Any) -> np.ndarray:
        import torch

        X = torch.as_tensor(_to_numpy(activations), dtype=torch.float32)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        was_training = self.sae.training
        self.sae.eval()
        try:
            with torch.no_grad():
                feats = self.sae.encode(X)
        finally:
            if was_training:
                self.sae.train()
        return _to_numpy(feats).astype(np.float32)


@dataclass
class KMeansConceptProvider(_ProviderMixin):
    """Fallback ConceptProvider: k-means over activations; cluster id = concept.

    :meth:`encode` returns a softmax centroid-similarity matrix ``[N, K]`` (row =
    token, column = cluster), so the nearest centroid is the token's top concept
    and :meth:`top_features_per_token` ranks concepts identically to the SAE path.
    """

    centroids: np.ndarray
    layer: int = DEFAULT_LAYER
    kind: str = "kmeans"
    temperature: float = 1.0

    @classmethod
    def fit(
        cls,
        activations: Any,
        *,
        n_clusters: int = 256,
        layer: int = DEFAULT_LAYER,
        seed: int = 0,
        temperature: float = 1.0,
    ) -> "KMeansConceptProvider":
        """Fit k-means (scikit-learn, lazy import) over ``[N, D]`` activations."""
        from sklearn.cluster import KMeans

        X = _to_numpy(activations).astype(np.float32)
        km = KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init="auto")
        km.fit(X)
        return cls(
            centroids=km.cluster_centers_.astype(np.float32),
            layer=int(layer),
            temperature=float(temperature),
        )

    def n_concepts(self) -> int:
        return int(self.centroids.shape[0])

    def encode(self, activations: Any) -> np.ndarray:
        X = _to_numpy(activations).astype(np.float32)
        if X.ndim == 1:
            X = X[None, :]
        # Squared euclidean distance to each centroid → softmax similarity.
        d2 = (
            (X ** 2).sum(1, keepdims=True)
            - 2.0 * X @ self.centroids.T
            + (self.centroids ** 2).sum(1)[None, :]
        )
        d2 = np.maximum(d2, 0.0)
        scale = float(self.temperature) * (np.median(d2) + 1e-6)
        logits = -d2 / scale
        logits -= logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        return (ex / (ex.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


# --------------------------------------------------------------------------- #
# dictionary building
# --------------------------------------------------------------------------- #


def build_concept_dictionary(
    provider: ConceptProvider,
    activations: Any,
    exemplar_refs: List[ExemplarRef],
    *,
    num_classes: int,
    model: str = "model",
    dataset: str = "dataset",
    layer: int = DEFAULT_LAYER,
    topk_exemplars: int = DEFAULT_TOPK_EXEMPLARS,
    active_threshold: float = 0.0,
    label_hook: Optional[Callable[[int, List[ExemplarRef]], Optional[str]]] = None,
) -> ConceptDictionary:
    """Build a :class:`ConceptDictionary` from a provider + a bank of activations.

    Parameters
    ----------
    provider:
        A :class:`ConceptProvider` (SAE or k-means).
    activations:
        ``[N, D]`` token activations (the SAE/k-means input space).
    exemplar_refs:
        Length-``N`` list of :class:`ExemplarRef` — one per activation row —
        carrying ``(image_id, token_idx, class_label)`` so exemplar patches and
        class-conditional firing rates can be attributed.
    num_classes:
        Number of dataset classes (firing-rate vector length).
    label_hook:
        Optional ``(feature_id, exemplars) -> Optional[str]`` callable that
        suggests a label (e.g. a CLIP text probe). **No CLIP dependency is
        added**; the hook is the pluggable seam. Defaults to ``None`` labels.
    """
    X = _to_numpy(activations).astype(np.float32)
    n = X.shape[0]
    if len(exemplar_refs) != n:
        raise ValueError(
            f"exemplar_refs has {len(exemplar_refs)} entries but activations has {n} rows"
        )
    feats = provider.encode(X)                      # [N, C]
    C = feats.shape[1]
    classes = np.array([int(e.class_label) for e in exemplar_refs], dtype=np.int64)

    features: List[ConceptFeature] = []
    for c in range(C):
        col = feats[:, c]
        active = col > active_threshold
        # class-conditional firing rate: P(feature active | class).
        firing = [0.0] * int(num_classes)
        for cls in range(int(num_classes)):
            in_cls = classes == cls
            denom = int(in_cls.sum())
            if denom:
                firing[cls] = float((active & in_cls).sum()) / denom

        # top exemplars: strongest active tokens (descending activation).
        order = np.argsort(-col, kind="stable")
        exemplars: List[ExemplarRef] = []
        for i in order[: max(topk_exemplars, 1)]:
            if not active[i]:
                break
            ref = exemplar_refs[int(i)]
            exemplars.append(
                ExemplarRef(
                    image_id=ref.image_id,
                    token_idx=int(ref.token_idx),
                    activation=float(col[int(i)]),
                    class_label=int(ref.class_label),
                )
            )

        # coherence proxy: class purity of the top exemplars (∈ [0,1]).
        coherence = 0.0
        if exemplars:
            labels = [e.class_label for e in exemplars]
            coherence = max(labels.count(x) for x in set(labels)) / len(labels)

        suggested = label_hook(c, exemplars) if label_hook is not None else None
        features.append(
            ConceptFeature(
                id=c,
                exemplars=exemplars,
                firing_rates=firing,
                coherence=float(coherence),
                suggested_label=suggested,
            )
        )

    return ConceptDictionary(
        model=model,
        dataset=dataset,
        layer=int(layer),
        provider_kind=getattr(provider, "kind", "sae"),
        n_concepts=int(C),
        num_classes=int(num_classes),
        features=features,
    )


# --------------------------------------------------------------------------- #
# quality gate
# --------------------------------------------------------------------------- #


@dataclass
class QualityReport:
    """SAE quality assessment + the SAE-vs-fallback decision (§9)."""

    dead_feature_rate: float
    exemplar_coherence: float
    duplicate_feature_rate: float
    use_sae: bool
    reasons: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "dead_feature_rate": round(float(self.dead_feature_rate), 6),
            "exemplar_coherence": round(float(self.exemplar_coherence), 6),
            "duplicate_feature_rate": round(float(self.duplicate_feature_rate), 6),
            "use_sae": bool(self.use_sae),
            "reasons": list(self.reasons),
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
        }


def quality_gate(
    sae_stats: Optional[SAEStats],
    dictionary: ConceptDictionary,
    *,
    max_dead_rate: float = 0.5,
    min_exemplar_coherence: float = 0.30,
    max_duplicate_rate: float = 0.35,
) -> QualityReport:
    """Decide whether the SAE is good enough, else fall back to k-means (§9).

    Thresholds (defaults, calibrated for ~10-class EuroSAT-scale dictionaries;
    override per dataset):

    * **dead-feature rate ≤ 0.50** — at most half the features may never fire.
      Mostly-dead dictionaries carry little signal.
    * **exemplar coherence ≥ 0.30** — mean class-purity of each live feature's
      top exemplars. A degenerate/random SAE scores near ``1/num_classes``; the
      floor sits comfortably above chance for the target class counts.
    * **duplicate-feature rate ≤ 0.35** — fraction of live features sharing their
      single strongest exemplar with another feature (a cheap collision proxy for
      the decoder-direction duplication that plagues small-data SAEs).

    ``dead_feature_rate`` is read from ``sae_stats`` when present (the SAE's own
    measurement); the coherence and duplicate proxies are computed from the
    dictionary, so the gate also works for a dictionary built without stats
    (``sae_stats=None`` → dead-rate 0).
    """
    live = [f for f in dictionary.features if f.exemplars]
    n_live = len(live)

    dead_rate = float(sae_stats.dead_feature_rate) if sae_stats is not None else 0.0

    coherence = (
        float(np.mean([f.coherence for f in live])) if n_live else 0.0
    )

    if n_live:
        top1 = [(f.exemplars[0].image_id, f.exemplars[0].token_idx) for f in live]
        dup_rate = 1.0 - (len(set(top1)) / float(n_live))
    else:
        dup_rate = 1.0

    reasons: List[str] = []
    if dead_rate > max_dead_rate:
        reasons.append(
            f"dead_feature_rate {dead_rate:.3f} > {max_dead_rate:.3f}"
        )
    if coherence < min_exemplar_coherence:
        reasons.append(
            f"exemplar_coherence {coherence:.3f} < {min_exemplar_coherence:.3f}"
        )
    if dup_rate > max_duplicate_rate:
        reasons.append(
            f"duplicate_feature_rate {dup_rate:.3f} > {max_duplicate_rate:.3f}"
        )

    return QualityReport(
        dead_feature_rate=dead_rate,
        exemplar_coherence=coherence,
        duplicate_feature_rate=dup_rate,
        use_sae=not reasons,
        reasons=reasons,
        thresholds={
            "max_dead_rate": float(max_dead_rate),
            "min_exemplar_coherence": float(min_exemplar_coherence),
            "max_duplicate_rate": float(max_duplicate_rate),
        },
    )


# --------------------------------------------------------------------------- #
# pack integration spec
# --------------------------------------------------------------------------- #


@dataclass
class ConceptPackSpec:
    """What :func:`vitreous.packs.build_pack` needs to emit ``concepts.json`` (§9).

    ``provider`` supplies per-token top-k features; ``dictionary_id`` is the
    Postgres/Storage reference (``concept_dictionaries.id``) the frontend follows
    to load exemplars. ``concepts.json`` is an **additive** pack asset — packs
    without concepts simply omit it, and ``pack_version`` stays ``1.0.0``.
    """

    provider: ConceptProvider
    dictionary_id: str
    layer: int = DEFAULT_LAYER
    topk: int = DEFAULT_PACK_TOPK

    def build_asset(self, trace: Any) -> Dict[str, Any]:
        """Build the ``concepts.json`` payload for one image's trace."""
        ids, acts = self.provider.top_features_per_token(
            trace, layer=self.layer, topk=self.topk
        )
        return {
            "layer": int(self.layer),
            "topk": int(ids.shape[1]),
            "dictionary_id": self.dictionary_id,
            "provider_kind": getattr(self.provider, "kind", "sae"),
            "n_concepts": int(self.provider.n_concepts()),
            "num_tokens": int(ids.shape[0]),
            "feature_ids": ids.astype(np.int64).tolist(),
            "activations": np.round(acts.astype(np.float32), 6).tolist(),
        }


__all__ = [
    "DEFAULT_LAYER",
    "DEFAULT_N_FEATURES",
    "DEFAULT_K",
    "DEFAULT_TOPK_EXEMPLARS",
    "DEFAULT_PACK_TOPK",
    "ExemplarRef",
    "ConceptFeature",
    "ConceptDictionary",
    "ConceptProvider",
    "SAEStats",
    "train_sae",
    "SAEConceptProvider",
    "KMeansConceptProvider",
    "build_concept_dictionary",
    "QualityReport",
    "quality_gate",
    "ConceptPackSpec",
    "layer_token_activations",
    # KSparseAutoencoder is provided lazily via module __getattr__.
]
