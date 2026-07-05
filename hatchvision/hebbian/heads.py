"""Gradient-free classifiers built directly from Hebbian statistics.

The Hebbian feature memory is normally observation-only and classification is
left to a trained gradient head.  These heads instead turn the accumulated
co-activation / class-firing statistics into *actual* classifiers, with **no
gradient descent** on the classification weights:

* :class:`HebbianPrototypeHead` — nearest-prototype (cosine) over per-class
  mean firing vectors.  Supports few-shot ``enroll`` of brand-new classes the
  backbone never trained on, by a running mean of their activations.
* :class:`TreeRoutedHead` — routes an image down a :class:`ConceptNode` tree
  (decision-tree style) and reads the class distribution off the reached
  leaf/leaves; hard and soft routing.
* :class:`ConceptBottleneckHead` — flat concepts → concept scores → classes
  via the concepts' class-affinity matrix (optionally a fitted logistic
  upper bound).

Every head consumes the **same** activation representation the memory tracks:
pooled, ReLU'd, then L2-normalized ``a / (||a|| + 1e-8)`` activations of the
observed layer, restricted to the tracked ``unit_index``.  Callers obtain
these with :func:`~hatchvision.explain.concepts.probe_activations` and pass
the single layer's tensor ``acts[layer]`` (shape ``[n_images, units]``).
Matching the memory's normalization exactly is what makes the cosine
prototypes meaningful — a magnitude mismatch here collapses accuracy to
chance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from hatchvision.explain.concepts import (
    Concept,
    cluster_concepts,
    concept_scores,
)
from hatchvision.hebbian.hierarchy import ConceptNode
from hatchvision.hebbian.memory import HebbianFeatureMemory


def _normalize(a: torch.Tensor) -> torch.Tensor:
    """L2-normalize each row exactly as the memory does."""
    a = a.float()
    return a / (a.norm(dim=1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------- prototypes


class HebbianPrototypeHead:
    """Nearest-prototype classifier over per-class mean firing vectors.

    The prototype of a class is the L2-normalized mean of the (already
    L2-normalized) activation vectors of its training images — exactly
    ``normalize(class_act / class_count)`` from the memory.  Prediction is
    temperature-scaled cosine similarity to every prototype.  No gradients are
    involved, and :meth:`enroll` adds or refreshes a prototype from a handful
    of activation vectors, so classes absent from training can be taught
    few-shot at inference time.
    """

    def __init__(
        self,
        layer: str,
        class_names: Sequence[str],
        mean_vectors: torch.Tensor,     # [C, U] pre-normalization means
        counts: torch.Tensor,           # [C]
        temperature: float = 0.1,
    ) -> None:
        self.layer = layer
        self.class_names = list(class_names)
        self._means = mean_vectors.float().clone()
        self._counts = counts.float().clone()
        self.temperature = float(temperature)
        self.prototypes = _normalize(self._means)

    @classmethod
    def from_memory(
        cls,
        memory: HebbianFeatureMemory,
        layer: str,
        class_names: Sequence[str],
        temperature: float = 0.1,
    ) -> "HebbianPrototypeHead":
        st = memory.stats[layer]
        counts = st.class_count.clone()
        means = st.class_act / torch.clamp(counts[:, None], min=1.0)
        return cls(layer, class_names, means, counts, temperature)

    @classmethod
    def from_activations(
        cls,
        layer: str,
        class_names: Sequence[str],
        activations: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.1,
    ) -> "HebbianPrototypeHead":
        """Build prototypes from *final-model* activations instead of memory.

        The memory's ``class_act`` is a plain sum over the whole training run,
        so its prototypes average the network at every stage of training —
        including the random early epochs.  A class enrolled later (from
        final-model activations) has no such handicap, sits systematically
        closer to every final-model activation, and swallows the stale
        classes' images.  Building (or refreshing) all prototypes from one
        post-training pass puts every class on equal footing — essential for
        interference-free few-shot enrollment.  Still gradient-free:
        prototypes remain per-class means of L2-normalized activations.
        """
        a_hat = _normalize(activations)
        units = a_hat.shape[1]
        means = torch.zeros(len(class_names), units)
        counts = torch.zeros(len(class_names))
        for i in range(len(class_names)):
            mask = labels == i
            counts[i] = float(mask.sum())
            if counts[i] > 0:
                means[i] = a_hat[mask].mean(dim=0)
        return cls(layer, class_names, means, counts, temperature)

    # ------------------------------------------------------------ inference

    def logits(self, activations: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled cosine similarity to each prototype ``[n, C]``."""
        a_hat = _normalize(activations)
        return (a_hat @ self.prototypes.t()) / self.temperature

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        return self.logits(activations).argmax(dim=1)

    def predict_names(self, activations: torch.Tensor) -> List[str]:
        return [self.class_names[i] for i in self.predict(activations).tolist()]

    # -------------------------------------------------------------- few-shot

    def enroll(self, name: str, activations: torch.Tensor) -> int:
        """Add or update a class prototype from activation vectors — no grads.

        ``activations`` is ``[k, U]`` (pooled/ReLU'd/tracked, unnormalized).
        Each row is L2-normalized, then folded into a running mean so repeated
        enrollment of the same class accumulates evidence.  Returns the class
        index.  A name not seen before extends the classifier with a new
        class, which is how a backbone-unseen category is taught few-shot.
        """
        a_hat = _normalize(activations)
        add_sum = a_hat.sum(dim=0)
        add_n = float(a_hat.shape[0])
        if name in self.class_names:
            idx = self.class_names.index(name)
            prev_n = float(self._counts[idx])
            self._means[idx] = (self._means[idx] * prev_n + add_sum) / max(prev_n + add_n, 1e-8)
            self._counts[idx] = prev_n + add_n
        else:
            idx = len(self.class_names)
            self.class_names.append(name)
            self._means = torch.cat([self._means, (add_sum / max(add_n, 1e-8))[None]], dim=0)
            self._counts = torch.cat([self._counts, torch.tensor([add_n])])
        self.prototypes = _normalize(self._means)
        return idx


# --------------------------------------------------------------- tree router


class TreeRoutedHead:
    """Route an image through a :class:`ConceptNode` tree to classify it.

    At every internal node each child is scored by the mean activation of its
    member units on the L2-normalized activation vector, optionally divided by
    the child's ``importance`` prior (so large/hot sub-concepts do not always
    win the split).  Two modes:

    * ``"hard"`` — descend the single best child at each node; predict the
      reached leaf's class-affinity argmax.
    * ``"soft"`` — at each node the sibling scores are softmaxed; a leaf's
      weight is the product of these probabilities along its root path; the
      class distribution is the affinity-weighted sum over all leaves.

    Soft routing is the default (smoother, and it uses every leaf), but both
    are exposed via ``mode``.
    """

    def __init__(
        self,
        tree: ConceptNode,
        class_names: Sequence[str],
        temperature: float = 0.25,
        use_importance_prior: bool = True,
        mode: str = "soft",
    ) -> None:
        self.tree = tree
        self.class_names = list(class_names)
        self.temperature = float(temperature)
        self.use_importance_prior = use_importance_prior
        self.mode = mode
        self._aff_cache: Dict[str, torch.Tensor] = {}

    @classmethod
    def from_tree(
        cls,
        tree: ConceptNode,
        class_names: Sequence[str],
        **kwargs,
    ) -> "TreeRoutedHead":
        return cls(tree, class_names, **kwargs)

    # ---------------------------------------------------------- node scoring

    def _affinity_vec(self, node: ConceptNode) -> torch.Tensor:
        if node.node_id not in self._aff_cache:
            v = torch.zeros(len(self.class_names))
            for name, score in node.class_affinity.items():
                if name in self.class_names:
                    v[self.class_names.index(name)] = score
            self._aff_cache[node.node_id] = v
        return self._aff_cache[node.node_id]

    @staticmethod
    def _score(node: ConceptNode, a_hat: torch.Tensor) -> float:
        if not node.units:
            return 0.0
        return float(a_hat[node.units].mean())

    def _child_scores(self, node: ConceptNode, a_hat: torch.Tensor) -> List[float]:
        scores = []
        for child in node.children:
            s = self._score(child, a_hat)
            if self.use_importance_prior:
                s = s / (child.importance + 1e-8)
            scores.append(s)
        return scores

    # ------------------------------------------------------------- routing

    def decision_path(self, activation: torch.Tensor) -> List[str]:
        """Hard-route ``activation`` ([U]) and return the node ids visited."""
        a_hat = _normalize(activation[None])[0]
        node = self.tree
        path = [node.node_id]
        while node.children:
            scores = self._child_scores(node, a_hat)
            node = node.children[int(np.argmax(scores))]
            path.append(node.node_id)
        return path

    def _hard_dist(self, a_hat: torch.Tensor) -> torch.Tensor:
        node = self.tree
        while node.children:
            scores = self._child_scores(node, a_hat)
            node = node.children[int(np.argmax(scores))]
        return self._affinity_vec(node)

    def _soft_dist(self, a_hat: torch.Tensor) -> torch.Tensor:
        dist = torch.zeros(len(self.class_names))

        def descend(node: ConceptNode, weight: float) -> None:
            if not node.children:
                dist.add_(self._affinity_vec(node) * weight)
                return
            scores = torch.tensor(self._child_scores(node, a_hat))
            probs = torch.softmax(scores / self.temperature, dim=0)
            for child, p in zip(node.children, probs):
                descend(child, weight * float(p))

        descend(self.tree, 1.0)
        return dist

    def logits(self, activations: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        """Per-image class distribution ``[n, C]`` under the routing mode."""
        mode = mode or self.mode
        a_hats = _normalize(activations)
        rows = []
        for i in range(a_hats.shape[0]):
            if mode == "hard":
                rows.append(self._hard_dist(a_hats[i]))
            else:
                rows.append(self._soft_dist(a_hats[i]))
        return torch.stack(rows, dim=0)

    def predict(self, activations: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        return self.logits(activations, mode=mode).argmax(dim=1)


# ------------------------------------------------------------ concept bottleneck


class ConceptBottleneckHead:
    """Flat concepts → concept scores → classes, gradient-free.

    The interpretable pipeline: cluster the units into concepts
    (:func:`cluster_concepts`), score each concept per image
    (:func:`concept_scores`), then map concept scores to classes through the
    concepts' class-affinity matrix (each concept votes for the classes its
    units fire on).  With ``fit_logistic=True`` a scikit-learn
    ``LogisticRegression`` is fitted on training concept scores as a small
    supervised *upper bound* on what the same bottleneck can achieve.
    """

    def __init__(
        self,
        layer: str,
        class_names: Sequence[str],
        concepts: Sequence[Concept],
        affinity_matrix: torch.Tensor,   # [n_concepts, C]
    ) -> None:
        self.layer = layer
        self.class_names = list(class_names)
        self.concepts = list(concepts)
        self.affinity_matrix = affinity_matrix.float()
        self._logreg = None

    @classmethod
    def from_memory(
        cls,
        memory: HebbianFeatureMemory,
        layer: str,
        class_names: Sequence[str],
        n_concepts: int = 12,
        min_units: int = 2,
        activity_threshold: float = 0.02,
    ) -> "ConceptBottleneckHead":
        concepts = cluster_concepts(
            memory, layer, class_names, n_concepts=n_concepts,
            min_units=min_units, activity_threshold=activity_threshold,
        )
        affinity = memory.class_affinity(layer).numpy()   # [C, U]
        rows = []
        for c in concepts:
            scores = affinity[:, c.units].mean(axis=1)
            peak = scores.max()
            rows.append(scores / peak if peak > 0 else scores)
        mat = torch.tensor(np.stack(rows), dtype=torch.float32) if rows else torch.zeros(0, len(class_names))
        return cls(layer, class_names, concepts, mat)

    def _scores(self, activations: torch.Tensor) -> torch.Tensor:
        a_hat = _normalize(activations)
        return concept_scores(self.concepts, {self.layer: a_hat})

    def logits(self, activations: torch.Tensor) -> torch.Tensor:
        """Affinity-weighted class scores ``[n, C]``."""
        cs = self._scores(activations)
        return cs @ self.affinity_matrix

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        return self.logits(activations).argmax(dim=1)

    def fit_logistic(self, activations: torch.Tensor, labels: torch.Tensor) -> "ConceptBottleneckHead":
        """Fit a logistic regression on concept scores (supervised upper bound)."""
        from sklearn.linear_model import LogisticRegression

        cs = self._scores(activations).numpy()
        y = labels.numpy()
        self._logreg = LogisticRegression(max_iter=1000)
        self._logreg.fit(cs, y)
        return self

    def predict_logistic(self, activations: torch.Tensor) -> torch.Tensor:
        if self._logreg is None:
            raise RuntimeError("call fit_logistic() first")
        cs = self._scores(activations).numpy()
        return torch.tensor(self._logreg.predict(cs))
