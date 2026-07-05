"""Pixel identity for concept-tree nodes — "discard the name, show pixels".

A concept node is normally described by the classes it fires on.  This module
instead produces the node's *visual* identity: for each node it finds the
probe images that activate it most, runs a coarse occlusion search to locate
the image region the node responds to, crops that region, and PNG-encodes it
as a base64 ``data:`` URI.  The web app can then render the tree with little
image thumbnails at every node instead of (or alongside) text labels.

The occlusion search slides a neutral (dataset-mean) square over an ``S×S``
grid and measures how much the node's score drops; the cell whose occlusion
hurts the score most is the node's feature area.  All ``S²`` occluded copies
of an image go through the model in a single forward pass, so the whole thing
stays cheap even on CPU.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from hatchvision.explain.concepts import probe_activations
from hatchvision.explain.gradcam import denormalize
from hatchvision.hebbian.hierarchy import ConceptNode, node_scores
from hatchvision.hebbian.memory import HebbianFeatureMemory


def _score_images(
    model,
    memory: HebbianFeatureMemory,
    images: torch.Tensor,
    layer: str,
    units: Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    """Node score (mean member-unit activation of the L2-normalized vector)."""
    acts = probe_activations(model, images, memory=memory, batch_size=batch_size)[layer]
    a_hat = acts.float() / (acts.float().norm(dim=1, keepdim=True) + 1e-8)
    if not units:
        return torch.zeros(a_hat.shape[0])
    return a_hat[:, units].mean(dim=1)


def _occluded_batch(
    image: torch.Tensor, grid: int, half: int
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
    """``grid²`` copies of ``image``, each with one cell zeroed (dataset mean).

    Returns the batch and the pixel bounding box ``(y0, x0, y1, x1)`` occluded
    in each copy, in the same order.
    """
    c, h, w = image.shape
    copies, boxes = [], []
    for gy in range(grid):
        cy = int((gy + 0.5) * h / grid)
        for gx in range(grid):
            cx = int((gx + 0.5) * w / grid)
            y0, y1 = max(0, cy - half), min(h, cy + half)
            x0, x1 = max(0, cx - half), min(w, cx + half)
            occ = image.clone()
            occ[:, y0:y1, x0:x1] = 0.0        # 0 in normalized space = mean
            copies.append(occ)
            boxes.append((y0, x0, y1, x1))
    return torch.stack(copies), boxes


def _crop_to_datauri(
    image: torch.Tensor,
    box: Tuple[int, int, int, int],
    mean: Sequence[float],
    std: Sequence[float],
    out_size: int,
) -> str:
    from PIL import Image

    y0, x0, y1, x1 = box
    denorm = denormalize(image[None], mean, std)[0]         # [C, H, W] in [0,1]
    crop = denorm[:, y0:y1, x0:x1]
    if crop.shape[1] < 2 or crop.shape[2] < 2:              # degenerate box
        crop = denorm
    arr = (crop.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    if arr.shape[2] == 1:
        arr = arr.repeat(3, axis=2)
    pil = Image.fromarray(arr, mode="RGB").resize((out_size, out_size), Image.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@torch.no_grad()
def node_patch_uris(
    tree: ConceptNode,
    model,
    memory: HebbianFeatureMemory,
    probe_images: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    exemplars: int = 3,
    grid: int = 7,
    patch_frac: float = 0.28,
    out_size: int = 56,
    max_depth: Optional[int] = None,
    batch_size: int = 64,
) -> Dict[str, List[str]]:
    """Return ``{node_id: [data_uri, ...]}`` — the pixel identity of each node.

    For every node (down to ``max_depth`` if given) the top ``exemplars``
    probe images by node score are selected; each is occlusion-searched on an
    ``grid×grid`` grid, and the most important region is cropped, resized to
    ``out_size`` and base64 PNG-encoded.  Budgets are deliberately small:
    3 exemplars/node × one batched forward per exemplar.
    """
    layer = tree.layer
    scores = node_scores(tree, probe_activations(model, probe_images, memory=memory, batch_size=batch_size))
    half = max(1, int(round(probe_images.shape[-1] * patch_frac / 2)))

    result: Dict[str, List[str]] = {}
    for node in tree.walk():
        if max_depth is not None and node.depth > max_depth:
            continue
        sc = scores[node.node_id]
        k = min(exemplars, sc.shape[0])
        if k == 0:
            result[node.node_id] = []
            continue
        top_idx = sc.topk(k).indices.tolist()
        uris: List[str] = []
        for img_i in top_idx:
            image = probe_images[img_i]
            occ, boxes = _occluded_batch(image, grid, half)
            base = _score_images(model, memory, image[None], layer, node.units, batch_size)[0]
            occ_scores = _score_images(model, memory, occ, layer, node.units, batch_size)
            drop = base - occ_scores
            best = int(drop.argmax())
            uris.append(_crop_to_datauri(image, boxes[best], mean, std, out_size))
        result[node.node_id] = uris
    return result


def attach_patches(tree: ConceptNode, patches: Dict[str, List[str]]) -> ConceptNode:
    """Attach ``node_patch_uris`` output onto the tree's nodes in place."""
    for node in tree.walk():
        node.patches = patches.get(node.node_id, [])
    return tree
