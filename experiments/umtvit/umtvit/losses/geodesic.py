"""Ablation-gated geodesic loss over a mini-batch k-NN graph (ARCHITECTURE §3.6).

Builds a small graph whose nodes are the two views' mean embeddings plus a
capped sample of voxel features, connects each node to its ``k`` nearest
neighbours, and takes the Dijkstra shortest path between the two view nodes as a
learned geodesic distance ``L_geo = D_g(z_a, z_b)``. Gradients flow through the
edge *lengths* on the chosen path, while the path itself (which neighbours, in
which order) is selected on the **detached** distance matrix — so the discrete
routing is not differentiated, only the continuous edge lengths are.

Speculative and expensive with a degenerate optimum (RESEARCH §6), so it is
gated to weight ``0`` by default and the trainer only ever calls this function
when ``loss.lambda_geodesic > 0`` — off means zero overhead, never computed.
Ported from the notebook reference's ``geodesic_loss``.
"""

from __future__ import annotations

import heapq

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["geodesic_loss"]


def geodesic_loss(v_sub: Tensor, za: Tensor, zb: Tensor, k: int = 6) -> Tensor:
    """k-NN-graph geodesic distance between the two views' mean embeddings.

    Args:
        v_sub: Sampled voxel features ``[M, C]`` (only the first ``proj_dim``
            channels and first 256 rows are used as graph nodes).
        za: View-A projected embeddings ``[B, proj_dim]``.
        zb: View-B projected embeddings ``[B, proj_dim]``.
        k: Neighbours per node in the k-NN graph.

    Returns:
        Scalar geodesic distance along the (detached-path) shortest route from
        node ``0`` (view-A mean) to node ``1`` (view-B mean). Falls back to the
        direct edge ``d[0, 1]`` when the target is unreachable.

    Shape:
        - Output: scalar.
    """
    nodes = torch.cat(
        [
            F.normalize(za, dim=1).mean(0, keepdim=True),
            F.normalize(zb, dim=1).mean(0, keepdim=True),
            F.normalize(v_sub[:256, : za.shape[1]], dim=1),
        ]
    )
    d = torch.cdist(nodes, nodes)
    knn = d.detach().topk(min(k + 1, len(nodes)), largest=False).indices[:, 1:]

    dist = {0: 0.0}
    prev: dict[int, int] = {}
    pq: list[tuple[float, int]] = [(0.0, 0)]
    seen: set[int] = set()
    while pq:
        du, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == 1:
            break
        for vtx in knn[u].tolist():
            alt = du + d[u, vtx].item()
            if alt < dist.get(vtx, float("inf")):
                dist[vtx] = alt
                prev[vtx] = u
                heapq.heappush(pq, (alt, vtx))

    if 1 not in prev:
        return d[0, 1]
    loss = d.new_zeros(())
    node = 1
    while node != 0:
        loss = loss + d[prev[node], node]
        node = prev[node]
    return loss
