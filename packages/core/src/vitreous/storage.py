"""Storage abstraction (§2, §15).

Packs, projections, and concept artifacts live behind a :class:`StorageAdapter`
so the compute venue and the storage backend are decoupled: Supabase Storage is
the v1 primary, with a public HF-dataset repo as the overflow valve — same
interface. M0 ships the Protocol; concrete adapters land at M4.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class StorageAdapter(Protocol):
    """Read/write blobs by key against a $0-tier storage backend (§15)."""

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        """Store ``data`` under ``key``; return the public URL."""
        ...

    def get(self, key: str) -> bytes:
        """Fetch the full blob at ``key``."""
        ...

    def get_range(self, key: str, start: int, end: int) -> bytes:
        """Fetch bytes ``[start, end)`` — HTTP range read for per-view streaming."""
        ...

    def url(self, key: str) -> str:
        """Return the public-read URL for ``key`` (no fetch)."""
        ...

    def list(self, prefix: str) -> Iterable[str]:
        """List keys under ``prefix``."""
        ...

    def exists(self, key: str) -> bool:
        """Return whether ``key`` exists."""
        ...


__all__ = ["StorageAdapter"]
