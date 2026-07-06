"""PackWriter — assembles Explanation Pack directories (§5, §6).

M0 ships the interface only. The single ``PackWriter`` enforces the schema and
the quantization rules for every method's output; the implementation lands at
M2 when the pack format is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .manifest import AssetEntry, PackManifest


@dataclass
class PackWriter:
    """Accumulates assets and emits a schema-valid pack directory.

    Parameters
    ----------
    out_dir:
        Destination directory for the pack (``manifest.json`` + binaries).
    """

    out_dir: Path
    _assets: Dict[str, AssetEntry] = field(default_factory=dict, init=False)

    def add_array(
        self,
        filename: str,
        array: "np.ndarray",
        *,
        encoding: str = "raw",
    ) -> AssetEntry:
        """Register and serialize a numpy array asset. Not implemented at M0."""
        raise NotImplementedError("PackWriter.add_array lands at M2 (pack freeze)")

    def add_json(self, filename: str, payload: Dict[str, Any]) -> AssetEntry:
        """Register and serialize a JSON asset. Not implemented at M0."""
        raise NotImplementedError("PackWriter.add_json lands at M2 (pack freeze)")

    def write_manifest(self, manifest: PackManifest) -> Path:
        """Validate and write ``manifest.json``. Not implemented at M0."""
        raise NotImplementedError(
            "PackWriter.write_manifest lands at M2 (pack freeze)"
        )


__all__ = ["PackWriter"]
