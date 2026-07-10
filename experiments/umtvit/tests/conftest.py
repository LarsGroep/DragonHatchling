"""Shared pytest configuration for the UMT-ViT test suite.

Puts the package root (``experiments/umtvit``) on ``sys.path`` so the tests
import ``umtvit`` whether or not the package is pip-installed. All tests are
CPU-only, require no downloads, and use ``tmp_path`` for any file output.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))
