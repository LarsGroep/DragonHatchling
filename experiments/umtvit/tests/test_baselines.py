"""Read-out baselines — the floor a probe has to clear (umtvit.eval.baselines).

These tests load the module **by file path** rather than via ``umtvit.eval``,
because that package's ``__init__`` imports torch. ``baselines.py`` is pure
stdlib on purpose, so this file runs in any environment — unlike the rest of
this suite, which needs the ML stack and is therefore out of CI scope.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "umtvit" / "eval" / "baselines.py"


def _load():
    spec = importlib.util.spec_from_file_location("umtvit_eval_baselines", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bl = _load()

# Published HAM10000 class counts (Tschandl et al. 2018), scaled 1:10 to keep
# the fixture small. Proportions — and therefore every baseline — are identical.
_HAM10000_COUNTS = {"nv": 670, "mel": 111, "bkl": 110, "bcc": 51, "akiec": 33, "vasc": 14, "df": 12}
_NAMES = sorted(_HAM10000_COUNTS)
_NV = _NAMES.index("nv")


def _ham10000_labels():
    labels = []
    for name in _NAMES:
        labels += [_NAMES.index(name)] * _HAM10000_COUNTS[name]
    return labels


def test_majority_baseline_is_the_real_floor():
    labels = _ham10000_labels()
    majority = bl.majority_baseline(labels, 7)
    assert majority == pytest.approx(0.669, abs=0.002)
    # The number this repo used to quote instead.
    assert 1.0 / 7 == pytest.approx(0.1429, abs=1e-4)
    # ...which understates the floor by a factor of ~4.7.
    assert majority / (1.0 / 7) > 4.5


def test_always_predict_nv_scores_majority_but_chance_balanced_accuracy():
    """The exact trap: a model that learned nothing looks good on accuracy."""
    labels = _ham10000_labels()
    per_class = {i: (1.0 if i == _NV else 0.0) for i in range(7)}
    block = bl.baseline_block(labels, 7, per_class_recall=per_class)

    assert block["majority_baseline"] == pytest.approx(0.669, abs=0.002)
    # Balanced accuracy cannot be gamed by predicting the majority class.
    assert block["balanced_accuracy"] == pytest.approx(1.0 / 7)
    assert block["imbalanced"] is True
    assert block["meaningful_floor"] == "majority_baseline"


def test_recorded_probe_result_read_against_the_right_floor():
    """SGP run 3's 0.7922 is ~+12 pp over majority, not ~5.5x chance."""
    labels = _ham10000_labels()
    majority = bl.majority_baseline(labels, 7)
    probe = 0.7922
    assert probe - majority == pytest.approx(0.123, abs=0.005)
    assert probe / (1.0 / 7) > 5.0  # the flattering framing, for contrast


def test_balanced_split_is_not_flagged_imbalanced():
    block = bl.baseline_block([0, 1, 2, 0, 1, 2], 3)
    assert block["imbalanced"] is False
    assert block["meaningful_floor"] == "chance"
    assert block["majority_baseline"] == pytest.approx(1.0 / 3)


def test_empty_split_has_no_defined_floor():
    assert bl.majority_baseline([], 5) is None
    block = bl.baseline_block([], 5)
    assert block["majority_baseline"] is None
    assert block["imbalanced"] is False


def test_out_of_range_labels_are_ignored():
    # -1 / 99 are not valid class ids and must not inflate the denominator.
    assert bl.majority_baseline([0, 0, 1, -1, 99], 2) == pytest.approx(2 / 3)


def test_balanced_accuracy_skips_absent_classes():
    """_per_class_accuracy emits nan for a class with no support."""
    assert bl.balanced_accuracy({0: float("nan"), 1: 0.5, 2: 0.7}) == pytest.approx(0.6)
    assert bl.balanced_accuracy({0: float("nan")}) is None
    assert bl.balanced_accuracy({}) is None


def test_balanced_accuracy_never_returns_nan():
    for mapping in ({0: float("nan")}, {}, {0: 0.0, 1: float("nan")}):
        result = bl.balanced_accuracy(mapping)
        assert result is None or not math.isnan(result)


def test_baseline_block_omits_balanced_accuracy_when_not_supplied():
    assert "balanced_accuracy" not in bl.baseline_block([0, 1], 2)
