"""Registry behavior — the one piece of vitreous.data that is live at M0."""

from __future__ import annotations

import pytest

from vitreous.data import (
    DatasetAdapter,
    DatasetSpec,
    SplitPolicy,
    VizHooks,
    get_dataset,
    list_datasets,
    register_dataset,
)
from vitreous.data import _clear_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate each test from global registry state."""
    _clear_registry()
    yield
    _clear_registry()


def _make_adapter():
    class _Dummy(DatasetAdapter):
        spec = DatasetSpec(
            name="dummy", display_name="Dummy", num_classes=2,
            class_names=["a", "b"],
        )

        def load(self, root, split):
            return []

        def preprocess(self):
            return lambda x: x

        def augment(self):
            return lambda x: x

        def splits(self):
            return SplitPolicy()

        def gallery(self, n=75):
            return []

        def viz_hooks(self):
            return VizHooks()

    return _Dummy


def test_register_and_get_roundtrip():
    Dummy = register_dataset("dummy")(_make_adapter())
    assert get_dataset("dummy") is Dummy
    assert "dummy" in list_datasets()


def test_get_unknown_raises_keyerror_with_available_list():
    register_dataset("dummy")(_make_adapter())
    with pytest.raises(KeyError) as exc:
        get_dataset("nope")
    assert "dummy" in str(exc.value)


def test_duplicate_registration_raises():
    register_dataset("dummy")(_make_adapter())
    with pytest.raises(ValueError):
        register_dataset("dummy")(_make_adapter())


def test_empty_name_raises():
    with pytest.raises(ValueError):
        register_dataset("")


def test_register_non_adapter_raises():
    with pytest.raises(TypeError):
        register_dataset("bad")(object)  # type: ignore[arg-type]


def test_list_datasets_sorted():
    register_dataset("zeta")(_make_adapter())
    register_dataset("alpha")(_make_adapter())
    assert list_datasets() == ["alpha", "zeta"]


def test_spec_class_names_length_mismatch_raises():
    with pytest.raises(ValueError):
        DatasetSpec(
            name="x", display_name="X", num_classes=3, class_names=["a", "b"]
        )
