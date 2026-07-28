"""Real-analyzer wiring test (§14) — torch-free.

``test_health.py`` injects a fake analyzer via ``set_analyzer``, so the default
path (``app.main._default_analyzer``) is never exercised there. That is how a
typo on a real ``vitreous`` type survived: the fine-tuned-checkpoint branch
read ``loaded.model`` while :class:`vitreous.models.LoadedModel` exposes the
module as ``.module``, so every deployment that sets ``VITREOUS_WEIGHTS``
raised ``AttributeError`` on its first analyze job.

This test drives the default analyzer end to end through the HTTP + SSE flow
with ``VITREOUS_WEIGHTS`` pointing at a real file. The heavy dependencies
(torch, PIL) are stubbed into ``sys.modules`` and ``load_model`` is
monkeypatched to return a genuine ``LoadedModel`` — genuine so an attribute
that does not exist on the real dataclass still fails, which is exactly the
bug we are pinning.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

# vitreous (packages/core) is a hard runtime dep of the service — importing it
# here is deliberate: none of these names pull in torch (M0 guarantee).
from vitreous import data as vitreous_data
from vitreous import models as vitreous_models
from vitreous import packs as vitreous_packs

from app import main as m

client = TestClient(m.app)


class _FakeModule:
    """Stand-in for the torch module held by ``LoadedModel.module``."""

    def __init__(self) -> None:
        self.loaded_state = None

    def load_state_dict(self, state):
        self.loaded_state = state


class _FakeTensor:
    def unsqueeze(self, dim):
        return self


class _FakeImage:
    width = 224
    height = 224

    def convert(self, mode):
        return self


def _install_stub(monkeypatch, name: str, mod: types.ModuleType) -> None:
    monkeypatch.setitem(sys.modules, name, mod)


def test_default_analyzer_applies_finetuned_checkpoint(tmp_path, monkeypatch):
    """VITREOUS_WEIGHTS must load into LoadedModel.module (not .model)."""
    # --- stub the heavy imports made inside _default_analyzer ---------------- #
    fake_module = _FakeModule()
    state = {"head.weight": "sentinel"}

    torch_stub = types.ModuleType("torch")
    torch_stub.load = lambda path, map_location=None: state
    _install_stub(monkeypatch, "torch", torch_stub)

    image_stub = types.ModuleType("PIL.Image")
    image_stub.open = lambda fh: _FakeImage()
    pil_stub = types.ModuleType("PIL")
    pil_stub.Image = image_stub
    _install_stub(monkeypatch, "PIL", pil_stub)
    _install_stub(monkeypatch, "PIL.Image", image_stub)

    # --- stub the vitreous entry points, keeping the REAL LoadedModel ------- #
    spec = vitreous_data.get_dataset("eurosat").spec

    class _FakeAdapter:
        spec = None  # set below

        def preprocess(self):
            return lambda img: _FakeTensor()

    _FakeAdapter.spec = spec

    loaded = vitreous_models.LoadedModel(
        spec=vitreous_models.get_model_spec("vit_s16"),
        module=fake_module,
        num_classes=spec.num_classes,
    )

    monkeypatch.setattr(vitreous_data, "get_dataset", lambda name: _FakeAdapter)
    monkeypatch.setattr(vitreous_models, "load_model", lambda *a, **k: loaded)

    built = {}

    def _fake_build_pack(model, tensor, *, image_meta, dataset_spec, out_dir, **kw):
        built["image_meta"] = image_meta
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "manifest.json").write_text(json.dumps({"ok": True}))

    monkeypatch.setattr(vitreous_packs, "build_pack", _fake_build_pack)

    ckpt = tmp_path / "finetuned.pt"
    ckpt.write_bytes(b"not-really-a-checkpoint")
    monkeypatch.setenv("VITREOUS_WEIGHTS", str(ckpt))
    monkeypatch.setenv("VITREOUS_MODEL", "vit_s16")

    # _loaded is module-global state; restore it so other tests see a cold app.
    monkeypatch.setitem(m._loaded, "model", None)
    monkeypatch.setitem(m._loaded, "dataset", None)

    # --- drive the default analyzer through the real §14 flow ---------------- #
    assert m._analyzer is None, "default analyzer must be active for this test"
    r = client.post(
        "/analyze",
        files={"image": ("x.png", b"\x89PNG fake", "image/png")},
        params={"dataset": "eurosat"},
    )
    assert r.status_code == 200
    job = r.json()["job_id"]

    with client.stream("GET", f"/jobs/{job}/events") as s:
        events = [
            json.loads(line[len("data: "):])
            for line in s.iter_lines()
            if line.startswith("data: ")
        ]

    assert events[-1]["stage"] == "done", f"analyzer failed: {events[-1]}"
    # The checkpoint reached the module — this is the assertion that fails with
    # AttributeError ("'LoadedModel' object has no attribute 'model'") on the bug.
    assert fake_module.loaded_state is state
    assert [e["stage"] for e in events].count("predict") == 1
    assert built["image_meta"]["source"] == "upload"
    assert m._loaded == {"model": loaded.spec.arch, "dataset": "eurosat"}
    assert client.get(f"/jobs/{job}/pack/manifest.json").json() == {"ok": True}
