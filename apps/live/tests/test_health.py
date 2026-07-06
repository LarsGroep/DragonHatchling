"""Health endpoint test (§14) using FastAPI TestClient."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "model": None,
        "dataset": None,
        "warm": True,
    }


def test_analyze_flow_with_fake_analyzer(tmp_path):
    """§14 flow without torch: upload -> SSE stages -> pack asset served."""
    import json as _json
    from app import main as m

    def fake(data, dataset, out, progress):
        progress("predict")
        progress("attention")
        (out / "manifest.json").write_text(_json.dumps({"ok": True, "dataset": dataset}))

    m.set_analyzer(fake)
    try:
        r = client.post("/analyze", files={"image": ("x.png", b"\x89PNG fake", "image/png")},
                        params={"dataset": "eurosat"})
        assert r.status_code == 200
        job = r.json()["job_id"]
        with client.stream("GET", f"/jobs/{job}/events") as s:
            stages = [
                _json.loads(line[len("data: "):])["stage"]
                for line in s.iter_lines() if line.startswith("data: ")
            ]
        assert stages[-1] == "done" and "attention" in stages
        a = client.get(f"/jobs/{job}/pack/manifest.json")
        assert a.status_code == 200 and a.json()["dataset"] == "eurosat"
        assert client.get(f"/jobs/{job}/pack/../etc/passwd").status_code in (404, 400)
        assert client.post("/analyze", files={"image": ("e.png", b"", "image/png")}).status_code == 400
    finally:
        m.set_analyzer(None)
