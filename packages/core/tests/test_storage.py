"""StorageAdapter backends (§15) — offline.

LocalStorageAdapter round-trips put/get/exists and put_pack. The Supabase and
HF adapters are *constructed* without any network I/O and raise clear errors
when their env credentials are missing; their get_url is a pure string
computation (no fetch). Real uploads are never exercised here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vitreous.storage import (
    HFDatasetStorageAdapter,
    LocalStorageAdapter,
    StorageAdapter,
    SupabaseStorageAdapter,
    get_storage,
)

SUPA_ENV = ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "SUPABASE_ANON_KEY")
HF_ENV = ("HF_TOKEN", "HUGGINGFACE_TOKEN")


@pytest.fixture
def clean_env(monkeypatch):
    for k in SUPA_ENV + HF_ENV + ("VITREOUS_STORAGE",):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# --------------------------------------------------------------------------- #
# Local
# --------------------------------------------------------------------------- #


def test_local_put_get_exists_roundtrip(tmp_path):
    st = LocalStorageAdapter(tmp_path / "store")
    src = tmp_path / "a.bin"
    src.write_bytes(b"hello-vitreous")

    url = st.put_file(src, "packs/d/img/a.bin")
    assert url.startswith("file://")
    assert st.exists("packs/d/img/a.bin")
    assert not st.exists("packs/d/img/missing.bin")
    assert st.get_bytes("packs/d/img/a.bin") == b"hello-vitreous"
    # protocol-conformant
    assert isinstance(st, StorageAdapter)


def test_local_put_pack_uploads_all_files(tmp_path):
    pack = tmp_path / "pack"
    (pack / "sub").mkdir(parents=True)
    (pack / "manifest.json").write_text("{}")
    (pack / "attention.bin").write_bytes(b"\x00\x01")
    (pack / "sub" / "graph.json").write_text("[]")

    st = LocalStorageAdapter(tmp_path / "store")
    urls = st.put_pack(pack, "packs/eurosat/img_0")
    assert set(urls) == {"manifest.json", "attention.bin", "sub/graph.json"}
    assert st.exists("packs/eurosat/img_0/manifest.json")
    assert st.exists("packs/eurosat/img_0/sub/graph.json")
    listed = st.list("packs/eurosat/img_0")
    assert "packs/eurosat/img_0/attention.bin" in listed


def test_local_get_url_no_network(tmp_path):
    st = LocalStorageAdapter(tmp_path / "s")
    url = st.get_url("x/y.json")  # no file needed, pure path->uri
    assert url.startswith("file://") and url.endswith("x/y.json")


# --------------------------------------------------------------------------- #
# Supabase — construct offline, clear errors, pure URL
# --------------------------------------------------------------------------- #


def test_supabase_missing_url_raises(clean_env):
    with pytest.raises(EnvironmentError) as exc:
        SupabaseStorageAdapter()
    assert "SUPABASE_URL" in str(exc.value)


def test_supabase_missing_key_raises(clean_env):
    clean_env.setenv("SUPABASE_URL", "https://ref.supabase.co")
    with pytest.raises(EnvironmentError) as exc:
        SupabaseStorageAdapter()
    assert "key" in str(exc.value).lower()


def test_supabase_constructs_with_env_and_builds_public_url(clean_env):
    clean_env.setenv("SUPABASE_URL", "https://ref.supabase.co/")
    clean_env.setenv("SUPABASE_SERVICE_KEY", "svc-key")
    st = SupabaseStorageAdapter(bucket="packs")
    # get_url is a pure computation — no network.
    assert st.get_url("packs/eurosat/img/manifest.json") == (
        "https://ref.supabase.co/storage/v1/object/public/packs/"
        "packs/eurosat/img/manifest.json"
    )
    assert st.kind == "supabase"
    assert isinstance(st, StorageAdapter)


def test_supabase_anon_key_fallback(clean_env):
    clean_env.setenv("SUPABASE_URL", "https://ref.supabase.co")
    clean_env.setenv("SUPABASE_ANON_KEY", "anon-key")
    st = SupabaseStorageAdapter()
    assert st.key == "anon-key"


# --------------------------------------------------------------------------- #
# HF dataset — construct offline, clear errors, pure URL
# --------------------------------------------------------------------------- #


def test_hf_missing_token_raises(clean_env):
    with pytest.raises(EnvironmentError) as exc:
        HFDatasetStorageAdapter("user/vitreous-packs")
    assert "HF_TOKEN" in str(exc.value)


def test_hf_constructs_with_env_and_builds_resolve_url(clean_env):
    clean_env.setenv("HF_TOKEN", "hf-secret")
    st = HFDatasetStorageAdapter("user/vitreous-packs")
    assert st.get_url("packs/eurosat/img/manifest.json") == (
        "https://huggingface.co/datasets/user/vitreous-packs/resolve/main/"
        "packs/eurosat/img/manifest.json"
    )
    assert st.kind == "hf_dataset"
    assert isinstance(st, StorageAdapter)


def test_hf_requires_repo_id(clean_env):
    clean_env.setenv("HF_TOKEN", "hf-secret")
    with pytest.raises(ValueError):
        HFDatasetStorageAdapter("")


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #


def test_get_storage_defaults_to_local(clean_env, tmp_path):
    clean_env.setenv("VITREOUS_STORAGE_ROOT", str(tmp_path / "s"))
    st = get_storage()
    assert st.kind == "local"


def test_get_storage_selects_supabase(clean_env):
    clean_env.setenv("SUPABASE_URL", "https://ref.supabase.co")
    clean_env.setenv("SUPABASE_SERVICE_KEY", "svc")
    st = get_storage("supabase")
    assert st.kind == "supabase"


def test_get_storage_unknown_kind_raises(clean_env):
    with pytest.raises(ValueError):
        get_storage("s3")
