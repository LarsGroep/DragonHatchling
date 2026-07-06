"""Storage abstraction (§2, §15) — decouple the compute venue from the backend.

Packs, projections, and concept artifacts live behind a :class:`StorageAdapter`
so that Kaggle batch, the live CPU service, and offline tests all upload through
one interface. Backends:

* :class:`LocalStorageAdapter` — writes to a local dir; ``get_url`` returns a
  ``file://`` URL. Offline tests and dry-runs use this one.
* :class:`SupabaseStorageAdapter` — the v1 primary. Public-read Supabase Storage
  buckets over the Storage REST API (via ``httpx``; the ``supabase`` client is
  used instead when installed). Credentials come from ``SUPABASE_URL`` +
  ``SUPABASE_SERVICE_KEY`` / ``SUPABASE_ANON_KEY`` — **read from ``os.environ``,
  never hardcoded**.
* :class:`HFDatasetStorageAdapter` — the overflow valve. Uploads to a public HF
  dataset repo via ``huggingface_hub``; token from ``HF_TOKEN``.

Import & network discipline
---------------------------
No network calls and no heavy/network imports happen at module import: ``httpx``,
``supabase`` and ``huggingface_hub`` are imported lazily inside the methods that
touch the network. Constructors only read env vars (and raise a clear error when
a required credential is missing), so adapters can be *constructed* offline.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

_DEFAULT_CONTENT_TYPES = {
    ".json": "application/json",
    ".bin": "application/octet-stream",
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".joblib": "application/octet-stream",
    ".sql": "application/sql",
    ".txt": "text/plain",
}


def _content_type_for(name: str) -> str:
    return _DEFAULT_CONTENT_TYPES.get(Path(name).suffix.lower(), "application/octet-stream")


def _iter_pack_files(local_dir: Path) -> Iterable[Tuple[Path, str]]:
    """Yield ``(absolute_path, posix_relative_key)`` for every file under a dir."""
    local_dir = Path(local_dir)
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            yield p, p.relative_to(local_dir).as_posix()


@runtime_checkable
class StorageAdapter(Protocol):
    """Write pack/projection/concept artifacts to a $0-tier backend (§15).

    Keys are ``/``-joined prefixes (e.g. ``packs/eurosat/img_007/manifest.json``).
    The web app (M5) reads gallery assets straight from ``get_url`` over HTTPS
    range requests, so every backend's URL must be publicly GET-able without auth.
    """

    def put_file(
        self, local_path: Any, key: str, *, content_type: Optional[str] = None
    ) -> str:
        """Upload a single file to ``key``; return its public URL."""
        ...

    def put_pack(self, local_dir: Any, prefix: str) -> Dict[str, str]:
        """Upload every file under ``local_dir`` beneath ``prefix``.

        Returns ``{relative_key: public_url}`` for each uploaded file.
        """
        ...

    def get_url(self, key: str) -> str:
        """Return the public-read URL for ``key`` (no network)."""
        ...

    def exists(self, key: str) -> bool:
        """Return whether ``key`` exists in the backend."""
        ...


# --------------------------------------------------------------------------- #
# Local
# --------------------------------------------------------------------------- #


class LocalStorageAdapter:
    """Filesystem backend for offline tests and dry-runs (``file://`` URLs)."""

    kind = "local"

    def __init__(self, root: Any) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def put_file(
        self, local_path: Any, key: str, *, content_type: Optional[str] = None
    ) -> str:
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(local_path), dst)
        return self.get_url(key)

    def put_bytes(
        self, data: bytes, key: str, *, content_type: Optional[str] = None
    ) -> str:
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return self.get_url(key)

    def put_pack(self, local_dir: Any, prefix: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        prefix = prefix.strip("/")
        for path, rel in _iter_pack_files(Path(local_dir)):
            key = f"{prefix}/{rel}" if prefix else rel
            self.put_file(path, key)
            out[rel] = self.get_url(key)
        return out

    def get_url(self, key: str) -> str:
        return self._path(key).as_uri()

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str) -> List[str]:
        base = self._path(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        return [p.relative_to(self.root).as_posix() for p in sorted(base.rglob("*")) if p.is_file()]


# --------------------------------------------------------------------------- #
# Supabase Storage (REST, public-read buckets)
# --------------------------------------------------------------------------- #


class SupabaseStorageAdapter:
    """Supabase Storage backend — the v1 primary (§15).

    Reads credentials from the environment (never hardcoded)::

        SUPABASE_URL          e.g. https://<ref>.supabase.co
        SUPABASE_SERVICE_KEY  service-role key (writes) — preferred
        SUPABASE_ANON_KEY     anon key (fallback; only public reads)

    Buckets are public-read, so ``get_url`` returns the stable public object URL
    ``{url}/storage/v1/object/public/{bucket}/{key}`` with no network call.
    Uploads use the ``supabase`` client when installed, else the Storage REST API
    directly via ``httpx``. Both imports are lazy — constructing the adapter does
    no network I/O.
    """

    kind = "supabase"

    def __init__(
        self,
        bucket: str = "packs",
        *,
        url: Optional[str] = None,
        key: Optional[str] = None,
        upsert: bool = True,
    ) -> None:
        self.bucket = bucket
        self.upsert = bool(upsert)
        self.url = (url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = (
            key
            or os.environ.get("SUPABASE_SERVICE_KEY")
            or os.environ.get("SUPABASE_ANON_KEY")
            or ""
        )
        if not self.url:
            raise EnvironmentError(
                "SUPABASE_URL is not set — export it (e.g. https://<ref>.supabase.co) "
                "or pass url=. Credentials are read from the environment, never hardcoded."
            )
        if not self.key:
            raise EnvironmentError(
                "no Supabase key found — set SUPABASE_SERVICE_KEY (writes) or "
                "SUPABASE_ANON_KEY (reads), or pass key=. Never hardcode credentials."
            )

    # -- public URL (no network) -------------------------------------------- #

    def get_url(self, key: str) -> str:
        return f"{self.url}/storage/v1/object/public/{self.bucket}/{key.lstrip('/')}"

    # -- uploads ------------------------------------------------------------- #

    def put_file(
        self, local_path: Any, key: str, *, content_type: Optional[str] = None
    ) -> str:
        data = Path(local_path).read_bytes()
        ct = content_type or _content_type_for(key)
        return self.put_bytes(data, key, content_type=ct)

    def put_bytes(
        self, data: bytes, key: str, *, content_type: Optional[str] = None
    ) -> str:
        import httpx

        key = key.lstrip("/")
        ct = content_type or _content_type_for(key)
        endpoint = f"{self.url}/storage/v1/object/{self.bucket}/{key}"
        headers = {
            "Authorization": f"Bearer {self.key}",
            "apikey": self.key,
            "Content-Type": ct,
            "x-upsert": "true" if self.upsert else "false",
        }
        resp = httpx.post(endpoint, content=data, headers=headers, timeout=60.0)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Supabase upload failed for {key!r}: {resp.status_code} {resp.text}"
            )
        return self.get_url(key)

    def put_pack(self, local_dir: Any, prefix: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        prefix = prefix.strip("/")
        for path, rel in _iter_pack_files(Path(local_dir)):
            key = f"{prefix}/{rel}" if prefix else rel
            out[rel] = self.put_file(path, key)
        return out

    def exists(self, key: str) -> bool:
        import httpx

        # A HEAD against the public object URL is authoritative for public buckets.
        resp = httpx.head(self.get_url(key), timeout=30.0, follow_redirects=True)
        return resp.status_code == 200


# --------------------------------------------------------------------------- #
# Hugging Face dataset repo (overflow valve)
# --------------------------------------------------------------------------- #


class HFDatasetStorageAdapter:
    """Overflow-valve backend: a public HF dataset repo (§5, §15).

    Uploads through ``huggingface_hub`` (lazy import); the token comes from
    ``HF_TOKEN``. ``get_url`` returns the stable ``resolve/main`` raw URL, which
    is publicly GET-able (and range-request capable) for public repos.
    """

    kind = "hf_dataset"

    def __init__(
        self,
        repo_id: str,
        *,
        token: Optional[str] = None,
        revision: str = "main",
    ) -> None:
        if not repo_id:
            raise ValueError("repo_id is required (e.g. 'user/vitreous-packs').")
        self.repo_id = repo_id
        self.revision = revision
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not self.token:
            raise EnvironmentError(
                "HF_TOKEN is not set — export a Hugging Face write token or pass "
                "token=. Credentials are read from the environment, never hardcoded."
            )

    def get_url(self, key: str) -> str:
        return (
            f"https://huggingface.co/datasets/{self.repo_id}/resolve/"
            f"{self.revision}/{key.lstrip('/')}"
        )

    def _api(self) -> Any:
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def put_file(
        self, local_path: Any, key: str, *, content_type: Optional[str] = None
    ) -> str:
        api = self._api()
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=key.lstrip("/"),
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
        )
        return self.get_url(key)

    def put_pack(self, local_dir: Any, prefix: str) -> Dict[str, str]:
        api = self._api()
        prefix = prefix.strip("/")
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=prefix,
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
        )
        return {
            rel: self.get_url(f"{prefix}/{rel}" if prefix else rel)
            for _p, rel in _iter_pack_files(Path(local_dir))
        }

    def exists(self, key: str) -> bool:
        from huggingface_hub import HfApi

        return HfApi(token=self.token).file_exists(
            repo_id=self.repo_id,
            filename=key.lstrip("/"),
            repo_type="dataset",
            revision=self.revision,
        )


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #


def get_storage(kind: Optional[str] = None, **kwargs: Any) -> StorageAdapter:
    """Return a configured :class:`StorageAdapter` (§15).

    ``kind`` defaults to the ``VITREOUS_STORAGE`` env var, else ``"local"``:

    * ``"local"``    — ``LocalStorageAdapter(root=VITREOUS_STORAGE_ROOT or './.vitreous-storage')``
    * ``"supabase"`` — ``SupabaseStorageAdapter`` (bucket via ``VITREOUS_BUCKET`` or ``"packs"``)
    * ``"hf"`` / ``"hf_dataset"`` — ``HFDatasetStorageAdapter`` (repo via ``VITREOUS_HF_REPO``)

    Explicit ``kwargs`` override env-derived defaults. No network at construction.
    """
    kind = (kind or os.environ.get("VITREOUS_STORAGE") or "local").lower()

    if kind == "local":
        root = kwargs.pop("root", None) or os.environ.get(
            "VITREOUS_STORAGE_ROOT", "./.vitreous-storage"
        )
        return LocalStorageAdapter(root=root, **kwargs)
    if kind == "supabase":
        bucket = kwargs.pop("bucket", None) or os.environ.get("VITREOUS_BUCKET", "packs")
        return SupabaseStorageAdapter(bucket=bucket, **kwargs)
    if kind in ("hf", "hf_dataset", "huggingface"):
        repo_id = kwargs.pop("repo_id", None) or os.environ.get("VITREOUS_HF_REPO", "")
        return HFDatasetStorageAdapter(repo_id=repo_id, **kwargs)
    raise ValueError(
        f"unknown storage kind {kind!r}; expected local | supabase | hf"
    )


__all__ = [
    "StorageAdapter",
    "LocalStorageAdapter",
    "SupabaseStorageAdapter",
    "HFDatasetStorageAdapter",
    "get_storage",
]
