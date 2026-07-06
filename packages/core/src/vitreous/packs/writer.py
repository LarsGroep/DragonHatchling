"""PackWriter / PackReader — Explanation Pack directories (§5, §6).

PACK FORMAT v1, FROZEN AT M2. This module owns the exact on-disk layout of every
pack asset and the quantization rules; the manifest (``manifest.json``) is the
index and is schema-validated on write and on read.

Binary layouts (frozen)
------------------------
* **attention.bin** — ``[L,H,T,T]`` softmax maps, per-row max-quantized to
  ``uint8``. A "row" is the last axis (the distribution over keys), so there are
  ``L·H·T`` rows of length ``T``. File = ``uint8`` data block (C-order) followed
  by the per-row ``float32`` scales block; ``scale[r] = max(row_r)`` so
  ``dequant = data/255 · scale`` has error ``<= 0.5/255`` per element. The exact
  offsets are recorded in the asset's ``quant`` field.
* **tokens.bin** — ``[L+1,T,D]`` ``float16``, raw C-order.
* **attr_rollout.bin / attr_chefer.bin** — ``[L,T]`` ``float32``, raw.
* **attr_gradcam.bin** — ``[14,14]`` ``float32``, raw.
* **attr_ig.bin** — ``[T]`` ``float32``, raw (token-level IG).
* **attr_ig_pixel.png** — ``[224,224]`` grayscale PNG of ``|IG|`` (pixel-level).
* **attributions.json / faithfulness.json** — JSON.
* **image.webp** (or **image.png** fallback) — the display image.

``build_pack`` orchestrates capture → all XAI methods → faithfulness → write.
Only ``build_pack`` needs torch (imported lazily); ``PackWriter``/``PackReader``
themselves are numpy-only, so ``import vitreous`` stays torch-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .manifest import (
    AssetEntry,
    DatasetInfo,
    ImageMeta,
    ModelInfo,
    PackManifest,
    Prediction,
    QuantInfo,
)

PACK_VERSION = "1.0.0"  # pack format v1, frozen at M2

_NP_TO_DTYPE = {
    "uint8": "uint8",
    "int8": "int8",
    "uint16": "uint16",
    "int16": "int16",
    "int32": "int32",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
}
_DTYPE_TO_NP = {
    "uint8": np.uint8,
    "int8": np.int8,
    "uint16": np.uint16,
    "int16": np.int16,
    "int32": np.int32,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}


def _quantize_per_row(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row (last-axis) max-quantize a float array to uint8 + float32 scales."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    T = a.shape[-1]
    flat = a.reshape(-1, T)
    scales = flat.max(axis=1)
    safe = np.where(scales > 0, scales, 1.0)
    q = np.rint(flat / safe[:, None] * 255.0)
    q = np.clip(q, 0, 255).astype(np.uint8).reshape(a.shape)
    return q, scales.astype(np.float32)


def _dequantize_per_row(
    data: np.ndarray, scales: np.ndarray, shape: Tuple[int, ...]
) -> np.ndarray:
    """Inverse of :func:`_quantize_per_row` → float32 array of ``shape``."""
    T = shape[-1]
    d = data.reshape(-1, T).astype(np.float32)
    s = scales.reshape(-1)
    out = d / 255.0 * s[:, None]
    return out.reshape(shape).astype(np.float32)


@dataclass
class PackWriter:
    """Accumulates assets and emits a schema-valid pack directory.

    Files are written immediately as they are added; :meth:`write_manifest`
    stamps the asset index (with real byte sizes) at the end.
    """

    out_dir: Path
    _assets: Dict[str, AssetEntry] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def assets(self) -> Dict[str, AssetEntry]:
        return dict(self._assets)

    # -- array assets -------------------------------------------------------- #

    def add_array(
        self,
        filename: str,
        array: Any,
        *,
        encoding: str = "raw",
        dtype: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AssetEntry:
        """Serialize a numpy array asset with ``encoding`` in {raw, per_row_uint8}.

        ``meta`` is optional additive free-form metadata recorded on the asset
        entry (used e.g. for the ``gaussians.bin`` channel order); it never
        describes the frozen binary layout.
        """
        arr = np.asarray(array)
        if dtype is not None:
            arr = arr.astype(_DTYPE_TO_NP[dtype])
        path = self.out_dir / filename

        if encoding == "raw":
            key = np.dtype(arr.dtype).name
            if key not in _NP_TO_DTYPE:
                raise ValueError(f"unsupported raw dtype {key!r} for {filename}")
            data = np.ascontiguousarray(arr).tobytes()
            path.write_bytes(data)
            entry = AssetEntry(
                dtype=_NP_TO_DTYPE[key],
                shape=list(arr.shape),
                encoding="raw",
                bytes=len(data),
                meta=meta,
            )
        elif encoding == "per_row_uint8":
            q, scales = _quantize_per_row(arr)
            data = np.ascontiguousarray(q).tobytes()
            scale_bytes = np.ascontiguousarray(scales, dtype="<f4").tobytes()
            path.write_bytes(data + scale_bytes)
            entry = AssetEntry(
                dtype="uint8",
                shape=list(arr.shape),
                encoding="per_row_uint8",
                bytes=len(data) + len(scale_bytes),
                quant=QuantInfo(
                    scheme="per_row_uint8",
                    row_axis=-1,
                    scale_dtype="float32",
                    data_offset=0,
                    data_bytes=len(data),
                    scale_offset=len(data),
                    scale_count=int(scales.size),
                ),
                meta=meta,
            )
        else:
            raise ValueError(f"unknown array encoding {encoding!r}")

        self._assets[filename] = entry
        return entry

    # -- json assets --------------------------------------------------------- #

    def add_json(
        self, filename: str, payload: Any, *, meta: Optional[Dict[str, Any]] = None
    ) -> AssetEntry:
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        (self.out_dir / filename).write_bytes(data)
        entry = AssetEntry(
            dtype="json", shape=[], encoding="json", bytes=len(data), meta=meta
        )
        self._assets[filename] = entry
        return entry

    # -- image assets -------------------------------------------------------- #

    def add_image(self, basename: str, hwc_uint8: np.ndarray) -> Tuple[str, AssetEntry]:
        """Save an ``[H,W,3] uint8`` display image as WebP (PNG fallback).

        Returns the actual filename written (extension reflects the real format).
        """
        from PIL import Image, features  # lazy

        img = Image.fromarray(np.ascontiguousarray(hwc_uint8, dtype=np.uint8))
        if features.check("webp"):
            filename = f"{basename}.webp"
            fmt, dtype, encoding = "WEBP", "webp", "webp"
        else:
            filename = f"{basename}.png"
            fmt, dtype, encoding = "PNG", "png", "png"
        img.save(self.out_dir / filename, fmt)
        nbytes = (self.out_dir / filename).stat().st_size
        entry = AssetEntry(
            dtype=dtype, shape=[int(hwc_uint8.shape[0]), int(hwc_uint8.shape[1])],
            encoding=encoding, bytes=nbytes,
        )
        self._assets[filename] = entry
        return filename, entry

    def add_png_gray(self, filename: str, hw_float: np.ndarray) -> AssetEntry:
        """Save a ``[H,W]`` float map as a normalized 8-bit grayscale PNG."""
        from PIL import Image  # lazy

        a = np.abs(np.asarray(hw_float, dtype=np.float32))
        mx = float(a.max())
        norm = (a / mx * 255.0) if mx > 0 else a
        img = Image.fromarray(norm.astype(np.uint8), mode="L")
        img.save(self.out_dir / filename, "PNG")
        nbytes = (self.out_dir / filename).stat().st_size
        entry = AssetEntry(
            dtype="png", shape=[int(a.shape[0]), int(a.shape[1])],
            encoding="png", bytes=nbytes,
        )
        self._assets[filename] = entry
        return entry

    # -- manifest ------------------------------------------------------------ #

    def build_manifest(
        self,
        *,
        model: ModelInfo,
        dataset: DatasetInfo,
        image: ImageMeta,
        prediction: Prediction,
        timings: Optional[Dict[str, float]] = None,
    ) -> PackManifest:
        return PackManifest(
            pack_version=PACK_VERSION,
            model=model,
            dataset=dataset,
            image=image,
            prediction=prediction,
            assets=self._assets,
            timings=timings or {},
        )

    def write_manifest(self, manifest: PackManifest) -> Path:
        """Validate (Pydantic + JSON Schema) and write ``manifest.json``."""
        import jsonschema

        from . import load_pack_schema

        payload = manifest.model_dump(mode="json", exclude_none=True)
        jsonschema.validate(instance=payload, schema=load_pack_schema())
        path = self.out_dir / "manifest.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #


class PackReader:
    """Loads a pack directory back into arrays; validates the manifest.

    ``PackReader(dir).manifest`` is the validated :class:`PackManifest`;
    :meth:`read_attention` dequantizes attention to ``float32``; :meth:`read_array`
    reads any raw/per-row asset; :meth:`read_json` reads JSON assets.
    """

    def __init__(self, pack_dir: Any, *, validate: bool = True) -> None:
        self.dir = Path(pack_dir)
        raw = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        if validate:
            import jsonschema

            from . import load_pack_schema

            jsonschema.validate(instance=raw, schema=load_pack_schema())
        self.manifest = PackManifest.model_validate(raw)

    def _entry(self, name: str) -> AssetEntry:
        try:
            return self.manifest.assets[name]
        except KeyError as exc:
            raise KeyError(f"no asset {name!r} in pack {self.dir}") from exc

    def asset_path(self, name: str) -> Path:
        return self.dir / name

    def read_json(self, name: str) -> Any:
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    def read_array(self, name: str) -> np.ndarray:
        """Read a raw or per-row-quantized array asset back to a numpy array."""
        entry = self._entry(name)
        blob = (self.dir / name).read_bytes()
        shape = tuple(entry.shape)
        if entry.encoding == "raw":
            arr = np.frombuffer(blob, dtype=_DTYPE_TO_NP[entry.dtype])
            return arr.reshape(shape).copy()
        if entry.encoding == "per_row_uint8":
            q = entry.quant
            assert q is not None, "per_row_uint8 asset missing quant info"
            data = np.frombuffer(
                blob[q.data_offset : q.data_offset + q.data_bytes], dtype=np.uint8
            ).reshape(shape)
            scales = np.frombuffer(
                blob[q.scale_offset : q.scale_offset + q.scale_count * 4], dtype="<f4"
            )
            return _dequantize_per_row(data, scales, shape)
        raise ValueError(f"cannot read_array for encoding {entry.encoding!r}")

    def read_attention(self) -> np.ndarray:
        """Return dequantized attention ``[L,H,T,T]`` float32."""
        return self.read_array("attention.bin")

    def read_tokens(self) -> np.ndarray:
        return self.read_array("tokens.bin")

    def read_gaussians(self) -> np.ndarray:
        """Return the Gaussian Feature Field ``[13,197,12]`` float16 (§7)."""
        return self.read_array("gaussians.bin")

    def gaussian_channels(self) -> List[str]:
        """Channel order for ``gaussians.bin`` from the asset meta (§7)."""
        entry = self._entry("gaussians.bin")
        meta = entry.meta or {}
        return list(meta.get("channels", []))

    def read_graph(self) -> Dict[str, Any]:
        """Return the parsed ``graph.json`` structure (§8)."""
        return self.read_json("graph.json")


# --------------------------------------------------------------------------- #
# build_pack — the orchestrator (§5, §6)
# --------------------------------------------------------------------------- #


def build_pack(
    model: Any,
    image_tensor: Any,
    image_meta: Dict[str, Any],
    dataset_spec: Any,
    out_dir: Any,
    *,
    methods: Optional[Tuple[str, ...]] = None,
    class_idx: Optional[int] = None,
    display_image: Optional[Any] = None,
    model_info: Optional[Dict[str, Any]] = None,
    ig_steps: int = 20,
    faithfulness_steps: int = 20,
    faithfulness_methods: Optional[Tuple[str, ...]] = None,
    seed: int = 0,
) -> Path:
    """Capture, run every XAI method + faithfulness, and write a complete pack.

    Parameters
    ----------
    model:
        A ``LoadedModel`` or a raw timm ViT module.
    image_tensor:
        Model input ``[3,224,224]`` or ``[1,3,224,224]`` (already preprocessed).
    image_meta:
        At least ``{"id": str, "source": "gallery"|"upload"}``; ``width``/``height``
        default to the image size.
    dataset_spec:
        Anything with ``num_classes`` and (optionally) ``name``/``display_name``/
        ``class_names``.
    out_dir:
        Destination pack directory (created if missing).
    methods:
        Which attribution methods to run (default all: rollout, chefer, gradcam, ig).
    class_idx:
        Explanation target class; defaults to the predicted (argmax) class.
    """
    import time

    import torch

    from ..instrument import Instrumenter
    from ..xai import (
        attention_rollout,
        chefer_relevance,
        grad_cam,
        integrated_gradients,
    )
    from ..xai._common import as_batch, unwrap
    from ..xai.eval import FaithfulnessResult, deletion_insertion, method_agreement

    if methods is None:
        methods = ("rollout", "chefer", "gradcam", "ig")

    module = unwrap(model)
    module.eval()
    x = as_batch(image_tensor)
    timings: Dict[str, float] = {}

    # -- capture (attention + tokens + logits) ------------------------------- #
    t0 = time.perf_counter()
    trace = Instrumenter(module).capture(x)
    timings["predict_ms"] = (time.perf_counter() - t0) * 1000.0

    logits = trace.logits
    probs = logits.softmax(dim=-1)[0]
    pred_idx = int(probs.argmax().item())
    target = pred_idx if class_idx is None else int(class_idx)
    confidence = float(probs[pred_idx].item())

    # -- attribution methods ------------------------------------------------- #
    attrs: Dict[str, Any] = {}
    if "rollout" in methods:
        t0 = time.perf_counter()
        attrs["rollout"] = attention_rollout(trace)
        timings["rollout_ms"] = (time.perf_counter() - t0) * 1000.0
    if "chefer" in methods:
        t0 = time.perf_counter()
        attrs["chefer"] = chefer_relevance(module, x, class_idx=target)
        timings["chefer_ms"] = (time.perf_counter() - t0) * 1000.0
    if "gradcam" in methods:
        t0 = time.perf_counter()
        attrs["gradcam"] = grad_cam(module, x, class_idx=target)
        timings["gradcam_ms"] = (time.perf_counter() - t0) * 1000.0
    if "ig" in methods:
        t0 = time.perf_counter()
        attrs["ig"] = integrated_gradients(module, x, class_idx=target, steps=ig_steps)
        timings["ig_ms"] = (time.perf_counter() - t0) * 1000.0

    # -- writer + asset serialization --------------------------------------- #
    writer = PackWriter(out_dir)
    attr_index: Dict[str, Dict[str, str]] = {}

    writer.add_array(
        "attention.bin", trace.attention.cpu().numpy(), encoding="per_row_uint8"
    )
    writer.add_array("tokens.bin", trace.tokens.cpu().numpy(), encoding="raw", dtype="float16")

    if "rollout" in attrs:
        writer.add_array("attr_rollout.bin", attrs["rollout"].token_scores, encoding="raw", dtype="float32")
        attr_index["rollout"] = {"asset": "attr_rollout.bin", "kind": "per_layer_tokens"}
    if "chefer" in attrs:
        writer.add_array("attr_chefer.bin", attrs["chefer"].token_scores, encoding="raw", dtype="float32")
        attr_index["chefer"] = {"asset": "attr_chefer.bin", "kind": "per_layer_tokens"}
    if "gradcam" in attrs:
        writer.add_array("attr_gradcam.bin", attrs["gradcam"].pixel_map, encoding="raw", dtype="float32")
        attr_index["gradcam"] = {"asset": "attr_gradcam.bin", "kind": "token_grid"}
    if "ig" in attrs:
        writer.add_array("attr_ig.bin", attrs["ig"].token_scores, encoding="raw", dtype="float32")
        writer.add_png_gray("attr_ig_pixel.png", attrs["ig"].pixel_map)
        attr_index["ig"] = {
            "asset": "attr_ig.bin", "kind": "tokens", "pixel_asset": "attr_ig_pixel.png",
        }

    writer.add_json("attributions.json", attr_index)

    # -- faithfulness -------------------------------------------------------- #
    if faithfulness_methods is None:
        faithfulness_methods = tuple(attrs.keys())
    t0 = time.perf_counter()
    faith = FaithfulnessResult(steps=faithfulness_steps)
    ranking_arrays: Dict[str, Any] = {}
    for name, attr in attrs.items():
        ranking_arrays[name] = (
            attr.pixel_map if attr.token_scores is None else attr.token_scores
        )
    # Random baseline ranking (seeded for determinism).
    rng = np.random.default_rng(seed)
    npatch = (x.shape[-1] // 16) ** 2
    ranking_arrays["random"] = rng.random(npatch).astype(np.float32)

    for name in list(faithfulness_methods) + ["random"]:
        if name not in ranking_arrays:
            continue
        di = deletion_insertion(
            module, x, ranking_arrays[name], steps=faithfulness_steps, target=target
        )
        faith.deletion_curves[name] = di.deletion
        faith.insertion_curves[name] = di.insertion
        faith.deletion_auc[name] = di.deletion_auc
        faith.insertion_auc[name] = di.insertion_auc
    faith.agreement = method_agreement({k: v for k, v in ranking_arrays.items() if k != "random"})
    timings["faithfulness_ms"] = (time.perf_counter() - t0) * 1000.0
    writer.add_json("faithfulness.json", faith.to_json())

    # -- display image ------------------------------------------------------- #
    if display_image is not None:
        disp = np.asarray(display_image, dtype=np.uint8)
    else:
        img = x[0].detach().cpu().numpy()  # [C,H,W]
        img = np.transpose(img, (1, 2, 0))
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
        disp = (img * 255.0).astype(np.uint8)
    writer.add_image("image", disp)

    # -- Gaussian Feature Field (§7) + Interaction Graph (§8) ----------------- #
    from ..gaussians import build_gaussian_field
    from ..graph import build_graph_asset

    t0 = time.perf_counter()
    chefer_layers = attrs["chefer"].token_scores if "chefer" in attrs else None
    field_obj = build_gaussian_field(trace, disp, chefer_layers=chefer_layers)
    writer.add_array(
        "gaussians.bin", field_obj.data, encoding="raw", dtype="float16",
        meta=field_obj.to_meta(),
    )
    timings["gaussians_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    graph_asset = build_graph_asset(trace, seed=seed)
    writer.add_json(
        "graph.json", graph_asset,
        meta={"k": graph_asset["k"], "num_layers": graph_asset["num_layers"],
              "residual": "implicit; see graph.json residual flag"},
    )
    timings["graph_ms"] = (time.perf_counter() - t0) * 1000.0

    # -- manifest ------------------------------------------------------------ #
    num_classes = int(getattr(dataset_spec, "num_classes", probs.shape[0]))
    class_names = list(getattr(dataset_spec, "class_names", []) or [])
    if len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]

    if model_info is not None:
        minfo = ModelInfo(**model_info)
    elif hasattr(model, "to_model_info_kwargs"):
        minfo = ModelInfo(**model.to_model_info_kwargs())
    else:
        minfo = ModelInfo(
            arch=str(getattr(module, "__class__").__name__),
            hf_repo="local",
            num_layers=int(trace.meta.get("num_layers", len(module.blocks))),
            num_heads=int(trace.meta.get("num_heads", module.blocks[0].attn.num_heads)),
            num_tokens=int(trace.attention.shape[-1]),
            embed_dim=int(trace.tokens.shape[-1]),
            patch_size=int(round(int(x.shape[-1]) / ((trace.attention.shape[-1] - 1) ** 0.5))),
        )

    hw = int(x.shape[-1])
    manifest = writer.build_manifest(
        model=minfo,
        dataset=DatasetInfo(
            name=str(getattr(dataset_spec, "name", "dataset")),
            display_name=getattr(dataset_spec, "display_name", None),
            num_classes=num_classes,
            class_names=class_names,
        ),
        image=ImageMeta(
            id=str(image_meta.get("id", "image")),
            width=int(image_meta.get("width", hw)),
            height=int(image_meta.get("height", hw)),
            source=image_meta.get("source", "gallery"),
        ),
        prediction=Prediction(
            label=class_names[pred_idx],
            class_index=pred_idx,
            confidence=confidence,
            probabilities=[float(p) for p in probs.detach().cpu().tolist()],
        ),
        timings=timings,
    )
    writer.write_manifest(manifest)
    return writer.out_dir


__all__ = ["PackWriter", "PackReader", "build_pack", "PACK_VERSION"]
