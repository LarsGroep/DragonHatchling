"""ISIC skin-lesion loader (dermoscopy) — HAM10000, ISIC 2019, or folders.

Skin-lesion classification is a natural stress test for the Hebbian
explainability pipeline: unlike CUB-200 there are **no per-image visual
attribute annotations**, so concepts are grounded by class affinity +
exemplars (the framework does this automatically when
:meth:`attribute_names` returns ``None``).

The dataset ships on Kaggle in several shapes; this loader auto-detects the
three common ones so a dataset swap stays a one-liner:

1. **HAM10000 metadata CSV** (``kmader/skin-cancer-mnist-ham10000``): a
   ``*metadata*.csv`` with an image-id column, a ``dx`` diagnosis column and
   (usually) a ``lesion_id`` column. Images live in one or more folders and
   are matched by id.
2. **ISIC one-hot ground-truth CSV** (e.g. ISIC 2019): an id column followed
   by one column per class (``MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK``) holding a
   one-hot row; the label is the arg-max column.
3. **Folder-per-class**: ``train/<class>/*`` and ``val/`` or ``test/``
   (any capitalisation), like a plain ImageFolder.

Splits: folder layouts keep their own train/test; CSV layouts get a
deterministic **stratified split grouped by ``lesion_id``** when present, so
the multiple dermoscopy images of one lesion never straddle train and val
(a classic ISIC leakage trap that inflates accuracy).

``dx`` codes are expanded to readable class names ("mel" → "Melanoma").

.. note::
   Research / education only. A model trained here is **not** a diagnostic
   device and its explanations describe the model, not clinical ground truth.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

from hatchvision.data.base import DatasetLoader, DatasetSpec, register_loader
from hatchvision.data.builtin import eval_transforms, train_transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ISIC / HAM diagnosis codes → human-readable class names.
DX_NAMES = {
    "akiec": "Actinic keratosis",
    "ak": "Actinic keratosis",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "scc": "Squamous cell carcinoma",
    "vasc": "Vascular lesion",
    "unk": "Unknown",
}
ONEHOT_CODES = {c.upper() for c in DX_NAMES}


def pretty_dx(code: str) -> str:
    """``mel`` → ``Melanoma``; unknown codes are title-cased as-is."""
    c = code.strip()
    return DX_NAMES.get(c.lower(), c.replace("_", " ").strip().title() or c)


class _ImageList(Dataset):
    """A flat ``(path, label)`` list; exposes ``.targets`` for balanced
    sampling (mirrors torchvision's ``ImageFolder``)."""

    def __init__(self, samples: List[Tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _index_images(base: Path) -> Dict[str, Path]:
    """Map every image file's stem → path (first occurrence wins)."""
    index: Dict[str, Path] = {}
    for p in base.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.stem, p)
    return index


def _id_column(low: Sequence[str]) -> Optional[int]:
    for name in ("image_id", "image", "image_name", "isic_id", "img_id", "name"):
        if name in low:
            return low.index(name)
    for i, c in enumerate(low):
        if "image" in c:
            return i
    return None


def _to_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _from_csv(base: Path) -> Optional[Tuple[List[Tuple[Path, int]], List[str], Optional[List[str]]]]:
    """Parse the first CSV that looks like an ISIC label file.

    Returns ``(int-labelled samples, class_names, group_keys | None)`` where
    ``group_keys[i]`` is the lesion id of sample ``i`` (for leakage-free
    splitting) or ``None`` when no ``lesion_id`` column is present.
    """
    for csv_path in sorted(base.rglob("*.csv")):
        try:
            with open(csv_path, newline="") as f:
                rows = list(csv.reader(f))
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        if len(rows) < 3:
            continue
        header = [h.strip() for h in rows[0]]
        low = [h.lower() for h in header]
        id_idx = _id_column(low)
        if id_idx is None:
            continue
        if "dx" in low:
            mode, dx_idx, class_cols = "dx", low.index("dx"), None
        else:
            class_cols = [i for i, h in enumerate(header)
                          if i != id_idx and h.upper() in ONEHOT_CODES]
            if len(class_cols) < 2:
                continue
            mode, dx_idx = "onehot", None
        group_idx = low.index("lesion_id") if "lesion_id" in low else None

        index = _index_images(base)
        samples: List[Tuple[Path, str]] = []
        groups: List[Optional[str]] = []
        for r in rows[1:]:
            if len(r) <= id_idx or not r[id_idx].strip():
                continue
            path = index.get(Path(r[id_idx].strip()).stem)
            if path is None:
                continue
            if mode == "dx":
                code = r[dx_idx].strip() if dx_idx < len(r) else ""
                if not code:
                    continue
            else:
                vals = [_to_float(r[i]) if i < len(r) else 0.0 for i in class_cols]
                if max(vals) <= 0:
                    continue
                code = header[class_cols[vals.index(max(vals))]]
            samples.append((path, code))
            groups.append(r[group_idx].strip() if group_idx is not None and group_idx < len(r) else None)

        if not samples:
            continue
        codes = sorted({c for _, c in samples}, key=lambda c: pretty_dx(c).lower())
        cidx = {c: i for i, c in enumerate(codes)}
        int_samples = [(p, cidx[c]) for p, c in samples]
        class_names = [pretty_dx(c) for c in codes]
        grp = groups if any(g for g in groups) else None
        return int_samples, class_names, grp
    return None


def _find_dir(base: Path, names: set) -> Optional[Path]:
    if base.name.lower() in names and any(c.is_dir() for c in base.iterdir()):
        return base
    for p in sorted(base.rglob("*")):
        if p.is_dir() and p.name.lower() in names and any(c.is_dir() for c in p.iterdir()):
            return p
    return None


def _from_folders(base: Path):
    """Detect a ``train/<class>`` (+ optional ``val``/``test``) tree."""
    train_dir = _find_dir(base, {"train", "training"})
    if train_dir is None:
        return None
    classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    if not classes:
        return None
    cidx = {c: i for i, c in enumerate(classes)}

    def collect(d: Path) -> List[Tuple[Path, int]]:
        out: List[Tuple[Path, int]] = []
        for c in d.iterdir():
            if c.is_dir() and c.name in cidx:
                for img in c.rglob("*"):
                    if img.suffix.lower() in IMAGE_EXTS:
                        out.append((img, cidx[c.name]))
        return out

    train = collect(train_dir)
    val_dir = _find_dir(base, {"val", "valid", "validation", "test", "testing"})
    val = collect(val_dir) if val_dir is not None else []
    return train, val, [pretty_dx(c) for c in classes]


def _stratified_split(
    samples: List[Tuple[Path, int]],
    groups: Optional[List[str]],
    val_frac: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Deterministic per-class split; group-aware when ``groups`` given so
    all images of one lesion land in the same split."""
    rng = random.Random(seed)
    if groups is None:
        by_label: Dict[int, List[int]] = defaultdict(list)
        for i, (_, label) in enumerate(samples):
            by_label[label].append(i)
        val_idx = set()
        for idxs in by_label.values():
            idxs = sorted(idxs, key=lambda i: str(samples[i][0]))
            rng.shuffle(idxs)
            k = max(1, round(len(idxs) * val_frac)) if len(idxs) > 1 else 0
            val_idx.update(idxs[:k])
    else:
        members: Dict[str, List[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            members[g].append(i)
        group_label = {g: samples[m[0]][1] for g, m in members.items()}
        by_label = defaultdict(list)
        for g, label in group_label.items():
            by_label[label].append(g)
        val_groups = set()
        for gs in by_label.values():
            gs = sorted(gs)
            rng.shuffle(gs)
            k = max(1, round(len(gs) * val_frac)) if len(gs) > 1 else 0
            val_groups.update(gs[:k])
        val_idx = {i for i, g in enumerate(groups) if g in val_groups}
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


@register_loader("isic")
class ISICLoader(DatasetLoader):
    """Skin-lesion loader; auto-detects CSV or folder-per-class layouts."""

    def __init__(
        self,
        root: str = "./data",
        image_size: int = 224,
        val_frac: float = 0.15,
        seed: int = 0,
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
        **_,
    ) -> None:
        super().__init__(limit_train=limit_train, limit_val=limit_val)
        base = Path(root)
        if not base.exists():
            raise FileNotFoundError(f"ISIC root {base} does not exist")

        parsed = _from_csv(base)
        if parsed is not None:
            samples, class_names, groups = parsed
            self._train, self._val = _stratified_split(samples, groups, val_frac, seed)
        else:
            folders = _from_folders(base)
            if folders is None:
                raise FileNotFoundError(
                    f"No ISIC data found under {base}: expected a metadata/"
                    "ground-truth .csv (with a 'dx' or one-hot class columns) "
                    "or a train/<class>/ folder tree."
                )
            train, val, class_names = folders
            if not val:
                train, val = _stratified_split(train, None, val_frac, seed)
            self._train, self._val = train, val

        if not self._train:
            raise RuntimeError(f"ISIC loader found no training images under {base}")
        self.spec = DatasetSpec(
            name="isic",
            num_classes=len(class_names),
            class_names=tuple(class_names),
            image_size=image_size,
            in_channels=3,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

    def train_dataset(self) -> Dataset:
        return _ImageList(self._train, train_transforms(self.spec))

    def val_dataset(self) -> Dataset:
        return _ImageList(self._val, eval_transforms(self.spec))
