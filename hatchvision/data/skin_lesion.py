"""HAM10000 skin lesion loader with epidemiological attribute annotations.

HAM10000 (ISIC 2018 / Kaggle — ``kmader/skin-lesion-analysis-toward-melanoma-detection``):
    <root>/
        HAM10000_metadata.csv
        HAM10000_images_part_1/*.jpg
        HAM10000_images_part_2/*.jpg

For other ISIC formats (CSV one-hot, or folder-per-class) use the ``isic``
loader (hatchvision/data/isic.py — ISICLoader) which auto-detects all layouts.

The HAM10000 loader splits by lesion_id (patient-level) to avoid data leakage
and exposes binary attribute annotations derived from metadata:
  - sex: male / female
  - location: face, back, trunk, scalp, …
  - age group: <20, 20-40, 40-60, 60-80, 80+

These drive the Hebbian concept grounding pipeline — concepts that activate
for high-age back-lesions vs. young face-lesions get informative names without
any manually annotated dermoscopic labels.

7 diagnostic classes (HAM10000):
  akiec  Actinic keratoses / intraepithelial carcinoma
  bcc    Basal cell carcinoma
  bkl    Benign keratosis-like lesions
  df     Dermatofibroma
  mel    Melanoma
  nv     Melanocytic nevi
  vasc   Vascular lesions
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from hatchvision.data.base import DatasetLoader, DatasetSpec, register_loader
from hatchvision.data.builtin import eval_transforms, train_transforms

# ImageNet stats — pretrained hybrid backbone expects this normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HAM10000_CLASSES: Dict[str, str] = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

# Canonical order — index-aligned with the model logits
HAM10000_CLASS_NAMES = tuple(HAM10000_CLASSES[k] for k in sorted(HAM10000_CLASSES))
HAM10000_CLASS_KEYS  = tuple(sorted(HAM10000_CLASSES))

# Age-group bucket boundaries (years)
_AGE_GROUPS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 200)]
_AGE_LABELS  = ["age: <20", "age: 20-40", "age: 40-60", "age: 60-80", "age: 80+"]

_LOCATIONS = [
    "location: abdomen", "location: acral", "location: back",
    "location: chest", "location: ear", "location: face",
    "location: foot", "location: genital", "location: hand",
    "location: lower extremity", "location: neck", "location: scalp",
    "location: trunk", "location: upper extremity",
]

_SEX_ATTRS = ["sex: male", "sex: female"]

ALL_ATTRS: List[str] = _SEX_ATTRS + _AGE_LABELS + _LOCATIONS


def _age_group(age: float) -> int:
    for i, (lo, hi) in enumerate(_AGE_GROUPS):
        if lo <= age < hi:
            return i
    return len(_AGE_GROUPS) - 1


class _SkinSplit(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


@register_loader("ham10000")
class Ham10000Loader(DatasetLoader):
    """HAM10000 loader with patient-level train/val split and metadata attributes."""

    def __init__(
        self,
        root: str = "./data",
        image_size: int = 224,
        val_ratio: float = 0.15,
        seed: int = 42,
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
        **_,
    ) -> None:
        super().__init__(limit_train=limit_train, limit_val=limit_val)
        self.root = Path(root)
        self.image_size = image_size

        self.spec = DatasetSpec(
            name="ham10000",
            num_classes=len(HAM10000_CLASS_NAMES),
            class_names=HAM10000_CLASS_NAMES,
            image_size=image_size,
            in_channels=3,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        meta_path = self._find_csv()
        rows = self._parse_csv(meta_path)

        # Patient-level split to prevent leakage
        lesion_ids = sorted({r["lesion_id"] for r in rows})
        rng = random.Random(seed)
        rng.shuffle(lesion_ids)
        n_val = max(1, int(len(lesion_ids) * val_ratio))
        val_lesions: Set[str] = set(lesion_ids[:n_val])

        cls_key_to_idx = {k: i for i, k in enumerate(HAM10000_CLASS_KEYS)}
        img_dirs = self._find_image_dirs()

        self._train: List[Tuple[Path, int]] = []
        self._val:   List[Tuple[Path, int]] = []
        self._val_meta: List[Dict] = []

        for r in rows:
            img_path = self._locate_image(r["image_id"], img_dirs)
            if img_path is None:
                continue
            label = cls_key_to_idx.get(r["dx"])
            if label is None:
                continue
            sample = (img_path, label)
            if r["lesion_id"] in val_lesions:
                self._val.append(sample)
                self._val_meta.append(r)
            else:
                self._train.append(sample)

        if not self._train:
            raise FileNotFoundError(
                f"No HAM10000 images found under {self.root}. "
                "Expected HAM10000_metadata.csv and image directories."
            )

    # ---------------------------------------------------------------- layout

    def _find_csv(self) -> Path:
        for cand in (
            self.root / "HAM10000_metadata.csv",
            self.root / "ham10000_metadata.csv",
        ):
            if cand.is_file():
                return cand
        raise FileNotFoundError(
            f"HAM10000_metadata.csv not found under {self.root}. "
            "Download the dataset from Kaggle (skin-lesion-analysis-toward-melanoma-detection)."
        )

    def _find_image_dirs(self) -> List[Path]:
        dirs = []
        for pattern in ("HAM10000_images_part*", "images", "HAM10000_images"):
            dirs.extend(sorted(self.root.glob(pattern)))
        if not dirs and (self.root / "images").is_dir():
            dirs = [self.root / "images"]
        return [d for d in dirs if d.is_dir()]

    @staticmethod
    def _locate_image(image_id: str, img_dirs: List[Path]) -> Optional[Path]:
        for d in img_dirs:
            for ext in ("jpg", "jpeg", "png"):
                p = d / f"{image_id}.{ext}"
                if p.is_file():
                    return p
        return None

    @staticmethod
    def _parse_csv(path: Path) -> List[Dict]:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    # ------------------------------------------------------------ datasets

    def train_dataset(self) -> Dataset:
        return _SkinSplit(self._train, train_transforms(self.spec))

    def val_dataset(self) -> Dataset:
        return _SkinSplit(self._val, eval_transforms(self.spec))

    # --------------------------------------------------------- attributes

    def attribute_names(self) -> Optional[List[str]]:
        return list(ALL_ATTRS)

    def val_attribute_matrix(self) -> Optional[torch.Tensor]:
        n = len(self._val_meta)
        a = len(ALL_ATTRS)
        mat = torch.zeros(n, a)
        for i, r in enumerate(self._val_meta):
            sex = r.get("sex", "").lower()
            if sex == "male":
                mat[i, 0] = 1.0
            elif sex == "female":
                mat[i, 1] = 1.0

            try:
                age = float(r.get("age", -1))
            except (ValueError, TypeError):
                age = -1.0
            if age >= 0:
                mat[i, 2 + _age_group(age)] = 1.0

            loc = r.get("localization", "").lower().strip()
            loc_attr = f"location: {loc}"
            if loc_attr in _LOCATIONS:
                mat[i, 2 + len(_AGE_LABELS) + _LOCATIONS.index(loc_attr)] = 1.0

        return mat

