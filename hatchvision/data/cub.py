"""CUB-200-2011 loader (200 bird species) with attribute annotations.

The Caltech-UCSD Birds dataset ships per-image binary labels for 312
human-readable attributes ("has_wing_color::yellow", "has_bill_shape::hooked",
...) plus continuous class-level attribute frequencies.  This loader exposes
them through the generic :class:`DatasetLoader` attribute interface, which is
what lets the explainability layer auto-name Hebbian concepts with visual
features instead of "concept 7".

Directory layout expected (the official ``CUB_200_2011.tgz``, also mirrored
on Kaggle, e.g. dataset ``wenewone/cub2002011``)::

    <root>/CUB_200_2011/
        images.txt  image_class_labels.txt  train_test_split.txt  classes.txt
        images/<class>/<img>.jpg
        attributes/image_attribute_labels.txt
        attributes/class_attribute_labels_continuous.txt
    <root>/attributes.txt        # sometimes inside CUB_200_2011/

``root`` may point at either ``<root>`` or the ``CUB_200_2011`` directory
itself; both are detected.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from hatchvision.data.base import DatasetLoader, DatasetSpec, register_loader
from hatchvision.data.builtin import eval_transforms, train_transforms

DOWNLOAD_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def prettify_attribute(raw: str) -> str:
    """``has_wing_color::yellow`` -> ``wing color: yellow``."""
    name = raw.strip()
    name = re.sub(r"^has_", "", name)
    name = name.replace("::", ": ").replace("_", " ")
    return name


class _CubSplit(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


@register_loader("cub200")
class Cub200Loader(DatasetLoader):
    def __init__(
        self,
        root: str = "./data",
        image_size: int = 224,
        download: bool = False,
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
        **_,
    ) -> None:
        super().__init__(limit_train=limit_train, limit_val=limit_val)
        self.base = self._locate(Path(root), download)

        classes = self._read_indexed(self.base / "classes.txt")
        # "001.Black_footed_Albatross" -> "Black-footed Albatross"
        pretty = tuple(
            c.split(".", 1)[-1].replace("_", " ") for c in classes
        )
        self.spec = DatasetSpec(
            name="cub200",
            num_classes=len(classes),
            class_names=pretty,
            image_size=image_size,
            in_channels=3,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        images = self._read_indexed(self.base / "images.txt")
        labels = self._read_indexed(self.base / "image_class_labels.txt")
        is_train = self._read_indexed(self.base / "train_test_split.txt")
        img_dir = self.base / "images"
        self._train: List[Tuple[Path, int]] = []
        self._val: List[Tuple[Path, int]] = []
        self._val_image_ids: List[int] = []  # 1-based ids, val order
        for i, (rel, cls, tr) in enumerate(zip(images, labels, is_train)):
            sample = (img_dir / rel, int(cls) - 1)
            if tr == "1":
                self._train.append(sample)
            else:
                self._val.append(sample)
                self._val_image_ids.append(i + 1)

        self._attr_names: Optional[List[str]] = None
        self._val_attr: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------- layout

    @staticmethod
    def _locate(root: Path, download: bool) -> Path:
        for cand in (root, root / "CUB_200_2011", root / "CUB_200_2011" / "CUB_200_2011"):
            if (cand / "images.txt").is_file():
                return cand
        if download:
            from torchvision.datasets.utils import download_and_extract_archive

            download_and_extract_archive(DOWNLOAD_URL, str(root))
            return Cub200Loader._locate(root, download=False)
        raise FileNotFoundError(
            f"CUB_200_2011 not found under {root}. Download {DOWNLOAD_URL} "
            "and extract it there, attach the Kaggle dataset, or pass "
            "download=True."
        )

    @staticmethod
    def _read_indexed(path: Path) -> List[str]:
        """Files of the form ``<1-based-id> <value...>`` -> values in order."""
        out: List[str] = []
        with open(path) as f:
            for line in f:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    out.append(parts[1].strip())
        return out

    # --------------------------------------------------------------- datasets

    def train_dataset(self) -> Dataset:
        return _CubSplit(self._train, train_transforms(self.spec))

    def val_dataset(self) -> Dataset:
        return _CubSplit(self._val, eval_transforms(self.spec))

    # ------------------------------------------------------------- attributes

    def attribute_names(self) -> Optional[List[str]]:
        if self._attr_names is None:
            for cand in (
                self.base / "attributes" / "attributes.txt",
                self.base / "attributes.txt",
                self.base.parent / "attributes.txt",
            ):
                if cand.is_file():
                    self._attr_names = [
                        prettify_attribute(v) for v in self._read_indexed(cand)
                    ]
                    break
        return self._attr_names

    def val_attribute_matrix(self) -> Optional[torch.Tensor]:
        """[len(val), 312] binary presence matrix, parsed lazily (~80 MB file)."""
        if self._val_attr is not None:
            return self._val_attr
        names = self.attribute_names()
        labels_file = self.base / "attributes" / "image_attribute_labels.txt"
        if names is None or not labels_file.is_file():
            return None
        val_row = {img_id: row for row, img_id in enumerate(self._val_image_ids)}
        mat = torch.zeros(len(self._val), len(names))
        with open(labels_file) as f:
            for line in f:
                # <image_id> <attr_id> <is_present> <certainty> <time>
                # (a handful of lines in the official file have extra fields)
                parts = line.split()
                if len(parts) < 3:
                    continue
                row = val_row.get(int(parts[0]))
                if row is not None and parts[2] == "1":
                    mat[row, int(parts[1]) - 1] = 1.0
        self._val_attr = mat
        return mat

    def class_attribute_matrix(self) -> Optional[torch.Tensor]:
        """[200, 312] continuous class-level attribute frequencies (0-100)."""
        path = self.base / "attributes" / "class_attribute_labels_continuous.txt"
        if not path.is_file():
            return None
        rows = []
        with open(path) as f:
            for line in f:
                rows.append([float(v) for v in line.split()])
        return torch.tensor(rows)
