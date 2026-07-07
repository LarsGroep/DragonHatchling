"""Dataset abstraction layer (§4).

A dataset is a declarative :class:`DatasetSpec` plus a :class:`DatasetAdapter`
subclass. Everything downstream (transforms, training, packs, UI labels,
colors) derives from the spec.

The registry (register / get / list), the adapters (EuroSAT, Oxford-IIIT Pet,
imagefolder), the deterministic split logic, and the ``make_synthetic_dataset``
test utility are all implemented here.

**Import discipline:** importing this module must not require torch/torchvision
/timm/PIL. Transforms are described by a small :class:`Transform` protocol and
the concrete torchvision/timm transform objects are constructed lazily inside
``preprocess``/``augment`` (imports happen *inside* the call). The synthetic
fixture writer uses only the standard library so adapter tests need no image
libraries and never touch the network.
"""

from __future__ import annotations

import os
import random
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    runtime_checkable,
)

# --------------------------------------------------------------------------- #
# Transforms — described structurally so importing this module stays torch-free.
# --------------------------------------------------------------------------- #


@runtime_checkable
class Transform(Protocol):
    """A preprocessing/augmentation transform: ``image -> image/tensor``.

    Any callable of one argument satisfies this protocol, so plain lambdas and
    the torchvision ``Compose`` objects built lazily by the adapters both
    qualify. Defining it as a :class:`~typing.Protocol` keeps this module free
    of a hard torch/torchvision dependency at import time.
    """

    def __call__(self, image: Any) -> Any:  # pragma: no cover - structural
        ...


# Image file extensions the folder adapters recognise.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of a dataset (§4)."""

    name: str
    display_name: str
    num_classes: int
    image_size: int = 224
    channels: int = 3
    class_names: List[str] = field(default_factory=list)
    class_colors: List[str] = field(default_factory=list)
    license: str = ""
    citation: str = ""
    kaggle_sources: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.class_names and len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but "
                f"num_classes is {self.num_classes}"
            )
        if self.class_colors and len(self.class_colors) != self.num_classes:
            raise ValueError(
                f"class_colors has {len(self.class_colors)} entries but "
                f"num_classes is {self.num_classes}"
            )


@dataclass
class Sample:
    """One dataset item: an image reference plus its label and split.

    ``image`` is either a filesystem path (``str``) or an in-memory array; the
    adapters in this module always yield paths (opening the pixels is the
    transform's job, so ``load`` needs no image library).
    """

    image: Any
    label: int
    split: Optional[str] = None
    image_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitPolicy:
    """How a dataset is partitioned into train/val/test.

    Describes the *policy* (fractions, RNG seed, optional grouping key for
    leak-free grouped splits) rather than materialised index lists — the
    adapters apply it deterministically inside :meth:`DatasetAdapter.load`.
    """

    fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 1234
    group_key: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.fractions) != 3:
            raise ValueError("fractions must be a 3-tuple (train, val, test)")
        if any(f < 0 for f in self.fractions):
            raise ValueError("fractions must be non-negative")
        if abs(sum(self.fractions) - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1.0, got {self.fractions}")


@dataclass
class VizHooks:
    """Per-dataset UI extras (color overrides, legends, etc.)."""

    extras: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Deterministic, stratified split assignment.
# --------------------------------------------------------------------------- #


def _split_counts(n: int, fractions: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Split ``n`` items into (train, val, test) counts summing to ``n``.

    Uses rounding for train/val and gives the remainder to test, so the counts
    are stable and always sum to ``n`` even for tiny ``n``.
    """
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    # Clamp so the three counts stay within [0, n] and sum exactly to n.
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def deterministic_splits(
    keys: Sequence[Any],
    policy: SplitPolicy,
    *,
    salt: str = "",
) -> Dict[Any, str]:
    """Assign each key in ``keys`` to a split, deterministically.

    ``keys`` are sorted first (so the result is independent of input order),
    then shuffled with a seeded RNG whose seed folds in ``policy.seed`` and
    ``salt`` (used per-class to make the split stratified). The mapping is
    identical across processes and runs for the same inputs.
    """
    unique = sorted(set(keys), key=lambda k: str(k))
    rng = random.Random(f"{policy.seed}:{salt}")
    order = list(unique)
    rng.shuffle(order)
    n_train, n_val, _ = _split_counts(len(order), policy.fractions)
    out: Dict[Any, str] = {}
    for i, key in enumerate(order):
        if i < n_train:
            out[key] = "train"
        elif i < n_train + n_val:
            out[key] = "val"
        else:
            out[key] = "test"
    return out


def _list_images(directory: Path) -> List[Path]:
    """Return image files directly under ``directory``, sorted by name."""
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _folder_per_class(root: Path) -> Tuple[List[str], Dict[str, List[Path]]]:
    """Scan ``root/<class>/*`` → (sorted class names, class -> image paths)."""
    classes = sorted(
        p.name for p in root.iterdir() if p.is_dir() and _list_images(p)
    )
    by_class = {c: _list_images(root / c) for c in classes}
    return classes, by_class


def _stratified_split_samples(
    by_class: Dict[str, List[Path]],
    class_index: Dict[str, int],
    policy: SplitPolicy,
) -> List[Sample]:
    """Build every :class:`Sample` for a folder-per-class layout.

    The split is assigned per class (stratified), deterministically seeded, so
    per-class proportions match ``policy.fractions`` and are reproducible.
    """
    samples: List[Sample] = []
    for cls, paths in by_class.items():
        keys = [p.name for p in paths]
        mapping = deterministic_splits(keys, policy, salt=cls)
        for p in paths:
            samples.append(
                Sample(
                    image=str(p),
                    label=class_index[cls],
                    split=mapping[p.name],
                    image_id=f"{cls}/{p.stem}",
                    meta={"class_name": cls, "path": str(p)},
                )
            )
    return samples


def _filter_split(samples: Iterable[Sample], split: Optional[str]) -> List[Sample]:
    if split is None or split == "all":
        return list(samples)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS} or 'all'")
    return [s for s in samples if s.split == split]


# --------------------------------------------------------------------------- #
# Adapter ABC.
# --------------------------------------------------------------------------- #


class DatasetAdapter(ABC):
    """Adapter turning a raw on-disk dataset into the ViTreous pipeline (§4).

    Concrete adapters set a class-level :attr:`spec` and implement the data
    methods. ``preprocess``/``augment`` import torchvision lazily so that
    importing this module never requires the ML stack.
    """

    spec: DatasetSpec

    @abstractmethod
    def load(self, root: str, split: str) -> Iterable[Sample]:
        """Yield samples for a split from a raw dataset directory."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self) -> Transform:
        """Return the deterministic eval transform (lazily built)."""
        raise NotImplementedError

    @abstractmethod
    def augment(self) -> Transform:
        """Return the train-time augmentation transform (lazily built)."""
        raise NotImplementedError

    def splits(self) -> SplitPolicy:
        """Return the split policy (default seeded 80/10/10)."""
        return SplitPolicy()

    def gallery(self, root: str, n: int = 75) -> List[Sample]:
        """Return up to ``n`` curated demo images (first-n of the test split)."""
        samples = list(self.load(root, "test"))
        return samples[:n]

    def viz_hooks(self) -> VizHooks:
        """Return per-dataset UI extras."""
        return VizHooks()

    # -- shared transform factories (torchvision imported lazily) ------------ #

    def _eval_transform(self) -> Transform:
        return _build_eval_transform(self.spec.image_size)

    def _train_transform(self) -> Transform:
        return _build_train_transform(self.spec.image_size)


# --------------------------------------------------------------------------- #
# Lazy torchvision transform factories.
# --------------------------------------------------------------------------- #

# ImageNet statistics — the DeiT-S weights we fine-tune from expect these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_eval_transform(image_size: int) -> Transform:
    """Deterministic eval transform: resize → center-crop → normalize.

    torchvision/PIL are imported *inside* this function so that ``import
    vitreous`` and ``import vitreous.data`` stay torch-free.
    """
    from torchvision import transforms as T  # lazy

    resize = int(round(image_size * 256 / 224))
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _build_train_transform(image_size: int) -> Transform:
    """Train augmentation: random-resized-crop → h-flip → normalize."""
    from torchvision import transforms as T  # lazy

    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# Registry — fully functional at M0, unchanged at M1.
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}

A = TypeVar("A", bound=DatasetAdapter)


def register_dataset(name: str) -> Callable[[Type[A]], Type[A]]:
    """Class decorator registering a :class:`DatasetAdapter` under ``name``."""

    if not name:
        raise ValueError("dataset name must be a non-empty string")

    def _decorator(cls: Type[A]) -> Type[A]:
        if not isinstance(cls, type) or not issubclass(cls, DatasetAdapter):
            raise TypeError(
                f"{cls!r} must be a subclass of DatasetAdapter to register"
            )
        if name in _REGISTRY:
            raise ValueError(f"dataset {name!r} is already registered")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_dataset(name: str) -> Type[DatasetAdapter]:
    """Return the registered adapter class for ``name``."""

    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"no dataset registered as {name!r}; available: {available}"
        ) from exc


def list_datasets() -> List[str]:
    """Return the sorted names of all registered datasets."""

    return sorted(_REGISTRY)


def _clear_registry() -> None:
    """Test helper: empty the registry. Not part of the public API."""

    _REGISTRY.clear()


def _register_builtins() -> None:
    """Idempotently (re)register the adapters shipped in this module.

    The decorators register them at import time; this helper restores them if a
    test cleared the global registry (see ``_clear_registry``), so behavior does
    not depend on test ordering.
    """
    for name, cls in (
        ("eurosat", EuroSATAdapter),
        ("oxford_pet", OxfordPetAdapter),
        ("imagefolder", ImageFolderAdapter),
    ):
        if name not in _REGISTRY:
            _REGISTRY[name] = cls


# --------------------------------------------------------------------------- #
# Adapters.
# --------------------------------------------------------------------------- #

# EuroSAT's canonical 10 land-use classes (RGB variant).
_EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# A qualitative palette (one hex per class) for UI legends.
_EUROSAT_COLORS = [
    "#e6b800", "#2e7d32", "#7cb342", "#546e7a", "#b71c1c",
    "#9ccc65", "#c0ca33", "#8e24aa", "#1e88e5", "#00838f",
]


@register_dataset("eurosat")
class EuroSATAdapter(DatasetAdapter):
    """EuroSAT (Sentinel-2 land use), folder-per-class as shipped on Kaggle.

    Directory layout::

        root/<ClassName>/*.jpg

    Classes are discovered from the directory so the adapter also works on
    subsets/fixtures; the canonical spec describes the full 10-class dataset.
    A deterministic, seeded, stratified 80/10/10 split is applied.
    """

    spec = DatasetSpec(
        name="eurosat",
        display_name="EuroSAT — land use from Sentinel-2",
        num_classes=10,
        image_size=224,
        channels=3,
        class_names=list(_EUROSAT_CLASSES),
        class_colors=list(_EUROSAT_COLORS),
        license="MIT",
        citation="Helber et al. 2019",
        kaggle_sources=["apollo2506/eurosat-dataset"],
    )

    def load(self, root: str, split: str = "all") -> List[Sample]:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"EuroSAT root not found: {root}")
        classes, by_class = _folder_per_class(root_path)
        if not classes:
            raise FileNotFoundError(
                f"no class subdirectories with images under {root}"
            )
        class_index = {c: i for i, c in enumerate(classes)}
        samples = _stratified_split_samples(by_class, class_index, self.splits())
        return _filter_split(samples, split)

    def preprocess(self) -> Transform:
        return self._eval_transform()

    def augment(self) -> Transform:
        return self._train_transform()

    def viz_hooks(self) -> VizHooks:
        return VizHooks(extras={"palette": list(self.spec.class_colors)})


def _oxford_pet_class_from_stem(stem: str) -> str:
    """Oxford-IIIT Pet class = filename stem minus the trailing ``_<number>``.

    Breed names may themselves contain underscores (``american_bulldog_1`` →
    ``american_bulldog``), so we strip only the final numeric segment.
    """
    head, sep, tail = stem.rpartition("_")
    if sep and tail.isdigit():
        return head
    return stem


@register_dataset("oxford_pet")
class OxfordPetAdapter(DatasetAdapter):
    """Oxford-IIIT Pet: flat ``images/*.jpg`` with class encoded in the name.

    When the official split lists are present
    (``annotations/trainval.txt`` and ``annotations/test.txt``) they define
    train+val vs test; trainval is further split into train/val by the seeded
    policy. Without them the adapter falls back to parsing every image in
    ``images/`` and applying the full seeded 80/10/10 split.
    """

    spec = DatasetSpec(
        name="oxford_pet",
        display_name="Oxford-IIIT Pet",
        num_classes=37,
        image_size=224,
        channels=3,
        class_names=[],  # 37 breeds; discovered from disk at load time
        class_colors=[],
        license="CC BY-SA 4.0",
        citation="Parkhi et al. 2012",
        kaggle_sources=["tanlikesmath/the-oxfordiiit-pet-dataset"],
    )

    def _read_list(self, path: Path) -> List[str]:
        """Return image stems listed in an Oxford-Pet annotation file."""
        stems: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stems.append(line.split()[0])
        return stems

    def load(self, root: str, split: str = "all") -> List[Sample]:
        root_path = Path(root)
        images_dir = root_path / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Oxford-Pet images/ not found under {root}")

        all_images = _list_images(images_dir)
        if not all_images:
            raise FileNotFoundError(f"no images under {images_dir}")
        by_stem = {p.stem: p for p in all_images}

        # Discover classes from filenames (sorted, stable index).
        classes = sorted({_oxford_pet_class_from_stem(p.stem) for p in all_images})
        class_index = {c: i for i, c in enumerate(classes)}

        ann = root_path / "annotations"
        trainval_txt = ann / "trainval.txt"
        test_txt = ann / "test.txt"
        policy = self.splits()

        samples: List[Sample] = []
        if trainval_txt.is_file() and test_txt.is_file():
            trainval = [s for s in self._read_list(trainval_txt) if s in by_stem]
            test = [s for s in self._read_list(test_txt) if s in by_stem]
            # Sub-split trainval into train/val, stratified per class. We keep
            # the official test set intact and reallocate its share to
            # train/val so the trainval fractions still sum to 1.
            f_tr, f_va, _ = policy.fractions
            denom = f_tr + f_va if (f_tr + f_va) > 0 else 1.0
            sub_policy = SplitPolicy(
                fractions=(f_tr / denom, f_va / denom, 0.0),
                seed=policy.seed,
            )
            # group trainval stems by class for stratified sub-splitting
            tv_by_class: Dict[str, List[str]] = {}
            for stem in trainval:
                tv_by_class.setdefault(_oxford_pet_class_from_stem(stem), []).append(stem)
            stem_split: Dict[str, str] = {}
            for cls, stems in tv_by_class.items():
                mapping = deterministic_splits(stems, sub_policy, salt=cls)
                stem_split.update(mapping)
            for stem in trainval:
                p = by_stem[stem]
                cls = _oxford_pet_class_from_stem(stem)
                samples.append(
                    Sample(
                        image=str(p),
                        label=class_index[cls],
                        split=stem_split[stem],
                        image_id=stem,
                        meta={"class_name": cls, "path": str(p), "list": "trainval"},
                    )
                )
            for stem in test:
                p = by_stem[stem]
                cls = _oxford_pet_class_from_stem(stem)
                samples.append(
                    Sample(
                        image=str(p),
                        label=class_index[cls],
                        split="test",
                        image_id=stem,
                        meta={"class_name": cls, "path": str(p), "list": "test"},
                    )
                )
        else:
            # Fallback: seeded stratified split over all images by filename.
            by_class: Dict[str, List[Path]] = {}
            for p in all_images:
                by_class.setdefault(_oxford_pet_class_from_stem(p.stem), []).append(p)
            for cls, paths in by_class.items():
                keys = [p.stem for p in paths]
                mapping = deterministic_splits(keys, policy, salt=cls)
                for p in paths:
                    samples.append(
                        Sample(
                            image=str(p),
                            label=class_index[cls],
                            split=mapping[p.stem],
                            image_id=p.stem,
                            meta={"class_name": cls, "path": str(p), "list": "filename"},
                        )
                    )

        return _filter_split(samples, split)

    def preprocess(self) -> Transform:
        return self._eval_transform()

    def augment(self) -> Transform:
        return self._train_transform()


# HAM10000 diagnosis codes -> human-readable lesion names (7 classes).
_HAM10000_DX = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesion",
}
# Stable class order (sorted by code) so label indices are reproducible.
_HAM10000_CODES = sorted(_HAM10000_DX)  # akiec, bcc, bkl, df, mel, nv, vasc
_HAM10000_CLASSES = [_HAM10000_DX[c] for c in _HAM10000_CODES]
_HAM10000_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
]


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of dict rows (stdlib csv, utf-8)."""
    import csv

    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@register_dataset("ham10000")
class HAM10000Adapter(DatasetAdapter):
    """HAM10000 dermatoscopy skin lesions (7 diagnostic classes).

    The Kaggle ``kmader/skin-cancer-mnist-ham10000`` layout is a metadata CSV
    plus two image folders::

        root/HAM10000_metadata.csv
        root/HAM10000_images_part_1/*.jpg
        root/HAM10000_images_part_2/*.jpg

    Each row has an ``image_id`` (the jpg stem), a ``dx`` diagnosis code, and a
    ``lesion_id`` — several images can belong to the same physical lesion. The
    split is **grouped by ``lesion_id``** (a lesion's images all land in one
    split) so near-duplicate views of the same lesion never leak across
    train/val/test. Grouping is stratified by the lesion's diagnosis and
    seeded, so proportions are stable and reproducible.

    CSV/folder names are auto-detected case-insensitively; if the two
    ``part_*`` folders are absent the adapter falls back to any ``*.jpg`` found
    recursively under ``root``.
    """

    spec = DatasetSpec(
        name="ham10000",
        display_name="HAM10000 — dermatoscopy skin lesions",
        num_classes=7,
        image_size=224,
        channels=3,
        class_names=list(_HAM10000_CLASSES),
        class_colors=list(_HAM10000_COLORS),
        license="CC BY-NC 4.0",
        citation="Tschandl et al. 2018",
        kaggle_sources=["kmader/skin-cancer-mnist-ham10000"],
    )

    def _find_metadata(self, root: Path) -> Path:
        for p in sorted(root.rglob("*.csv")):
            if "metadata" in p.name.lower():
                return p
        raise FileNotFoundError(f"HAM10000 metadata CSV not found under {root}")

    def _index_images(self, root: Path) -> Dict[str, Path]:
        """Map image_id (jpg stem) -> path, across both part folders."""
        by_id: Dict[str, Path] = {}
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                by_id.setdefault(p.stem, p)
        return by_id

    def load(self, root: str, split: str = "all") -> List[Sample]:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"HAM10000 root not found: {root}")

        rows = _read_csv_rows(self._find_metadata(root_path))
        if not rows:
            raise FileNotFoundError("HAM10000 metadata CSV is empty")
        by_id = self._index_images(root_path)
        class_index = {code: i for i, code in enumerate(_HAM10000_CODES)}

        # Determine each lesion's diagnosis (stratify grouped split by dx).
        lesion_dx: Dict[str, str] = {}
        for r in rows:
            lesion = r.get("lesion_id") or r["image_id"]
            lesion_dx.setdefault(lesion, r["dx"])

        # Assign a split per lesion, stratified by dx and seeded.
        policy = self.splits()
        by_dx: Dict[str, List[str]] = {}
        for lesion, dx in lesion_dx.items():
            by_dx.setdefault(dx, []).append(lesion)
        lesion_split: Dict[str, str] = {}
        for dx, lesions in by_dx.items():
            lesion_split.update(deterministic_splits(lesions, policy, salt=dx))

        samples: List[Sample] = []
        for r in rows:
            dx = r["dx"]
            if dx not in class_index:
                continue  # unknown code — skip defensively
            img_id = r["image_id"]
            path = by_id.get(img_id)
            if path is None:
                continue  # metadata row with no image on disk
            lesion = r.get("lesion_id") or img_id
            samples.append(
                Sample(
                    image=str(path),
                    label=class_index[dx],
                    split=lesion_split[lesion],
                    image_id=img_id,
                    meta={
                        "class_name": _HAM10000_DX[dx],
                        "dx": dx,
                        "lesion_id": lesion,
                        "path": str(path),
                    },
                )
            )
        if not samples:
            raise FileNotFoundError(
                f"no HAM10000 samples matched between CSV and images under {root}"
            )
        return _filter_split(samples, split)

    def splits(self) -> SplitPolicy:
        # Grouped split keyed on lesion_id — declared for downstream tooling;
        # the grouping is applied per-lesion in load().
        return SplitPolicy(fractions=(0.8, 0.1, 0.1), seed=1234, group_key="lesion_id")

    def preprocess(self) -> Transform:
        return self._eval_transform()

    def augment(self) -> Transform:
        return self._train_transform()

    def viz_hooks(self) -> VizHooks:
        return VizHooks(extras={"palette": list(self.spec.class_colors)})


@register_dataset("imagefolder")
class ImageFolderAdapter(DatasetAdapter):
    """Generic ImageFolder adapter.

    Two layouts, auto-detected:

    * **Pre-split**: ``root/train/<class>/*`` and ``root/val/<class>/*`` (and
      optionally ``root/test/<class>/*``). Each split is read straight from its
      directory; classes are the union across split dirs.
    * **Flat**: ``root/<class>/*`` → a deterministic seeded 80/10/10 split,
      exactly like the EuroSAT layout.
    """

    spec = DatasetSpec(
        name="imagefolder",
        display_name="Generic ImageFolder",
        num_classes=1,  # placeholder; real value discovered from disk
        image_size=224,
        channels=3,
        class_names=[],
        class_colors=[],
        license="",
        citation="",
        kaggle_sources=[],
    )

    @staticmethod
    def _is_presplit(root: Path) -> bool:
        return (root / "train").is_dir() and (root / "val").is_dir()

    def _presplit_classes(self, root: Path) -> List[str]:
        classes: set = set()
        for sp in SPLITS:
            d = root / sp
            if d.is_dir():
                classes.update(
                    p.name for p in d.iterdir() if p.is_dir() and _list_images(p)
                )
        return sorted(classes)

    def load(self, root: str, split: str = "all") -> List[Sample]:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"imagefolder root not found: {root}")

        if self._is_presplit(root_path):
            classes = self._presplit_classes(root_path)
            if not classes:
                raise FileNotFoundError(f"no class dirs under {root}/train|val")
            class_index = {c: i for i, c in enumerate(classes)}
            samples: List[Sample] = []
            for sp in SPLITS:
                split_dir = root_path / sp
                if not split_dir.is_dir():
                    continue
                for cls in classes:
                    for p in _list_images(split_dir / cls):
                        samples.append(
                            Sample(
                                image=str(p),
                                label=class_index[cls],
                                split=sp,
                                image_id=f"{sp}/{cls}/{p.stem}",
                                meta={"class_name": cls, "path": str(p)},
                            )
                        )
            return _filter_split(samples, split)

        # Flat layout → seeded split.
        classes, by_class = _folder_per_class(root_path)
        if not classes:
            raise FileNotFoundError(
                f"no class subdirectories with images under {root}"
            )
        class_index = {c: i for i, c in enumerate(classes)}
        samples = _stratified_split_samples(by_class, class_index, self.splits())
        return _filter_split(samples, split)

    def preprocess(self) -> Transform:
        return self._eval_transform()

    def augment(self) -> Transform:
        return self._train_transform()


# --------------------------------------------------------------------------- #
# Synthetic fixtures — pure-stdlib PNG writer so tests need no image library.
# --------------------------------------------------------------------------- #


def _png_bytes(width: int, height: int, rgb: Tuple[int, int, int]) -> bytes:
    """Encode a solid-color ``width``×``height`` RGB image as PNG (stdlib only)."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    # IHDR: 8-bit RGB, no interlace.
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytes(rgb) * width
    raw = b"".join(b"\x00" + row for _ in range(height))  # filter byte 0 per row
    idat = zlib.compress(raw, 9)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _class_color(i: int) -> Tuple[int, int, int]:
    """A distinct-ish solid color per class index (deterministic)."""
    return (40 + (i * 60) % 200, 30 + (i * 90) % 200, 20 + (i * 130) % 200)


def make_synthetic_dataset(
    root: str,
    layout: str,
    *,
    num_classes: int = 3,
    per_class: int = 10,
    image_size: int = 8,
    with_annotations: bool = True,
) -> str:
    """Write a tiny fixture tree mimicking a real dataset layout.

    Parameters
    ----------
    root:
        Destination directory (created if missing).
    layout:
        One of ``"eurosat"`` / ``"folder_per_class"`` (``root/<class>/*.png``),
        ``"oxford_pet"`` (``root/images/*.png`` + optional annotation lists),
        ``"imagefolder_flat"`` (same as folder_per_class), or
        ``"imagefolder_split"`` (``root/{train,val,test}/<class>/*.png``).
    num_classes, per_class:
        Number of classes and images per class.
    image_size:
        Side length of each solid-color square PNG.
    with_annotations:
        For ``oxford_pet`` only: also write ``annotations/{trainval,test}.txt``.

    Returns
    -------
    str
        The ``root`` path (for convenience).
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    classes = [f"class_{chr(ord('a') + i)}" for i in range(num_classes)]

    def _write_img(path: Path, cls_idx: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png_bytes(image_size, image_size, _class_color(cls_idx)))

    if layout in ("eurosat", "folder_per_class", "imagefolder_flat"):
        for ci, cls in enumerate(classes):
            for k in range(per_class):
                _write_img(root_path / cls / f"{cls}_{k:03d}.png", ci)

    elif layout == "imagefolder_split":
        # Deterministic per-split counts mirroring an 80/10/10 policy.
        n_train, n_val, n_test = _split_counts(per_class, (0.8, 0.1, 0.1))
        counts = {"train": n_train, "val": n_val, "test": n_test}
        idx = 0
        for ci, cls in enumerate(classes):
            for sp, n in counts.items():
                for _ in range(n):
                    _write_img(root_path / sp / cls / f"{cls}_{idx:03d}.png", ci)
                    idx += 1

    elif layout == "oxford_pet":
        images_dir = root_path / "images"
        stems: List[Tuple[str, str]] = []  # (stem, class) in creation order
        for ci, cls in enumerate(classes):
            for k in range(per_class):
                # Oxford naming: <Class>_<n>.jpg — here PNG. Class has an
                # underscore so the parser's strip-trailing-number is exercised.
                stem = f"{cls}_{k + 1}"
                _write_img(images_dir / f"{stem}.png", ci)
                stems.append((stem, cls))
        if with_annotations:
            ann = root_path / "annotations"
            ann.mkdir(parents=True, exist_ok=True)
            # Put the last image of each class into test, the rest into trainval.
            trainval, test = [], []
            per: Dict[str, List[str]] = {}
            for stem, cls in stems:
                per.setdefault(cls, []).append(stem)
            for cls, cls_stems in per.items():
                test.append(cls_stems[-1])
                trainval.extend(cls_stems[:-1])
            cls_id = {c: i + 1 for i, c in enumerate(classes)}
            (ann / "trainval.txt").write_text(
                "".join(
                    f"{s} {cls_id[_oxford_pet_class_from_stem(s)]} 1 1\n" for s in trainval
                ),
                encoding="utf-8",
            )
            (ann / "test.txt").write_text(
                "".join(
                    f"{s} {cls_id[_oxford_pet_class_from_stem(s)]} 1 1\n" for s in test
                ),
                encoding="utf-8",
            )
    elif layout == "ham10000":
        # CSV + two image-part folders. Two images per lesion (same dx) so the
        # grouped split is exercised; dx codes cycle through the real 7.
        part1 = root_path / "HAM10000_images_part_1"
        part2 = root_path / "HAM10000_images_part_2"
        codes = _HAM10000_CODES
        rows = ["lesion_id,image_id,dx,dx_type,age,sex,localization"]
        idx = 0
        for ci in range(num_classes):
            dx = codes[ci % len(codes)]
            for k in range(per_class):
                lesion = f"HAM_{ci:02d}_{k:03d}"
                # two views of the lesion, split across the two part folders
                for v, part in enumerate((part1, part2)):
                    img_id = f"ISIC_{idx:07d}"
                    _write_img(part / f"{img_id}.png", ci)
                    rows.append(f"{lesion},{img_id},{dx},histo,50,male,back")
                    idx += 1
        (root_path / "HAM10000_metadata.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )

    else:
        raise ValueError(f"unknown synthetic layout: {layout!r}")

    return str(root_path)


__all__ = [
    "Transform",
    "IMAGE_EXTS",
    "SPLITS",
    "DatasetSpec",
    "Sample",
    "SplitPolicy",
    "VizHooks",
    "DatasetAdapter",
    "EuroSATAdapter",
    "OxfordPetAdapter",
    "HAM10000Adapter",
    "ImageFolderAdapter",
    "register_dataset",
    "get_dataset",
    "list_datasets",
    "deterministic_splits",
    "make_synthetic_dataset",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
