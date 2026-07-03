"""Built-in dataset loaders: torchvision classics + generic ImageFolder."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from hatchvision.data.base import DatasetLoader, DatasetSpec, register_loader


def train_transforms(spec: DatasetSpec, augment: bool = True):
    """Standard augmentation derived purely from the spec."""
    mean, std = spec.normalization()
    ops = []
    if spec.image_size <= 64:
        if augment:
            ops += [
                # square first: photo datasets aren't pre-sized like CIFAR
                transforms.Resize((spec.image_size, spec.image_size)),
                transforms.RandomCrop(spec.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            ops += [transforms.Resize((spec.image_size, spec.image_size))]
    else:
        if augment:
            ops += [
                transforms.RandomResizedCrop(spec.image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
            ]
        else:
            ops += [
                transforms.Resize(int(spec.image_size * 1.14)),
                transforms.CenterCrop(spec.image_size),
            ]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def eval_transforms(spec: DatasetSpec):
    return train_transforms(spec, augment=False)


class _TorchvisionLoader(DatasetLoader):
    """Shared wiring for torchvision train/test-style datasets."""

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
        **_,
    ) -> None:
        super().__init__(limit_train=limit_train, limit_val=limit_val)
        self.root = root
        self.download = download

    def _make(self, train: bool) -> Dataset:  # pragma: no cover - overridden
        raise NotImplementedError

    def train_dataset(self) -> Dataset:
        return self._make(train=True)

    def val_dataset(self) -> Dataset:
        return self._make(train=False)


@register_loader("cifar10")
class Cifar10Loader(_TorchvisionLoader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spec = DatasetSpec(
            name="cifar10",
            num_classes=10,
            class_names=(
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck",
            ),
            image_size=32,
            in_channels=3,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        )

    def _make(self, train: bool) -> Dataset:
        tf = train_transforms(self.spec) if train else eval_transforms(self.spec)
        return datasets.CIFAR10(self.root, train=train, download=self.download, transform=tf)


@register_loader("cifar100")
class Cifar100Loader(_TorchvisionLoader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Class names come from the dataset once instantiated; use metadata
        # from torchvision lazily to avoid a download at construction time.
        self._names: Optional[Tuple[str, ...]] = None
        self.spec = DatasetSpec(
            name="cifar100",
            num_classes=100,
            class_names=tuple(f"class_{i}" for i in range(100)),
            image_size=32,
            in_channels=3,
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
        )

    def _make(self, train: bool) -> Dataset:
        tf = train_transforms(self.spec) if train else eval_transforms(self.spec)
        ds = datasets.CIFAR100(self.root, train=train, download=self.download, transform=tf)
        if self._names is None:
            import dataclasses

            self._names = tuple(ds.classes)
            self.spec = dataclasses.replace(self.spec, class_names=self._names)
        return ds


@register_loader("fashion_mnist")
class FashionMnistLoader(_TorchvisionLoader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spec = DatasetSpec(
            name="fashion_mnist",
            num_classes=10,
            class_names=(
                "t-shirt", "trouser", "pullover", "dress", "coat",
                "sandal", "shirt", "sneaker", "bag", "ankle boot",
            ),
            image_size=28,
            in_channels=1,
            mean=(0.2860,),
            std=(0.3530,),
        )

    def _make(self, train: bool) -> Dataset:
        tf = train_transforms(self.spec) if train else eval_transforms(self.spec)
        return datasets.FashionMNIST(
            self.root, train=train, download=self.download, transform=tf
        )


@register_loader("imagefolder")
class ImageFolderLoader(DatasetLoader):
    """Any directory tree of ``train/<class>/*`` and ``val/<class>/*``."""

    def __init__(
        self,
        root: str = "./data",
        image_size: int = 224,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
        **_,
    ) -> None:
        super().__init__(limit_train=limit_train, limit_val=limit_val)
        self.root = Path(root)
        train_dir = self.root / "train"
        if not train_dir.is_dir():
            raise FileNotFoundError(f"expected {train_dir} to exist")
        classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        self.spec = DatasetSpec(
            name=self.root.name,
            num_classes=len(classes),
            class_names=tuple(classes),
            image_size=image_size,
            in_channels=3,
            mean=tuple(mean),
            std=tuple(std),
        )

    def train_dataset(self) -> Dataset:
        return datasets.ImageFolder(self.root / "train", train_transforms(self.spec))

    def val_dataset(self) -> Dataset:
        val_dir = self.root / "val"
        if not val_dir.is_dir():
            val_dir = self.root / "test"
        return datasets.ImageFolder(val_dir, eval_transforms(self.spec))
