from hatchvision.data.base import (
    DatasetLoader,
    DatasetSpec,
    available_loaders,
    build_loader,
    register_loader,
)
from hatchvision.data import builtin as _builtin  # noqa: F401 (registers loaders)
from hatchvision.data import cub as _cub          # noqa: F401 (registers cub200)
from hatchvision.data import isic as _isic        # noqa: F401 (registers isic)
from hatchvision.data.builtin import eval_transforms, train_transforms
from hatchvision.data.cub import Cub200Loader
from hatchvision.data.isic import ISICLoader

__all__ = [
    "DatasetLoader",
    "DatasetSpec",
    "available_loaders",
    "build_loader",
    "register_loader",
    "train_transforms",
    "eval_transforms",
    "Cub200Loader",
    "ISICLoader",
]
