from hatchvision.data.base import (
    DatasetLoader,
    DatasetSpec,
    available_loaders,
    build_loader,
    register_loader,
)
from hatchvision.data import builtin as _builtin          # noqa: F401 (registers loaders)
from hatchvision.data import cub as _cub                  # noqa: F401 (registers cub200)
from hatchvision.data import isic as _isic                # noqa: F401 (registers isic)
from hatchvision.data import skin_lesion as _skin_lesion  # noqa: F401 (registers ham10000)
from hatchvision.data.builtin import eval_transforms, train_transforms
from hatchvision.data.cub import Cub200Loader
from hatchvision.data.isic import ISICLoader
from hatchvision.data.skin_lesion import Ham10000Loader

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
    "Ham10000Loader",
]
