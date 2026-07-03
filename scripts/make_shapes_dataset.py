#!/usr/bin/env python3
"""Generate a small procedural shapes dataset (imagefolder layout).

Used for CI smoke runs and to give the deployed web app a self-contained
demo bundle in environments where real datasets can't be downloaded.
Six shape classes, random colors/positions/sizes on noisy backgrounds::

    python scripts/make_shapes_dataset.py --out data/shapes
    python scripts/train.py --dataset imagefolder --root data/shapes \
        --image-size 64 --backbone simple_cnn --epochs 4 --export-bundle webapp
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

CLASSES = ["circle", "cross", "ring", "square", "star", "triangle"]


def draw_shape(draw: ImageDraw.ImageDraw, name: str, cx, cy, r, color) -> None:
    if name == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif name == "ring":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color,
                     width=max(2, r // 3))
    elif name == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif name == "cross":
        w = max(2, r // 2)
        draw.rectangle([cx - r, cy - w, cx + r, cy + w], fill=color)
        draw.rectangle([cx - w, cy - r, cx + w, cy + r], fill=color)
    elif name == "triangle":
        pts = [
            (cx + r * math.cos(a), cy + r * math.sin(a))
            for a in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)
        ]
        draw.polygon(pts, fill=color)
    elif name == "star":
        pts = []
        for i in range(10):
            rad = r if i % 2 == 0 else r * 0.45
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
        draw.polygon(pts, fill=color)


def make_image(name: str, size: int, rng: random.Random) -> Image.Image:
    bg = tuple(rng.randint(150, 235) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    # background speckle so the model can't trivially threshold
    for _ in range(rng.randint(10, 30)):
        x, y = rng.randint(0, size - 1), rng.randint(0, size - 1)
        c = tuple(min(255, v + rng.randint(-25, 25)) for v in bg)
        draw.point((x, y), fill=c)
    color = tuple(rng.randint(0, 130) for _ in range(3))
    r = rng.randint(size // 6, size // 3)
    cx = rng.randint(r + 2, size - r - 2)
    cy = rng.randint(r + 2, size - r - 2)
    draw_shape(draw, name, cx, cy, r, color)
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/shapes")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--train", type=int, default=250, help="images per class")
    p.add_argument("--val", type=int, default=50, help="images per class")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    for split, count in (("train", args.train), ("val", args.val)):
        for cls in CLASSES:
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                make_image(cls, args.size, rng).save(d / f"{cls}_{i:04d}.png")
    total = (args.train + args.val) * len(CLASSES)
    print(f"wrote {total} images to {out}")


if __name__ == "__main__":
    main()
