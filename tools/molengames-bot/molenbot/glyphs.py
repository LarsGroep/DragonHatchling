"""Pixel-exact reader for Minecraft chat text.

Minecraft draws chat with a bitmap font, scaled by an integer GUI scale and
never anti-aliased. So every character is a fixed pattern of pixels. Instead
of general OCR (Tesseract mixes up c/e, l/I and similar), we:

1. keep only pixels in a known chat text colour (the drop shadow and
   background are darker and get dropped),
2. find the GUI scale, cut the image into lines and the lines into glyphs,
3. shrink each glyph back to 1 px per font pixel and look up the exact
   bitmap in a font table learned during calibration.

The font table maps a key ("<baseline offset>|<w>|<h>|<bits>") to a character.
It is built from your own screen with ``python -m molenbot calibrate``, so no
game assets are shipped with this tool.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The 16 Minecraft chat colours (§0-§f). §0 black is left out: it can't be
# told apart from the chat background.
MC_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 170), (0, 170, 0), (0, 170, 170), (170, 0, 0), (170, 0, 170),
    (255, 170, 0), (170, 170, 170), (85, 85, 85), (85, 85, 255),
    (85, 255, 85), (85, 255, 255), (255, 85, 85), (255, 85, 255),
    (255, 255, 85), (255, 255, 255),
]

UNKNOWN = "�"  # shown for glyphs the font table doesn't know yet


def text_mask(rgb: np.ndarray, colors=None, tol: int = 3) -> np.ndarray:
    """Return True where a pixel has (nearly) one of the chat text colours."""
    colors = colors or MC_COLORS
    img = rgb[..., :3].astype(np.int16)
    mask = np.zeros(img.shape[:2], dtype=bool)
    for c in colors:
        mask |= np.abs(img - np.asarray(c, dtype=np.int16)).sum(axis=2) <= tol
    return mask


def _shadow_ok(rgb: np.ndarray, mask: np.ndarray, s: int, tol: int = 40) -> np.ndarray:
    """Keep text pixels whose drop shadow (1 font px right+down, colour / 4)
    is where Minecraft would draw it -- or is covered by more text."""
    img = rgb[..., :3].astype(np.int16)
    h, w = mask.shape
    out = np.zeros_like(mask)
    src, dst = img[: h - s, : w - s], img[s:, s:]
    ok = (np.abs(dst * 4 - src).sum(axis=2) <= tol) | mask[s:, s:]
    out[: h - s, : w - s] = mask[: h - s, : w - s] & ok
    return out


def _open(mask: np.ndarray, s: int) -> np.ndarray:
    """Morphological opening with an s x s square: drops specks thinner than
    one font pixel (background texture that happens to match a colour)."""
    if s <= 1:
        return mask
    h, w = mask.shape
    er = np.ones((h - s + 1, w - s + 1), dtype=bool)
    for dy in range(s):
        for dx in range(s):
            er &= mask[dy : dy + h - s + 1, dx : dx + w - s + 1]
    out = np.zeros_like(mask)
    for dy in range(s):
        for dx in range(s):
            out[dy : dy + h - s + 1, dx : dx + w - s + 1] |= er
    return out


def clean_mask(rgb: np.ndarray, colors=None, scale: int | None = None) -> tuple[int, np.ndarray]:
    """Text mask with background noise removed. Finds the GUI scale if not given
    (the scale at which the most pixels have a correct drop shadow)."""
    raw = text_mask(rgb, colors)
    if scale:
        cands = [scale]
    else:
        cands = list(range(1, 7))
    best = None
    for s in cands:
        m = _open(_shadow_ok(rgb, raw, s), s)
        if best is None or m.sum() > best[1].sum():
            best = (s, m)
    return best


def _runs(arr1d: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True values as (start, end_exclusive)."""
    out, start = [], None
    for i, v in enumerate(arr1d):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(arr1d)))
    return out


def detect_scale(mask: np.ndarray) -> int:
    """Find the GUI scale: most font strokes are 1 font pixel wide, so the most
    common horizontal run of text pixels has length == scale."""
    lengths: Counter[int] = Counter()
    for row in mask[:: max(1, mask.shape[0] // 200)]:
        for a, b in _runs(row):
            lengths[b - a] += 1
    if not lengths:
        return 1
    best = lengths.most_common(1)[0][0]
    return max(1, best)


@dataclass
class Glyph:
    key: str
    x: int          # left edge in image pixels
    gap_before: int  # empty font-pixel columns before this glyph (-1 = first)


@dataclass
class Line:
    top: int
    bottom: int
    glyphs: list[Glyph] = field(default_factory=list)


def _glyph_key(bits: np.ndarray, dy: int) -> str:
    h, w = bits.shape
    packed = np.packbits(bits.astype(np.uint8).ravel()).tobytes().hex()
    return f"{dy}|{w}|{h}|{packed}"


def segment(mask: np.ndarray, scale: int | None = None) -> tuple[int, list[Line]]:
    """Cut a text mask into lines of glyphs."""
    s = scale or detect_scale(mask)
    rows = mask.any(axis=1)
    # Rows that belong to the same chat line are at most 8 font px apart.
    groups: list[list[int]] = []
    for a, b in _runs(rows):
        if groups and b - groups[-1][0] <= 9 * s and a - groups[-1][1] < 2 * s:
            groups[-1][1] = b
        else:
            groups.append([a, b])

    lines: list[Line] = []
    for top, bottom in groups:
        band = mask[top:bottom]
        pieces = _runs(band.any(axis=0))
        if not pieces:
            continue
        raw = []
        for x0, x1 in pieces:
            sub = band[:, x0:x1]
            ys = np.where(sub.any(axis=1))[0]
            y0, y1 = int(ys[0]), int(ys[-1]) + 1
            # Back to 1 px per font pixel: sample the centre of each cell.
            small = sub[y0 + s // 2 : y1 : s, s // 2 :: s]
            raw.append((x0, x1, y1, small))
        # The baseline is the most common glyph bottom (descenders sit lower).
        baseline = Counter(r[2] for r in raw).most_common(1)[0][0]
        line = Line(top=top, bottom=bottom)
        prev_end = None
        for x0, x1, y1, small in raw:
            dy = round((y1 - baseline) / s)
            gap = -1 if prev_end is None else round((x0 - prev_end) / s)
            line.glyphs.append(Glyph(key=_glyph_key(small, dy), x=x0, gap_before=gap))
            prev_end = x1
        lines.append(line)
    return s, lines


class Font:
    """Learned table of glyph bitmaps -> characters."""

    SPACE_GAP = 3  # an empty run of >= this many font px is a space

    def __init__(self, table: dict[str, str] | None = None):
        self.table: dict[str, str] = dict(table or {})

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Font":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.table, indent=0, sort_keys=True), encoding="utf-8")

    # -- reading -----------------------------------------------------------
    def read_line(self, line: Line) -> str:
        out = []
        for g in line.glyphs:
            if g.gap_before >= self.SPACE_GAP:
                out.append(" ")
            out.append(self.table.get(g.key, UNKNOWN))
        # Minecraft's double quote is two apostrophe-shaped columns.
        return "".join(out).replace("''", '"')

    def read_image(self, rgb: np.ndarray, colors=None, scale: int | None = None) -> list[str]:
        s, mask = clean_mask(rgb, colors, scale)
        _, lines = segment(mask, s)
        return [self.read_line(ln) for ln in lines]

    # -- learning ----------------------------------------------------------
    def learn_line(self, line: Line, text: str) -> int:
        """Label the glyphs of a line whose (ending) text is known.

        The known text is matched against the *end* of the line, so a chat
        line like "<Steve> abcdef" can be learned from just "abcdef".
        Returns the number of glyphs learned; raises ValueError if the glyph
        count doesn't fit the text.
        """
        chars = [c for c in text.replace('"', "''") if c != " "]
        if len(chars) > len(line.glyphs):
            raise ValueError(
                f"found {len(line.glyphs)} glyphs but the text has {len(chars)} characters"
            )
        glyphs = line.glyphs[len(line.glyphs) - len(chars):]
        # Check that the spaces line up too, so a bad capture can't poison the table.
        expected_spaces = []
        i = 0
        for c in text.replace('"', "''"):
            if c == " ":
                expected_spaces.append(i)
            else:
                i += 1
        found_spaces = [i for i, g in enumerate(glyphs) if i > 0 and g.gap_before >= self.SPACE_GAP]
        if expected_spaces != found_spaces:
            raise ValueError("spaces in the capture don't match the text; is the scale right?")
        pending: dict[str, str] = {}
        for g, c in zip(glyphs, chars):
            old = pending.get(g.key, self.table.get(g.key))
            if old is not None and old != c:
                raise ValueError(f"glyph already learned as {old!r}, now labelled {c!r}")
            pending[g.key] = c
        for g, c in zip(glyphs, chars):
            self.table[g.key] = c
        return len(chars)
