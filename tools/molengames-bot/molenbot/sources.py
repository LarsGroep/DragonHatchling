"""Where chat text comes from: the screen (computer vision) or the game log."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from .glyphs import Font

_FORMAT_CODE = re.compile(r"§.")
# "[18:20:01] [Render thread/INFO]: [System] [CHAT] message"
_LOG_CHAT = re.compile(r"^\[[^\]]*\]\s*\[[^\]]*\]:\s*(?:\[System\]\s*)?\[CHAT\]\s?(.*)$")
_LOG_ANY = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]")


def grab(region: dict) -> np.ndarray:
    """Screenshot of region {left, top, width, height} as an RGB array."""
    import mss

    with mss.mss() as sct:
        shot = sct.grab(region)
    return np.asarray(shot)[..., [2, 1, 0]]  # BGRA -> RGB


class ScreenSource:
    """Reads the chat by looking at the screen. Each poll returns every line
    currently visible in the chat region (oldest first)."""

    snapshot = True  # lines replace the previous view rather than append

    def __init__(self, region: dict, font: Font, colors=None, scale: int | None = None):
        self.region, self.font, self.colors, self.scale = region, font, colors, scale

    def poll(self) -> list[str]:
        return self.font.read_image(grab(self.region), self.colors, self.scale)


def default_log_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", "")) / ".minecraft"
    elif os.uname().sysname == "Darwin":
        base = Path.home() / "Library/Application Support/minecraft"
    else:
        base = Path.home() / ".minecraft"
    return base / "logs" / "latest.log"


class LogSource:
    """Reads chat lines from Minecraft's latest.log as they are written.
    Exact text, no OCR -- use this when your client writes the log."""

    snapshot = False

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_log_path()
        self._f = None
        self._in_chat = False

    def _open(self):
        self._f = self.path.open("r", encoding="utf-8", errors="replace")
        self._f.seek(0, os.SEEK_END)  # only new messages

    def poll(self) -> list[str]:
        if self._f is None:
            if not self.path.exists():
                return []
            self._open()
        # Minecraft starts a fresh latest.log on launch: reopen when it shrinks.
        if os.stat(self.path).st_size < self._f.tell():
            self._f.close()
            self._f = self.path.open("r", encoding="utf-8", errors="replace")
        out = []
        for raw in self._f.readlines():
            raw = raw.rstrip("\r\n")
            m = _LOG_CHAT.match(raw)
            if m:
                self._in_chat = True
                out.append(_FORMAT_CODE.sub("", m.group(1)))
            elif _LOG_ANY.match(raw):
                self._in_chat = False
            elif self._in_chat and raw.strip():
                # multi-line chat message: continuation lines have no prefix
                out.append(_FORMAT_CODE.sub("", raw))
        return out
