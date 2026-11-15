"""Types an answer into Minecraft chat with simulated key presses."""

from __future__ import annotations

import random
import sys
import time


def active_window_title() -> str | None:
    """Title of the focused window, or None if we can't tell on this OS."""
    try:
        if sys.platform == "win32":
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            return buf.value
    except Exception:
        pass
    return None


class ChatSender:
    def __init__(self, chat_key: str = "t", delay: tuple[float, float] = (1.2, 2.8),
                 key_delay: tuple[float, float] = (0.03, 0.09), window_hint: str = "Minecraft",
                 dry_run: bool = False):
        self.chat_key = chat_key
        self.delay = delay
        self.key_delay = key_delay
        self.window_hint = window_hint
        self.dry_run = dry_run
        self._kb = None

    def _keyboard(self):
        if self._kb is None:
            from pynput.keyboard import Controller

            self._kb = Controller()
        return self._kb

    def minecraft_focused(self) -> bool:
        title = active_window_title()
        return title is None or self.window_hint.lower() in title.lower()

    def send(self, text: str) -> bool:
        time.sleep(random.uniform(*self.delay))  # "reading" time
        if self.dry_run:
            print(f"   (dry run) would send: {text}")
            return True
        if not self.minecraft_focused():
            print("   Minecraft is not the active window; not typing.")
            return False
        from pynput.keyboard import Key

        kb = self._keyboard()
        kb.press(self.chat_key)
        kb.release(self.chat_key)
        time.sleep(0.25)  # let the chat box open
        for ch in text:
            kb.type(ch)
            time.sleep(random.uniform(*self.key_delay))
        time.sleep(0.1)
        kb.press(Key.enter)
        kb.release(Key.enter)
        return True
