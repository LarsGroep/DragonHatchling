"""Sends an answer into Minecraft chat via clipboard paste."""

from __future__ import annotations

import random
import sys
import time


def active_window_title() -> str | None:
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


def _set_clipboard(text: str) -> None:
    import pyperclip
    pyperclip.copy(text)


class ChatSender:
    def __init__(self, chat_key: str = "t", delay: float | list = 0.0,
                 window_hint: str = "Minecraft", dry_run: bool = False):
        self.chat_key = chat_key
        # delay can be a fixed float or [min, max] for a random range
        self.delay = delay
        self.window_hint = window_hint
        self.dry_run = dry_run
        self._kb = None

    def _actual_delay(self) -> float:
        if isinstance(self.delay, (list, tuple)) and len(self.delay) == 2:
            return random.uniform(self.delay[0], self.delay[1])
        return float(self.delay)

    def _keyboard(self):
        if self._kb is None:
            from pynput.keyboard import Controller
            self._kb = Controller()
        return self._kb

    def minecraft_focused(self) -> bool:
        title = active_window_title()
        return title is None or self.window_hint.lower() in title.lower()

    def send(self, text: str) -> bool:
        d = self._actual_delay()
        if d > 0:
            time.sleep(d)
        if self.dry_run:
            print(f"   (dry run) would send: {text}")
            return True
        if not self.minecraft_focused():
            print("   Minecraft is not the active window; not typing.")
            return False

        from pynput.keyboard import Key

        # Put answer on clipboard BEFORE opening chat (pyperclip ~4ms after warmup)
        _set_clipboard(text)

        kb = self._keyboard()

        # Open chat
        kb.press(self.chat_key)
        kb.release(self.chat_key)
        time.sleep(0.15)  # wait for chat box to open

        # Ctrl+V — instant paste regardless of answer length
        kb.press(Key.ctrl)
        kb.press("v")
        kb.release("v")
        kb.release(Key.ctrl)
        time.sleep(0.05)

        # Send
        kb.press(Key.enter)
        kb.release(Key.enter)
        return True
