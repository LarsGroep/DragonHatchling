"""MolenGames chat bot.

    python -m molenbot region        # draw a box around the chat once
    python -m molenbot calibrate     # teach it the font (type a few lines in chat)
    python -m molenbot read          # show what it reads right now
    python -m molenbot run --dry-run # detect + print answers, don't type
    python -m molenbot run           # detect + type answers
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .games import DEFAULT_PATTERNS, Detector, WordList
from .glyphs import UNKNOWN, Font, clean_mask, segment
from .sender import ChatSender
from .sources import LogSource, ScreenSource, grab

DEFAULTS = {
    "source": "screen",          # "screen" (computer vision) or "log" (latest.log)
    "region": None,              # {"left":..,"top":..,"width":..,"height":..}
    "scale": None,               # GUI scale; null = detect
    "extra_colors": [],          # [[r,g,b], ...] for custom hex chat colours
    "log_path": None,            # null = default .minecraft/logs/latest.log
    "font_file": "font.json",
    "wordlists": ["words.txt"],
    "learned_words": "learned_words.txt",
    "games": ["reverse", "type", "unscramble", "math"],
    "patterns": {},              # override/add regexes, see games.py
    "chat_key": "t",
    "answer_delay": [1.2, 2.8],  # seconds before typing
    "poll_interval": 0.3,
    "pause_key": "f8",
}

CALIBRATION_LINES = [
    "abcdefghijklm nopqrstuvwxyz",
    "ABCDEFGHIJKLM NOPQRSTUVWXYZ",
    "0123456789 .,!?'-+*=:;()",
]


def load_config(path: Path) -> dict:
    cfg = dict(DEFAULTS)
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    return cfg


def save_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _colors(cfg):
    from .glyphs import MC_COLORS

    return MC_COLORS + [tuple(c) for c in cfg["extra_colors"]]


def _need_region(cfg):
    if not cfg["region"]:
        sys.exit("No chat region set yet. Run:  python -m molenbot region")
    return cfg["region"]


def _load_image(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


# ---- commands --------------------------------------------------------------


def cmd_region(args, cfg, cfg_path):
    if args.set:
        l, t, w, h = args.set
        cfg["region"] = {"left": l, "top": t, "width": w, "height": h}
    else:
        import cv2
        import mss

        print("Open Minecraft with chat visible (press T, or wait for messages), then come back.")
        for i in range(args.countdown, 0, -1):
            print(f"  screenshot in {i}...", end="\r")
            time.sleep(1)
        with mss.mss() as sct:
            mon = sct.monitors[args.monitor]
            shot = np.asarray(sct.grab(mon))[..., :3].copy()
        print("\nDraw a box around the chat lines, then press ENTER (C to cancel).")
        x, y, w, h = cv2.selectROI("Select the chat area", shot, showCrosshair=False)
        cv2.destroyAllWindows()
        if w == 0 or h == 0:
            sys.exit("Cancelled.")
        cfg["region"] = {"left": mon["left"] + int(x), "top": mon["top"] + int(y),
                         "width": int(w), "height": int(h)}
    save_config(cfg_path, cfg)
    print(f"Saved chat region {cfg['region']} to {cfg_path}")


def _learn_from(font: Font, rgb: np.ndarray, text: str, cfg) -> int | None:
    """Try to learn `text` from the newest matching line in the image."""
    s, mask = clean_mask(rgb, _colors(cfg), cfg["scale"])
    _, lines = segment(mask, s)
    for line in reversed(lines):  # newest chat line is at the bottom
        trial = Font(font.table)
        try:
            n = trial.learn_line(line, text)
        except ValueError:
            continue
        font.table = trial.table
        return n
    return None


def cmd_calibrate(args, cfg, cfg_path):
    font = Font.load(cfg["font_file"])
    if args.image:
        if not args.text:
            sys.exit("--image needs --text with the exact text of the last chat line in it")
        n = _learn_from(font, _load_image(args.image), args.text, cfg)
        if n is None:
            sys.exit("Couldn't match that text to a line in the image.")
        font.save(cfg["font_file"])
        print(f"Learned {n} glyphs; font now knows {len(font.table)} glyph shapes.")
        return

    region = _need_region(cfg)
    texts = [args.text] if args.text else CALIBRATION_LINES
    print("Calibration: you'll send a few chat messages so the bot learns the font.")
    print("Tip: turn Chat Text Size to 100% and Chat Text Opacity to 100% first.\n")
    for text in texts:
        print(f"Next, send this exact message in Minecraft chat:\n\n    {text}\n")
        if not args.text:
            input("Press ENTER here, then switch to Minecraft and send it... ")
        deadline = time.time() + args.timeout
        n = None
        while time.time() < deadline and n is None:
            n = _learn_from(font, grab(region), text, cfg)
            time.sleep(0.3)
        if n is None:
            print("  Didn't see it in the chat region. Skipping (run calibrate again to retry).\n")
            continue
        font.save(cfg["font_file"])
        print(f"  Learned {n} glyphs. Font now knows {len(font.table)} shapes.\n")
    print("Done. If the [MolenGames] prefix is bold, also run when a game is on screen:")
    print('    python -m molenbot calibrate --text "[MolenGames] Het Omgekeerd spel is gestart."')
    print("(any MolenGames line works; it must be the newest line with that text)")


def _make_source(cfg, font):
    if cfg["source"] == "log":
        src = LogSource(cfg["log_path"])
        print(f"Reading chat from {src.path}")
        return src
    return ScreenSource(_need_region(cfg), font, _colors(cfg), cfg["scale"])


def cmd_read(args, cfg, cfg_path):
    font = Font.load(cfg["font_file"])
    if args.image:
        lines = font.read_image(_load_image(args.image), _colors(cfg), cfg["scale"])
    else:
        lines = _make_source(cfg, font).poll()
    for ln in lines:
        print(ln)
    if any(UNKNOWN in ln for ln in lines):
        print(f"\n({UNKNOWN} = glyph not learned yet; run calibrate)")


def _make_detector(cfg) -> Detector:
    patterns = {**DEFAULT_PATTERNS, **cfg["patterns"]}
    patterns = {k: v for k, v in patterns.items() if k in cfg["games"] or k in cfg["patterns"]}
    words = WordList(cfg["wordlists"], cfg["learned_words"])
    return Detector(patterns, words)


def _start_pause_listener(cfg, state):
    try:
        from pynput import keyboard

        target = getattr(keyboard.Key, cfg["pause_key"], None)

        def on_press(key):
            if key == target:
                state["paused"] = not state["paused"]
                print("== PAUSED ==" if state["paused"] else "== RUNNING ==")

        keyboard.Listener(on_press=on_press, daemon=True).start()
    except Exception as e:  # no pynput / no display: run without the hotkey
        print(f"(pause hotkey unavailable: {e})")


def cmd_run(args, cfg, cfg_path):
    font = Font.load(cfg["font_file"])
    if cfg["source"] == "screen" and not font.table:
        sys.exit("The font isn't calibrated yet. Run:  python -m molenbot calibrate")
    source = _make_source(cfg, font)
    detector = _make_detector(cfg)
    sender = ChatSender(cfg["chat_key"], tuple(cfg["answer_delay"]), dry_run=args.dry_run)
    state = {"paused": False}
    _start_pause_listener(cfg, state)
    recent: list[str] = []
    print(f"Watching chat ({cfg['source']}). {cfg['pause_key'].upper()} pauses/resumes, Ctrl+C quits.")
    try:
        while True:
            lines = source.poll()
            recent = lines if source.snapshot else (recent + lines)[-12:]
            for ch in detector.find(recent):
                if ch.answer is None:
                    print(f"[{ch.game}] {ch.question!r}: no sure answer, skipping")
                    continue
                if state["paused"]:
                    print(f"[{ch.game}] {ch.question!r} -> {ch.answer}  (paused, not sent)")
                    continue
                print(f"[{ch.game}] {ch.question!r} -> {ch.answer}")
                threading.Thread(target=sender.send, args=(ch.answer,), daemon=True).start()
            time.sleep(cfg["poll_interval"])
    except KeyboardInterrupt:
        print("\nBye.")


def main(argv=None):
    if sys.platform == "win32":  # real pixels on scaled (125%/150%) displays
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

    ap = argparse.ArgumentParser(prog="molenbot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("region", help="select the chat area on screen")
    p.add_argument("--set", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--countdown", type=int, default=5)
    p.set_defaults(fn=cmd_region)

    p = sub.add_parser("calibrate", help="learn the chat font from your screen")
    p.add_argument("--text", help="learn a single line with this exact text")
    p.add_argument("--image", help="learn from a screenshot file instead of the screen")
    p.add_argument("--timeout", type=float, default=20)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("read", help="print the chat as the bot reads it")
    p.add_argument("--image", help="read a screenshot file instead of the screen")
    p.set_defaults(fn=cmd_read)

    p = sub.add_parser("run", help="watch chat and answer games")
    p.add_argument("--dry-run", action="store_true", help="print answers, don't type them")
    p.set_defaults(fn=cmd_run)

    args = ap.parse_args(argv)
    cfg_path = Path(args.config)
    args.fn(args, load_config(cfg_path), cfg_path)


if __name__ == "__main__":
    main()
