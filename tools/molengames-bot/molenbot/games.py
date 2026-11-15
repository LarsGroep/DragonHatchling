"""Recognise MolenGames chat games and work out the answer.

Each game is a regex over the recent chat (consecutive lines joined with a
space, because the server splits the announcement over two lines). Only the
"Omgekeerd" (reverse the word) wording is confirmed from a real screenshot;
the others are educated guesses -- adjust ``patterns`` in config.json when
you see the real text (``python -m molenbot read`` shows what the bot reads).
"""

from __future__ import annotations

import ast
import operator
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .glyphs import UNKNOWN

PREFIX = "[MolenGames]"

# name -> regex. The regex must have a named group "q" (the puzzle text).
DEFAULT_PATTERNS: dict[str, str] = {
    # "- Je hebt 25 seconden om het woord "kcirB" om te keren."
    "reverse": r'woord\s*"(?P<q>[^"]+)"\s*om\s+te\s+keren',
    # guesses for other games:
    "type": r'woord\s*"(?P<q>[^"]+)"\s*(?:als\s+eerste\s+)?(?:te\s+)?(?:over)?typen',
    "unscramble": r'woord\s*"(?P<q>[^"]+)"\s*(?:te\s+)?(?:ontcijferen|ontwarren|ontrafelen|raden)',
    "math": r'(?:som|rekensom|antwoord\s+(?:op|van))\s*"?(?P<q>[-\d\s+*/x×:().^]+?)"?\s*(?:op\s+te\s+lossen|uit\s+te\s+rekenen|te\s+berekenen|\?|$)',
}


@dataclass
class Challenge:
    game: str
    question: str
    answer: str | None


# ---- solvers ---------------------------------------------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("not a plain sum")


def solve_math(expr: str) -> str | None:
    expr = expr.replace("×", "*").replace("x", "*").replace(":", "/").replace("^", "**")
    try:
        val = _eval(ast.parse(expr.strip(), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
    if isinstance(val, float):
        return str(int(val)) if val.is_integer() else f"{val:.2f}".rstrip("0").rstrip(".")
    return str(val)


class WordList:
    """Words for the unscramble game: a list you supply plus every word the
    bot sees in other games (saved, so it gets better over time)."""

    def __init__(self, paths: list[str | Path], learned_path: str | Path | None = None):
        self.words: set[str] = set()
        self.learned_path = Path(learned_path) if learned_path else None
        for p in [*paths, *( [self.learned_path] if self.learned_path else [])]:
            p = Path(p)
            if p.exists():
                self.words.update(
                    w.strip() for w in p.read_text(encoding="utf-8", errors="ignore").splitlines() if w.strip()
                )

    def learn(self, word: str) -> None:
        if word and word not in self.words and UNKNOWN not in word:
            self.words.add(word)
            if self.learned_path:
                with self.learned_path.open("a", encoding="utf-8") as f:
                    f.write(word + "\n")

    def unscramble(self, scrambled: str) -> str | None:
        key = sorted(scrambled.lower())
        hits = {w for w in self.words if len(w) == len(scrambled) and sorted(w.lower()) == key}
        hits.discard(scrambled)
        return hits.pop() if len(hits) == 1 else None


# ---- detection -------------------------------------------------------------


def has_prefix(line: str) -> bool:
    """True if the line starts with [MolenGames]. Glyphs the font table doesn't
    know (e.g. the bold prefix before you calibrated it) count as wildcards;
    every known glyph must match, so a player message ("<Name> ...",
    "[VIP] Name: ...") won't pass."""
    head = line.lstrip()[: len(PREFIX)]
    if len(head) < len(PREFIX):
        return False
    return all(a == b for a, b in zip(head, PREFIX) if a != UNKNOWN)


class Detector:
    def __init__(self, patterns: dict[str, str] | None = None, words: WordList | None = None,
                 require_prefix: bool = True, cooldown: float = 60.0):
        self.patterns = {k: re.compile(v, re.IGNORECASE) for k, v in (patterns or DEFAULT_PATTERNS).items()}
        self.words = words or WordList([])
        self.require_prefix = require_prefix
        self.cooldown = cooldown
        self._seen: dict[tuple[str, str], float] = {}

    def solve(self, game: str, q: str) -> str | None:
        if UNKNOWN in q:
            return None  # never send a guess built on a glyph we couldn't read
        if game == "reverse":
            self.words.learn(q[::-1])
            return q[::-1]
        if game == "type":
            self.words.learn(q)
            return q
        if game == "unscramble":
            return self.words.unscramble(q)
        if game == "math":
            return solve_math(q)
        return None

    def find(self, lines: list[str], now: float | None = None) -> list[Challenge]:
        """Look at the recent chat lines (oldest first) and return challenges
        that haven't been handled yet."""
        now = time.monotonic() if now is None else now
        found: list[Challenge] = []
        for i, line in enumerate(lines):
            if self.require_prefix and not has_prefix(line):
                continue
            # The game text can continue on the next couple of lines; stop at
            # the next line that starts a new message.
            parts = [line]
            for nxt in lines[i + 1 : i + 4]:
                if has_prefix(nxt) or nxt.lstrip().startswith("<"):
                    break
                parts.append(nxt)
            text = " ".join(parts)
            for game, rx in self.patterns.items():
                m = rx.search(text)
                if not m:
                    continue
                q = m.group("q").strip()
                key = (game, q)
                if now - self._seen.get(key, -1e9) < self.cooldown:
                    continue
                self._seen[key] = now
                found.append(Challenge(game, q, self.solve(game, q)))
        return found
