"""Recognise MolenGames chat games and work out the answer.

Confirmed game wordings from MolenCraft Survival (latest.log):

  Omgekeerd   - Je hebt 25 seconden om het woord "X" om te keren.
  Woordhussel - Je hebt 25 seconden om het woord "X" te ontcijferen.
  Willekeurige - Je hebt 25 seconden om "X" te typen.
  Rekenspel   - Je hebt 25 seconden om "X" op te lossen.
  Aanvul      - Je hebt 25 seconden om het woord "X" aan te vullen.
  Variabelen  - Je hebt 25 seconden om de waarde van "✗" te vinden.
               followed by equation lines:  ♡ + ♡ + ♡ = 90 / ♫ + ♫ + ♫ = 75 / ♡ + ♫ + ✗ = 90
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
# All patterns confirmed from MolenCraft Survival latest.log.
DEFAULT_PATTERNS: dict[str, str] = {
    # Omgekeerd: "- Je hebt 25 seconden om het woord "kcirB" om te keren."
    "reverse": r'woord\s*"(?P<q>[^"]+)"\s*om\s+te\s+keren',
    # Woordhussel: "- Je hebt 25 seconden om het woord "oahtnPm" te ontcijferen."
    "unscramble": r'woord\s*"(?P<q>[^"]+)"\s*te\s+ontcijferen',
    # Willekeurige: "- Je hebt 25 seconden om "uNlUtA2" te typen."
    "type": r'"(?P<q>[^"]+)"\s*te\s+typen',
    # Rekenspel: "- Je hebt 25 seconden om "879 - 249" op te lossen."
    "math": r'"(?P<q>[-\d\s+*/x×:.()^]+?)"\s*op\s+te\s+lossen',
    # Aanvul: "- Je hebt 25 seconden om het woord "_eacon" aan te vullen."
    "fill": r'woord\s*"(?P<q>[^"]+)"\s*aan\s+te\s+vullen',
    # Variabelen: "- Je hebt 25 seconden om de waarde van "✗" te vinden."
    # followed by 3 equation lines with emoji symbols.
    "variables": r'waarde\s+van\s+"(?P<q>[^"]+)"\s+te\s+vinden',
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


def solve_variables(lines: list[str], target: str) -> str | None:
    """Solve a Variabelen puzzle.

    Equation lines look like:  ♡ + ♡ + ♡ = 90
    The first N-1 lines define symbol values; the last line contains the
    target symbol (✗ by default) whose value we return.
    """
    eq_re = re.compile(r"(.+?)\s*=\s*(-?\d+(?:\.\d+)?)")
    known: dict[str, float] = {}
    last_line = None

    for ln in lines:
        m = eq_re.search(ln)
        if not m:
            continue
        lhs, rhs = m.group(1).strip(), float(m.group(2))
        # Extract unique symbols on the left side
        symbols = list(dict.fromkeys(c for c in lhs if c.strip() and c not in "+-×*/ "))
        if not symbols:
            continue
        if len(symbols) == 1 and target not in symbols:
            # Simple: sym + sym + sym = N  →  sym = N / count
            count = lhs.count(symbols[0])
            known[symbols[0]] = rhs / count
        else:
            last_line = (symbols, lhs, rhs)

    if last_line is None:
        return None
    symbols, lhs, rhs = last_line
    # Substitute known values and solve for the unknown
    val = rhs
    for sym in symbols:
        if sym == target:
            continue
        if sym not in known:
            return None
        # count how many times this symbol appears
        val -= known[sym] * lhs.count(sym)
    # remaining val = target * its_count
    count = lhs.count(target)
    if count == 0:
        return None
    result = val / count
    return str(int(result)) if result == int(result) else f"{result:.2f}".rstrip("0").rstrip(".")


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

    def fill(self, pattern: str) -> str | None:
        """Find a word matching a pattern like '_eacon' (underscores are wildcards)."""
        import re as _re
        rx = _re.compile(
            "^" + "".join(f"[{c}]" if c != "_" else "[a-zA-Z]" for c in pattern) + "$",
            _re.IGNORECASE,
        )
        hits = {w for w in self.words if rx.match(w)}
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
        if game == "fill":
            return self.words.fill(q)
        return None

    def solve_variables(self, target: str, context_lines: list[str]) -> str | None:
        return solve_variables(context_lines, target)

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
            for nxt in lines[i + 1 : i + 8]:
                if has_prefix(nxt) or nxt.lstrip().startswith("<"):
                    break
                parts.append(nxt)  # include blank lines so equation lines aren't cut off
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
                if game == "variables":
                    answer = self.solve_variables(q, parts)
                else:
                    answer = self.solve(game, q)
                found.append(Challenge(game, q, answer))
        return found
