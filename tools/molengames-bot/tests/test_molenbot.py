from pathlib import Path

import numpy as np
from PIL import Image

from molenbot.games import Detector, WordList, has_prefix, solve_math
from molenbot.glyphs import UNKNOWN, Font, clean_mask, segment
from molenbot.sources import LogSource

FIXTURE = Path(__file__).parent / "fixtures" / "omgekeerd.png"
LINE1 = "[MolenGames] Het Omgekeerd spel is gestart."
LINE2 = '- Je hebt 25 seconden om het woord "kcirB" om te keren.'


def _rgb():
    return np.asarray(Image.open(FIXTURE).convert("RGB"))


def _font():
    s, mask = clean_mask(_rgb())
    _, lines = segment(mask, s)
    f = Font()
    f.learn_line(lines[0], LINE1)
    f.learn_line(lines[1], LINE2)
    return f


def test_segmentation_finds_scale_and_lines():
    s, mask = clean_mask(_rgb())
    _, lines = segment(mask, s)
    assert s == 2
    assert len(lines) == 2
    assert len(lines[0].glyphs) == len(LINE1.replace(" ", ""))


def test_read_back_exact():
    assert _font().read_image(_rgb()) == [LINE1, LINE2]


def test_font_is_scale_independent():
    big = np.repeat(np.repeat(_rgb(), 2, axis=0), 2, axis=1)  # GUI scale 2 -> 4
    assert _font().read_image(big) == [LINE1, LINE2]


def test_unknown_glyphs_are_marked():
    s, mask = clean_mask(_rgb())
    _, lines = segment(mask, s)
    f = Font()
    f.learn_line(lines[0], LINE1)  # line 1 has no 'c', 'B', digits...
    line2 = f.read_line(lines[1])
    assert UNKNOWN in line2 and line2.endswith("te kere" + UNKNOWN + ".")


def test_learn_rejects_wrong_text():
    s, mask = clean_mask(_rgb())
    _, lines = segment(mask, s)
    f = Font()
    try:
        f.learn_line(lines[0], "[MolenGames] Het Omgekeerd spel is gestaart")
    except ValueError:
        pass
    else:
        raise AssertionError("mislabelled line was accepted")
    assert f.table == {}


def test_detects_reverse_game_from_screen():
    det = Detector()
    [ch] = det.find(_font().read_image(_rgb()), now=0)
    assert (ch.game, ch.question, ch.answer) == ("reverse", "kcirB", "Brick")
    assert det.find(_font().read_image(_rgb()), now=5) == []  # answered once


def test_player_cannot_spoof():
    det = Detector()
    assert det.find(["<Steve> [MolenGames] woord \"abc\" om te keren"], now=0) == []
    assert det.find(['[VIP] Steve: - Je hebt 25 seconden om het woord "abc" om te keren.'], now=0) == []


def test_prefix_tolerates_unlearned_bold_glyphs():
    assert has_prefix(UNKNOWN * 12 + " Het Omgekeerd spel is gestart.")
    assert has_prefix("[" + UNKNOWN * 10 + "] Het")
    assert not has_prefix("<Steve" + UNKNOWN * 6)


def test_never_answers_with_unread_glyph():
    det = Detector()
    [ch] = det.find([LINE1, f'- Je hebt 25 seconden om het woord "kc{UNKNOWN}rB" om te keren.'], now=0)
    assert ch.answer is None


def test_math_and_unscramble(tmp_path):
    assert solve_math("12 + 7 x 3") == "33"
    assert solve_math("10 : 4") == "2.5"
    assert solve_math("__import__('os')") is None
    wl = tmp_path / "words.txt"
    wl.write_text("molen\nlemon\nbrick\n")
    words = WordList([wl])
    assert words.unscramble("kcirb") == "brick"
    assert words.unscramble("nomel") is None  # ambiguous: molen / lemon


def test_log_source(tmp_path):
    log = tmp_path / "latest.log"
    log.write_text("[18:00:00] [main/INFO]: Setting user: Steve\n")
    src = LogSource(log)
    assert src.poll() == []
    with log.open("a") as f:
        f.write("[18:20:01] [Render thread/INFO]: [System] [CHAT] §a§l[MolenGames]§r Het Omgekeerd spel is gestart.\n")
        f.write(" - Je hebt 25 seconden om het woord \"kcirB\" om te keren.\n")
        f.write("[18:20:02] [Render thread/INFO]: Something else\n")
    lines = src.poll()
    assert lines[0] == "[MolenGames] Het Omgekeerd spel is gestart."
    [ch] = Detector().find(lines, now=0)
    assert ch.answer == "Brick"
