# MolenGames chat bot

Watches your Minecraft chat for MolenGames games such as

> **[MolenGames]** Het Omgekeerd spel is gestart.
> \- Je hebt 25 seconden om het woord "kcirB" om te keren.

and types the answer (`Brick`) into chat for you.

## How it reads the chat

- **Screen (computer vision, default).** The bot takes screenshots of the
  chat area. Minecraft draws chat as pixel-perfect bitmap text: no smoothing,
  fixed colours, and a drop shadow one pixel down and to the right. So instead
  of general OCR (Tesseract read `kcirB` as `keirB` or `koirB`), the bot:
  1. keeps only pixels that exactly match a chat colour and have a correct
     drop shadow, which removes the world behind the chat;
  2. works out the GUI scale and cuts the image into lines and characters;
  3. looks up each character's exact pixel pattern in a font table that you
     teach it once with `calibrate`.

  A character it hasn't learned shows up as `�`. The bot never sends an answer
  that contains one.
- **Game log (`"source": "log"`).** This reads `.minecraft/logs/latest.log`,
  which gets the text exactly and needs no calibration. Use it if your
  launcher writes that file (vanilla, Fabric and Forge do; other clients may not).

## Setup (Windows, Python 3.10+)

Run these commands in a terminal inside this folder.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create your settings file:

```bash
copy config.example.json config.json
```

In Minecraft, open **Options → Chat Settings** and set **Chat Text Size 100%**
and **Chat Text Opacity 100%**. Then:

1. Mark the chat area. Have chat visible in-game, then run the command below
   and drag a box over the chat lines:

```bash
python -m molenbot region
```

2. Teach it the font. Run the command below and, when asked, send each test
   line it shows you in chat:

```bash
python -m molenbot calibrate
```

   The `[MolenGames]` prefix is bold, which looks different from normal text.
   The next time a game appears on screen, teach it that line too:

```bash
python -m molenbot calibrate --text "[MolenGames] Het Omgekeerd spel is gestart."
```

3. Check what it reads:

```bash
python -m molenbot read
```

4. Test it without typing anything. It prints the answers instead:

```bash
python -m molenbot run --dry-run
```

5. Run it for real. Keep the Minecraft window focused; **F8** pauses and
   resumes, and **Ctrl+C** quits.

```bash
python -m molenbot run
```

## Games

| game         | status | how it answers |
|--------------|--------|----------------|
| `reverse` (Omgekeerd) | wording confirmed from a screenshot | reverses the word |
| `type`       | wording guessed | types the word back |
| `unscramble` | wording guessed | looks it up in `words.txt` (one word per line) plus words the bot has seen in other games |
| `math`       | wording guessed | works out the sum safely, without running code |

When you see the real wording of another game, run `python -m molenbot read`
while it's on screen. Then set a regex with a `(?P<q>...)` group for the
puzzle in `config.json` under `"patterns"`, for example:

```json
"patterns": { "type": "woord\\s*\"(?P<q>[^\"]+)\"\\s*te\\s+typen" }
```

## Settings worth knowing

- `answer_delay`: random wait before typing, in seconds (default 1.2–2.8).
  Instant answers look like a bot.
- `chat_key`: the key that opens chat (default `t`).
- `extra_colors`: add `[r, g, b]` values if the server uses custom hex
  colours for game messages.
- Changing the GUI scale doesn't need a new calibration. Changing the chat
  size or position needs a new `region`.

Many servers forbid automatic answers in chat games. Check the MolenGames
rules before running this unattended.

Run the tests:

```bash
python -m pytest tests
```
