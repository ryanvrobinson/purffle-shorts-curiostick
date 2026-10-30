"""Which words deserve a colour, a small scale bump or extra brightness on top of the plain
word-by-word highlight.

The captions already know *when* a word is spoken (TTS or alignment gives per-word timestamps) and
*that* the spoken word gets the signal orange of the style. That alone makes every word look equally
important, which reads as noise over footage. This module adds the second half: a deterministic,
adjustable reading of the *text* that marks

  * key technical terms  — acronyms (CPU, VRAM), version and model numbers (GPT-5, RTX 4090),
                           units and percentages (98%, 12 nm), hyphenated compounds (self-driving),
  * important narrative phrases — the opening word of a caption, the last word of a caption (where
                           the payoff lands), and the word right after an emphasis trigger
                           ("but", "never", "actually", "the truth is").

It is pure text in, bool out, so it is cheap, unit-testable and never depends on the network. The
rules are deliberately conservative: emphasis on a third of the words is the same as emphasis on
none, so a caption is capped at two emphasised words (``CAPTION_EMPHASIS_MAX``).
"""

from __future__ import annotations

import re
import unicodedata

# Leading/trailing punctuation we strip before looking at the word itself (mirrors display_word).
TRIM = ",.;:\"'“”()[]{}—–-…!?¡¿«»"

# "but", "never", "actually" ... the word *after* one of these is the turn of the sentence.
TRIGGERS = {
    "but", "however", "actually", "never", "not", "no", "only", "just", "instead", "because",
    "secret", "truth", "warning", "problem", "catch", "huge", "massive", "insane", "tiny",
    "wrong", "myth", "fact", "really", "most", "fastest", "biggest", "smallest", "deadliest",
    "strongest", "rarest", "first", "last", "every", "always", "zero", "nothing", "everything",
}

# Units and suffixes that make a number worth looking at (98%, 12 nm, 3x, 1 kg, 6 s).
UNIT = re.compile(r"\d\s?(?:%|x|×|k|m|b|nm|mm|cm|km|kg|mg|ms|s|h|hz|ghz|mhz|khz|gb|tb|mb|kb|fps|w|v|"
                  r"a|c|f)\b", re.IGNORECASE)
NUMBER = re.compile(r"\d")
ACRONYM = re.compile(r"^(?:[A-Z]{2,6}|\d+[A-Z]{1,4}|[A-Z]{1,4}\d{1,4}|[A-Z]{2,}\d+)$")
VERSION = re.compile(r"^v?\d+(?:\.\d+)+$")
MODEL = re.compile(r"^[a-z]{2,8}-?\d[\w.]*$", re.IGNORECASE)
COMPOUND = re.compile(r"^\w+(?:-\w+)+$")

MODES = {"auto", "off", "all"}


def normalize_mode(mode: str) -> str:
    mode = (mode or "auto").strip().lower()
    return mode if mode in MODES else "auto"


def strip_token(text: str) -> str:
    """The bare word: no surrounding punctuation, no zero-width joiners from emoji sequences."""
    t = unicodedata.normalize("NFC", text or "").strip().strip(TRIM)
    return t.strip("\u200d\ufe0f").strip(TRIM)



def is_numberish(token: str) -> bool:
    """True for figures a viewer reads as a datum (98%, 12 nm, 4x, 1.5 billion)."""
    if not token or not NUMBER.search(token):
        return False
    return bool(UNIT.search(token) or re.fullmatch(r"[+-]?[\d,.]+\w*", token))


def is_acronym(token: str) -> bool:
    """True for CPU, VRAM, GPT-5, RTX 4090, GPT4 — short and shouted."""
    if not token:
        return False
    bare = token.replace(" ", "")
    if ACRONYM.match(bare) or VERSION.match(bare):
        return True
    letters = [c for c in bare if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def is_technical(token: str) -> bool:
    """True for the vocabulary an AI-tech explainer calls a term rather than a word."""
    if not token:
        return False
    bare = token.replace(" ", "")
    if len(bare) < 3:
        return False
    return bool(is_acronym(bare) or is_numberish(bare) or COMPOUND.match(bare)
                or MODEL.match(bare) or VERSION.match(bare))


def is_emphasis(token: str, index: int | None = None, total: int | None = None,
                previous: str | None = None, mode: str = "auto") -> bool:
    """Should this single word carry emphasis?

    ``index``/``total`` mark the first and last word of a caption, ``previous`` is the word before it
    (checked against :data:`TRIGGERS`). ``mode="all"`` emphasises everything (a loud karaoke look),
    ``"off"`` nothing, ``"auto"`` the rules above.
    """
    mode = normalize_mode(mode)
    if mode == "off":
        return False
    token = strip_token(token)
    if not token:
        return False
    if mode == "all":
        return True
    if total is not None and total <= 2:
        return True                      # a two-word caption is one phrase; both words read as key
    if is_technical(token):
        return True
    if index == 0 or (total is not None and total > 0 and index == total - 1):
        return True                      # a caption opens and ends on its important words
    if token.lower() in TRIGGERS:
        return True
    if previous and strip_token(previous).lower() in TRIGGERS:
        return True
    return False


def emphasis_flags(words: list[str], mode: str = "auto", limit: int = 2) -> list[bool]:
    """Emphasis for a whole caption, capped so it never looks like a highlighter test.

    The cap keeps the strongest signals first: technical terms beat the first/last word rule and the
    last word beats the first (the payoff lands at the end of the line). Deterministic per caption.
    """
    mode = normalize_mode(mode)
    if mode == "off" or not words:
        return [False] * len(words)
    if mode == "all":
        return [True] * len(words)

    total = len(words)
    flags = [False] * total
    priority = [0] * total
    for i, raw in enumerate(words):
        token = strip_token(raw)
        prev = words[i - 1] if i else None
        if is_technical(token):
            flags[i], priority[i] = True, 3
        elif i == total - 1:
            flags[i], priority[i] = True, 2
        elif i == 0:
            flags[i], priority[i] = True, 1
        elif token.lower() in TRIGGERS:
            flags[i], priority[i] = True, 3      # the word that turns the sentence ("but", "never")
        elif prev and strip_token(prev).lower() in TRIGGERS:
            flags[i], priority[i] = True, 3      # the word after it ("but *nobody*")

    if limit and sum(flags) > limit:
        keep = set(sorted((i for i in range(total) if flags[i]), key=lambda i: (-priority[i], -i))[:limit])
        flags = [i in keep for i in range(total)]
    return flags


def emphasised(words: list[str], mode: str = "auto", limit: int = 2) -> list[int]:
    """The indices emphasised in a caption (the shape the renderers consume)."""
    return [i for i, on in enumerate(emphasis_flags(words, mode, limit)) if on]
