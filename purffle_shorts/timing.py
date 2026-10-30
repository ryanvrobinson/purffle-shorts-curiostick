"""Word timing: estimate it, transfer it from an aligner onto the script's own words, split the
narration into scenes and caption chunks.

Time convention (shared by every module): all times, durations and boundaries are in **seconds**
(floats), measured from the start of the narration track. A ``Word``'s ``start``/``end`` and a
``Chunk``'s ``start``/``end`` are absolute positions on that single timeline, which is the only
authoritative timeline for the render."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str = "A"


NO_SPACE_LANGS = {"zh", "ja"}


def tokenize(text: str, language: str = "en") -> list[str]:
    """Split narration into caption tokens. Chinese/Japanese have no spaces, so they are cut into
    short character groups at punctuation (timing for those is approximate)."""
    if language.split("-")[0] not in NO_SPACE_LANGS:
        return [t for t in text.split() if t.strip()]
    tokens: list[str] = []
    for part in re.findall(r"[^，。！？、；：,.!?;:\s]+[，。！？、；：,.!?;:]?", text):
        while len(part) > 5:
            tokens.append(part[:4])
            part = part[4:]
        if part:
            tokens.append(part)
    return tokens


def norm(token: str) -> str:
    return re.sub(r"[^\w]", "", token.lower(), flags=re.UNICODE)


def estimate_timings(tokens: list[str], duration: float, lead: float = 0.05, tail: float = 0.15) -> list[Word]:
    """Spread tokens over ``duration`` by length, with extra time after punctuation (natural pauses)."""
    if not tokens:
        return []
    weights = []
    for t in tokens:
        w = len(norm(t)) + 2.0
        if t[-1:] in ".!?":
            w += 4.0
        elif t[-1:] in ",;:":
            w += 2.0
        weights.append(w)
    span = max(0.1, duration - lead - tail)
    scale = span / sum(weights)
    out, t0 = [], lead
    for tok, w in zip(tokens, weights):
        d = w * scale
        pause = 0.0
        if tok[-1:] in ".!?":
            pause = min(4.0 * scale, d * 0.4)
        elif tok[-1:] in ",;:":
            pause = min(2.0 * scale, d * 0.3)
        out.append(Word(tok, round(t0, 3), round(t0 + d - pause, 3)))
        t0 += d
    return out


def transfer_timings(tokens: list[str], timed: list[Word], duration: float) -> list[Word]:
    """Give every script token a time using an aligner's (possibly different) word list.

    Matching is done on normalized text; tokens the aligner missed are interpolated between
    their matched neighbours so captions keep the script's exact wording and punctuation.
    """
    if not tokens:
        return []
    if not timed:
        return estimate_timings(tokens, duration)
    a = [norm(t) for t in tokens]
    b = [norm(w.text) for w in timed]
    starts: list[float | None] = [None] * len(tokens)
    ends: list[float | None] = [None] * len(tokens)
    for block in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            w = timed[block.b + k]
            starts[block.a + k], ends[block.a + k] = w.start, w.end

    # A token like "1,000" may be spoken as several aligner words; anchor unmatched runs between neighbours.
    n = len(tokens)
    i = 0
    while i < n:
        if starts[i] is not None:
            i += 1
            continue
        j = i
        while j < n and starts[j] is None:
            j += 1
        left = ends[i - 1] if i > 0 and ends[i - 1] is not None else 0.0
        right = starts[j] if j < n and starts[j] is not None else max(left, duration - 0.1)
        right = max(right, left + 0.05 * (j - i))
        sub = estimate_timings(tokens[i:j], right - left, lead=0.0, tail=0.0)
        for k, w in enumerate(sub):
            starts[i + k], ends[i + k] = left + w.start, left + w.end
        i = j

    out = []
    prev_start = 0.0
    for tok, s, e in zip(tokens, starts, ends):
        s = max(float(s), prev_start)  # keep starts monotonic even if the aligner jittered
        e = max(float(e), s + 0.04)
        out.append(Word(tok, round(s, 3), round(e, 3)))
        prev_start = s
    return out


@dataclass
class SceneTiming:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def scene_timeline(scene_texts: list[str], words: list[Word], total: float, language: str = "en",
                   starts: list[float] | None = None, min_len: float = 1.2) -> list[SceneTiming]:
    """Scene boundaries follow the voice: scene k starts when its first word is spoken. ``starts`` can
    give them directly (multi-voice audio is made line by line, so the boundaries are exact)."""
    if starts is None:
        counts = [len(tokenize(t, language)) for t in scene_texts]
        starts, idx = [], 0
        for c in counts:
            if idx < len(words):
                starts.append(words[idx].start)
            else:
                starts.append(starts[-1] if starts else 0.0)
            idx += c
    starts = list(starts)
    starts[0] = 0.0
    out = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else total
        out.append(SceneTiming(k, s, e))
    # Grow each shot until it lasts at least min_len (too-short scenes can't be read), then start the next;
    # a short final shot joins the one before it.
    merged: list[SceneTiming] = []
    for st in out:
        if merged and merged[-1].duration < min_len:
            merged[-1].end = st.end
        else:
            merged.append(st)
    if len(merged) > 1 and merged[-1].duration < min_len:
        last = merged.pop()
        merged[-1].end = last.end
    return merged


@dataclass
class Chunk:
    words: list[Word]
    start: float
    end: float

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def speaker(self) -> str:
        return self.words[0].speaker if self.words else "A"


# Function words a caption should never end on (they belong with what follows).
DANGLING_ENDINGS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "into", "onto", "over", "under",
    "is", "was", "were", "are", "am", "be", "been", "being", "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can", "shall", "than", "then",
    "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their",
})
_SENTENCE_END = ".!?"
_MINOR_BREAK = ",;:"
_TRIM = ".,!?;:\"'“”‘’()[]"


def _dangling(text: str) -> bool:
    """True when ``text`` is a function word that must not end a caption on its own."""
    return text.lower().strip().strip(_TRIM).strip() in DANGLING_ENDINGS


def _merge_short_chunks(chunks: list[Chunk], max_words: int) -> list[Chunk]:
    """Fold a one-word caption that would flash for a couple of frames into the caption before it."""
    out: list[Chunk] = []
    for c in chunks:
        if out:
            prev = out[-1]
            if (len(c.words) == 1 and (c.end - c.start) < 0.2 and c.speaker == prev.speaker
                    and len(prev.words) + 1 <= max_words + 1):
                prev.words.extend(c.words)
                prev.end = c.end
                continue
        out.append(c)
    return out


def caption_chunks(words: list[Word], max_words: int = 5, max_chars: int = 40,
                   total: float | None = None) -> list[Chunk]:
    """Group words into short on-screen captions (all times in seconds).

    A caption is flushed *before* the next word when the phrase is complete: the previous word ended
    a sentence, a pause in the voice is long, the speaker changes, or the caption reached its length
    limit. A length limit is relaxed by one word when the caption would otherwise end on a dangling
    function word (so phrases never break like "the study of"), and a sentence-closing word is always
    allowed to finish its phrase. A one-word caption shorter than a couple of frames is merged into
    its neighbour so it never flickers. Every returned chunk keeps the exact script wording.
    """
    max_words = max(1, max_words)
    chunks: list[Chunk] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            chunks.append(Chunk(list(cur), cur[0].start, cur[-1].end))
            cur.clear()

    for w in words:
        if cur:
            prev = cur[-1]
            gap = w.start - prev.end
            chars_if_added = sum(len(x.text) + 1 for x in cur) + len(w.text)
            word_closes = w.text[-1:] in _SENTENCE_END
            # Grow past the soft limit only to rescue a dangling word (possibly a short run of them)
            # or to finish a sentence; never grow past one word over the limit.
            rescue = len(cur) <= max_words + 1 and (_dangling(prev.text) or word_closes)
            at_limit = len(cur) >= max_words and not rescue
            too_long = chars_if_added > max_chars and len(cur) >= 2 and not rescue
            if (w.speaker != prev.speaker or gap > 0.45 or prev.text[-1:] in _SENTENCE_END
                    or at_limit or too_long
                    or (prev.text[-1:] in _MINOR_BREAK and len(cur) >= 2)):
                flush()
        cur.append(w)

    flush()
    chunks = _merge_short_chunks(chunks, max_words)
    # Hold each caption on screen until the next one starts (no flicker between words); keep a small
    # tail when the next caption is far away, and never let a caption end before it begins.
    for k in range(len(chunks) - 1):
        nxt = chunks[k + 1].start
        if nxt - chunks[k].end < 0.6:
            chunks[k].end = max(nxt, chunks[k].start + 0.05)
        else:
            chunks[k].end = chunks[k].end + 0.25
    if chunks:
        end = chunks[-1].end + 0.4
        chunks[-1].end = min(end, total) if total else end
    return chunks


def srt(chunks: list[Chunk]) -> str:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, c in enumerate(chunks, 1):
        lines += [str(i), f"{ts(c.start)} --> {ts(c.end)}", c.text, ""]
    return "\n".join(lines)
