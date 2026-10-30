"""Render JPEG frames and a short mp4 of the captions, so the look is checked before a full render.

    python scripts/preview_captions.py "The octopus has 3 hearts and blue blood"
    python scripts/preview_captions.py "..." --style kinetic --exit dust --position lower

It writes a strip of the caption states plus, when ffmpeg is available, a couple of seconds of video
with the caption playing over a black and a white plate, so contrast can be judged. Nothing here is
imported by the package or the tests; it is a look-checking tool.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from purffle_shorts import emphasis as em_rules  # noqa: E402
from purffle_shorts.config import Settings  # noqa: E402
from purffle_shorts.overlays import CAPTION_STYLES, PillowCaptions, resolve_font  # noqa: E402
from purffle_shorts.timing import Chunk, Word  # noqa: E402


def build_chunks(text: str, per_word: float = 0.32, max_words: int = 3) -> list[Chunk]:
    tokens = text.split()
    chunks, t = [], 0.0
    for i in range(0, len(tokens), max_words):
        words = []
        for tok in tokens[i:i + max_words]:
            words.append(Word(tok, t, t + per_word))
            t += per_word
        chunks.append(Chunk(words=words, start=words[0].start, end=words[-1].end))
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text", nargs="?", default="The octopus has 3 hearts, blue blood and a 98% hidden brain.")
    ap.add_argument("--style", default="kinetic", choices=sorted(CAPTION_STYLES))
    ap.add_argument("--position", default="center", choices=["upper", "center", "lower"])
    ap.add_argument("--exit", default="dust", choices=["auto", "none", "dust", "dissolve"])
    ap.add_argument("--emphasis", default="auto", choices=["auto", "off", "all"])
    ap.add_argument("--contrast", default="auto", choices=["auto", "never", "safe"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="caption_preview")
    args = ap.parse_args()

    settings = Settings(caption_style=args.style, caption_position=args.position,
                        caption_exit=args.exit, caption_emphasis=args.emphasis,
                        caption_contrast=args.contrast, caption_scale=args.scale)
    style = CAPTION_STYLES[args.style]
    font_path, family = resolve_font(settings, style.font)
    print(f"font: {font_path} ({family})")
    caps = PillowCaptions(settings, style, font_path)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # 1. A strip of every word state of the first caption, over black and over white.
    raw = [w.text for w in build_chunks(args.text)[0].words] or args.text.split()[:3]
    words = [w.upper() if settings.caption_uppercase else w for w in raw]
    em = set(em_rules.emphasised(words, settings.caption_emphasis, settings.caption_emphasis_max))
    size = caps.fit_size(words, em)
    states = []
    for k in [None, *range(len(words))]:
        states.append(caps.render(words, k, size, speaker="A", emphasis=em))
    print(f"emphasis words: {sorted(em)}  size: {size}  states: {len(states)}")

    # 2. Compose the strip on the two background extremes to judge contrast and the scrim.
    for name, bg in (("black", (0, 0, 0, 255)), ("white", (255, 255, 255, 255)),
                     ("grey", (128, 128, 128, 255))):
        strip = Image.new("RGB", (caps.W, caps.band_h * len(states)), bg[:3])
        for i, band in enumerate(states):
            strip.paste(band, (0, i * caps.band_h), band)
        strip.resize((caps.W // 2, caps.band_h * len(states) // 2)).save(out / f"states_{name}.jpg",
                                                                        quality=90)
    print(f"wrote {len(states)} states x 3 backgrounds to {out}")

    # 3. A short mp4 with the caption actually playing, over a white and a dark plate.
    chunks = build_chunks(args.text)
    total = chunks[-1].end + 0.4
    playlist = caps.build(chunks, total, out / "frames")
    print(f"playlist: {playlist}")
    for name, color in (("white", "white"), ("dark", "0x101014")):
        dest = out / f"preview_{name}.mp4"
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={caps.W}x{caps.H}:r={settings.fps}"
                                              f":d={total:.2f}",
               "-f", "concat", "-safe", "0", "-i", str(playlist.resolve()),
               "-filter_complex", f"[0:v][1:v]overlay=0:{caps.band_y}:format=auto[v]",
               "-map", "[v]", "-t", f"{total:.2f}", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
               str(dest)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"wrote {dest}")
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"ffmpeg preview skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
