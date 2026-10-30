"""Opt-in smoke test for Agnes AI image generation.

The default is a dry run: it shows the request shape and whether a key is loaded without
spending credits. Pass ``--live`` to make exactly one request and validate the saved image.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from purffle_shorts.media import agnes_image, validate_generated_image  # noqa: E402
from purffle_shorts.utils import redact  # noqa: E402

SIZES = {"portrait": "1024x1536", "landscape": "1536x1024", "square": "1024x1024"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="make one real, possibly billable request")
    parser.add_argument("--prompt", default="A single octopus in dark water, editorial lighting",
                        help="image prompt (no text or letters)")
    parser.add_argument("--model", default="", help="override AGNES_IMAGE_MODEL")
    parser.add_argument("--orientation", choices=sorted(SIZES), default="portrait")
    parser.add_argument("--out", default="", help="output PNG path (default: output_videos/agnes-smoke-*.png)")
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    key = os.getenv("AGNES_API_KEY", "").strip()
    model = args.model.strip() or os.getenv("AGNES_IMAGE_MODEL", "agnes-image-2.5-flash")
    dest = Path(args.out) if args.out else ROOT / "output_videos" / f"agnes-smoke-{int(time.time())}.png"

    print("endpoint: https://apihub.agnes-ai.com/v1/images/generations")
    print(f"model: {model}")
    print(f"size: {SIZES[args.orientation]} (n=1)")
    print(f"prompt: {args.prompt}")
    print(f"key loaded: {'yes' if key else 'no'}")
    print(f"output: {dest}")
    if not args.live:
        print("Dry run only. Re-run with --live to spend one Agnes image request.")
        return 0
    if not key:
        print("FAILED: AGNES_API_KEY is not set")
        return 2

    print("Calling Agnes once. This request may be billable.")
    try:
        agnes_image(args.prompt, dest, key, model, args.orientation)
        width, height = validate_generated_image(dest)
    except Exception as exc:
        print(f"FAILED: {redact(exc)}")
        return 1

    print(f"OK: {width}x{height}, {dest.stat().st_size} bytes, {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
