"""Locate and drive ffmpeg. Uses the system ffmpeg if present, otherwise the static build
shipped with the ``imageio-ffmpeg`` package — so no manual install is needed."""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

log = logging.getLogger("purffle")

_IS_WINDOWS = os.name == "nt"
# Windows process-creation flags (fall back to the documented numeric values on old Pythons).
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
_BELOW_NORMAL_PRIORITY = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)

_QUICK_TIMEOUT = 30        # version / encoder listings
_PROBE_TIMEOUT = 60        # per probe subprocess
_MAX_WIN_CMDLINE = 30000   # CreateProcess hard limit is 32767 characters
_MTIME_SLACK = 3.0         # seconds; FAT/exFAT timestamps have 2 s resolution

# Output extensions whose content we can sanity-check with probe().
_CONTAINER_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4a", ".mp3", ".wav"}   # must have duration > 0
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}              # must have a stream


class FFmpegError(RuntimeError):
    pass


# --------------------------------------------------------------------------- binaries

def _clean_env_path(name: str) -> str | None:
    val = os.getenv(name)
    if not val:
        return None
    val = val.strip().strip('"').strip("'")
    return val or None


@functools.lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    env = _clean_env_path("FFMPEG_BINARY")
    if env:
        cand = Path(env).expanduser()
        if cand.is_file():
            return str(cand)
        resolved = shutil.which(env)
        if resolved:
            return resolved
        log.warning("FFMPEG_BINARY=%r does not point to an existing file; ignoring it.", env)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover - depends on the machine
        raise FFmpegError("ffmpeg not found. Install it (brew/apt/winget install ffmpeg) "
                          "or `pip install imageio-ffmpeg`.") from e


@functools.lru_cache(maxsize=1)
def ffprobe_bin() -> str | None:
    env = _clean_env_path("FFPROBE_BINARY")
    if env:
        cand = Path(env).expanduser()
        if cand.is_file():
            return str(cand)
        resolved = shutil.which(env)
        if resolved:
            return resolved
        log.warning("FFPROBE_BINARY=%r does not point to an existing file; ignoring it.", env)
    found = shutil.which("ffprobe")
    if found:
        return found
    try:
        ff = Path(ffmpeg_bin())
    except FFmpegError:
        return None
    sibling = ff.with_name("ffprobe" + ff.suffix)
    return str(sibling) if sibling.is_file() else None


# --------------------------------------------------------------------------- subprocess helpers

def _creationflags(low_priority: bool = False) -> int:
    """Windows only: hide the console window and optionally lower priority so the UI stays responsive."""
    if not _IS_WINDOWS:
        return 0
    flags = _CREATE_NO_WINDOW
    if low_priority and (os.getenv("PURFFLE_FFMPEG_PRIORITY", "").strip().lower() != "normal"):
        flags |= _BELOW_NORMAL_PRIORITY
    return flags


def _exec(cmd: list[str], *, timeout: float | None, cwd: str | Path | None = None,
          low_priority: bool = False) -> subprocess.CompletedProcess:
    kwargs: dict = {}
    if _IS_WINDOWS:
        kwargs["creationflags"] = _creationflags(low_priority)
    return subprocess.run(
        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding="utf-8", errors="replace", cwd=None if cwd is None else str(cwd),
        timeout=timeout, **kwargs)


def _fmt_cmd(cmd: list[str], limit: int | None = None) -> str:
    text = subprocess.list2cmdline(cmd) if _IS_WINDOWS else " ".join(shlex.quote(c) for c in cmd)
    if limit and len(text) > limit:
        text = text[:limit] + f"... [{len(text) - limit} more chars]"
    return text


def _tail(text: str | bytes | None, lines: int = 15, width: int = 500) -> str:
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    out = []
    for ln in text.strip().splitlines()[-lines:]:
        out.append(ln if len(ln) <= width else ln[:width] + "...")
    return "\n".join(out)


def _describe_rc(rc: int) -> str:
    if _IS_WINDOWS:
        if rc < 0 or rc > 0x7FFFFFFF:
            return f"{rc} / 0x{rc & 0xFFFFFFFF:08X}"
        return str(rc)
    if rc < 0:
        return f"{rc} (killed by signal {-rc})"
    if rc == 137:
        return "137 (killed, SIGKILL)"
    return str(rc)


def _normalise_args(args) -> list[str]:
    out: list[str] = []
    for a in args:
        if a is None:
            raise FFmpegError("None found in ffmpeg argument list (a path or option was not set).")
        s = os.fspath(a) if isinstance(a, os.PathLike) else str(a)
        if "\x00" in s:
            raise FFmpegError("NUL character in ffmpeg argument.")
        out.append(s)
    return out


# --------------------------------------------------------------------------- capability queries

@functools.lru_cache(maxsize=1)
def version() -> str:
    cmd = [ffmpeg_bin(), "-hide_banner", "-version"]
    try:
        out = _exec(cmd, timeout=_QUICK_TIMEOUT).stdout
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("Could not query ffmpeg version: %s", e)
        return "unknown"
    m = re.search(r"ffmpeg version (\S+)", out or "")
    return m.group(1) if m else "unknown"


@functools.lru_cache(maxsize=4)
def _listing(kind: str) -> str:
    cmd = [ffmpeg_bin(), "-hide_banner", f"-{kind}"]
    try:
        return _exec(cmd, timeout=_QUICK_TIMEOUT).stdout or ""
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("ffmpeg -%s listing failed: %s", kind, e)
        return ""


def has_filter(name: str) -> bool:
    return re.search(rf"^\s*\S+\s+{re.escape(name)}\s", _listing("filters"), re.M) is not None


def has_encoder(name: str) -> bool:
    return re.search(rf"^\s*\S+\s+{re.escape(name)}\s", _listing("encoders"), re.M) is not None


# --------------------------------------------------------------------------- failure diagnostics

def _failure_hints(stderr: str, args: list[str], cwd: str | Path | None) -> list[str]:
    hints: list[str] = []
    text = stderr or ""

    # Missing / empty inputs (only checked after a failure, so it can never block a valid run).
    if "lavfi" not in args:
        for i, a in enumerate(args[:-1]):
            if a != "-i":
                continue
            p = args[i + 1]
            if p == "-" or "://" in p or p.startswith("pipe:"):
                continue
            pp = Path(p)
            if not pp.is_absolute() and cwd is not None:
                pp = Path(cwd) / pp
            try:
                if not pp.exists():
                    hints.append(f"input file not found: {pp}")
                elif pp.is_file() and pp.stat().st_size == 0:
                    hints.append(f"input file is empty (0 bytes): {pp}")
            except OSError:
                pass

    m = re.search(r"No such filter: '?([\w\-]+)'?", text)
    if m:
        hints.append(f"this ffmpeg build ({version()}) has no filter '{m.group(1)}'.")
    m = re.search(r"Unknown encoder '([^']+)'", text)
    if m:
        hints.append(f"this ffmpeg build ({version()}) has no encoder '{m.group(1)}'; "
                     "use has_encoder() to pick a fallback.")
    if re.search(r"No space left|disk full", text, re.I):
        hints.append("the output drive is out of disk space.")
    if re.search(r"Cannot allocate memory|Out of memory|memory allocation|malloc", text, re.I):
        hints.append("ffmpeg ran out of memory; try a lower resolution/frame rate or shorter segments.")
    if re.search(r"Permission denied", text, re.I):
        hints.append("permission denied: the output may be open in a media player, or the folder is read-only.")
    if re.search(r"font", text, re.I) and re.search(r"(cannot|could not|unable|error)", text, re.I):
        hints.append("font problem: on Windows, drawtext/subtitles paths must be escaped, e.g. "
                     "C\\:/Windows/Fonts/arial.ttf (forward slashes, escaped drive colon).")
    return hints


# --------------------------------------------------------------------------- running ffmpeg

def validate_output(path: str | Path, *, min_bytes: int = 1, since: float | None = None,
                    label: str = "ffmpeg") -> Path:
    """Raise FFmpegError unless ``path`` is a non-trivial, readable media/image file."""
    p = Path(path)
    if not p.exists():
        raise FFmpegError(f"{label} reported success but the output file was not created: {p}")
    if not p.is_file():
        raise FFmpegError(f"{label}: output path is not a regular file: {p}")
    st = p.stat()
    if st.st_size < max(1, min_bytes):
        raise FFmpegError(f"{label}: output file is too small ({st.st_size} bytes): {p}")
    if since is not None and st.st_mtime < since - _MTIME_SLACK:
        raise FFmpegError(f"{label} reported success but the output file was not updated "
                          f"(stale file from an earlier run?): {p}")
    ext = p.suffix.lower()
    if ext in _CONTAINER_EXTS or ext in _IMAGE_EXTS:
        info = probe(p)
        if not (info["has_video"] or info["has_audio"]):
            raise FFmpegError(f"{label}: output file is unreadable or has no streams: {p}")
        if ext in _CONTAINER_EXTS and info["duration"] <= 0:
            raise FFmpegError(f"{label}: output file has zero duration (truncated/corrupt?): {p}")
    return p


def _discard(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        log.debug("Could not remove partial output %s: %s", path, e)


def run(args: list[str], *, label: str = "ffmpeg", cwd: str | Path | None = None, timeout: int = 1800,
        expect_output: str | Path | None = None, min_output_bytes: int = 1) -> None:
    """Run ffmpeg with ``args``; raise FFmpegError on any failure.

    ``expect_output`` / ``min_output_bytes`` are optional: when given, the file is validated after a
    successful run, and removed if the run fails (so a partial file is never mistaken for a good one).
    Relative ``expect_output`` paths are resolved against ``cwd``, like ffmpeg does.
    """
    str_args = _normalise_args(args)
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-y", *str_args]

    if cwd is not None and not Path(cwd).is_dir():
        raise FFmpegError(f"{label}: working directory does not exist: {cwd}")
    if _IS_WINDOWS and len(subprocess.list2cmdline(cmd)) > _MAX_WIN_CMDLINE:
        raise FFmpegError(f"{label}: command line is too long for Windows "
                          f"({len(subprocess.list2cmdline(cmd))} chars, limit ~32767). "
                          "Move the filter graph into a file and use -filter_complex_script.")

    out_path: Path | None = None
    if expect_output is not None:
        out_path = Path(expect_output)
        if not out_path.is_absolute() and cwd is not None:
            out_path = Path(cwd) / out_path

    log.debug("%s: %s", label, _fmt_cmd(cmd))
    if timeout is not None and timeout <= 0:
        timeout = None
    started = time.time()
    try:
        proc = _exec(cmd, timeout=timeout, cwd=cwd, low_priority=True)
    except subprocess.TimeoutExpired as e:
        _discard(out_path)
        partial = _tail(e.stderr)
        msg = f"{label} timed out after {timeout}s"
        if partial:
            msg += f"\n{partial}"
        msg += f"\nCommand: {_fmt_cmd(cmd, 1500)}"
        raise FFmpegError(msg) from e
    except OSError as e:
        _discard(out_path)
        raise FFmpegError(f"{label}: could not launch ffmpeg ({cmd[0]}): {e}") from e

    if proc.returncode != 0:
        _discard(out_path)
        stderr = proc.stderr or ""
        log.debug("%s stderr:\n%s", label, stderr)
        parts = [f"{label} failed (exit {_describe_rc(proc.returncode)}):", _tail(stderr) or "(no error output)"]
        hints = _failure_hints(stderr, str_args, cwd)
        if proc.returncode in (-9, 137):
            hints.append("process was killed, most likely by the OS for running out of memory.")
        if hints:
            parts.append("Hints:\n" + "\n".join(f" - {h}" for h in hints))
        parts.append(f"Command: {_fmt_cmd(cmd, 1500)}")
        raise FFmpegError("\n".join(parts))

    if proc.stderr and proc.stderr.strip():
        log.debug("%s warnings:\n%s", label, _tail(proc.stderr))

    if out_path is not None:
        try:
            validate_output(out_path, min_bytes=min_output_bytes, since=started, label=label)
        except FFmpegError:
            _discard(out_path)
            raise


# --------------------------------------------------------------------------- probing

_probe_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_probe_lock = threading.Lock()
_PROBE_CACHE_MAX = 128


def _to_float(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) and f > 0 else 0.0


def _empty_probe() -> dict:
    return {"duration": 0.0, "width": 0, "height": 0, "has_audio": False, "has_video": False}


def _probe_uncached(path: str) -> dict:
    fp = ffprobe_bin()
    if fp:
        try:
            out = _exec(
                [fp, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,duration",
                 "-of", "json", path], timeout=_PROBE_TIMEOUT)
            if out.returncode == 0:
                data = json.loads(out.stdout or "{}")
                streams = data.get("streams") or []
                v = next((s for s in streams if s.get("codec_type") == "video"), {})
                dur = (_to_float((data.get("format") or {}).get("duration"))
                       or _to_float(v.get("duration"))
                       or max((_to_float(s.get("duration")) for s in streams), default=0.0))
                return {
                    "duration": dur,
                    "width": int(v.get("width") or 0),
                    "height": int(v.get("height") or 0),
                    "has_audio": any(s.get("codec_type") == "audio" for s in streams),
                    "has_video": bool(v),
                }
            log.debug("ffprobe failed on %s: %s", path, _tail(out.stderr, 3))
        except (subprocess.SubprocessError, OSError, ValueError, TypeError, AttributeError) as e:
            log.debug("ffprobe error on %s (%s); using ffmpeg -i fallback.", path, e)

    # Fallback: parse `ffmpeg -i` banner (imageio-ffmpeg ships without ffprobe).
    try:
        proc = _exec([ffmpeg_bin(), "-hide_banner", "-i", path], timeout=_PROBE_TIMEOUT)
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("Could not probe %s: %s", path, e)
        return _empty_probe()
    err = proc.stderr or ""
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    vm = re.search(r"Stream #\S+.*?Video:.*?[\s,\[](\d{2,5})x(\d{2,5})(?=[\s,\[\]]|$)", err)
    return {
        "duration": dur,
        "width": int(vm.group(1)) if vm else 0,
        "height": int(vm.group(2)) if vm else 0,
        "has_audio": re.search(r"Stream #\S+.*?Audio:", err) is not None,
        "has_video": re.search(r"Stream #\S+.*?Video:", err) is not None,
    }


def probe(path: str | Path) -> dict:
    """Return {duration, width, height, has_audio, has_video} for a media file."""
    path = str(path)
    if path.startswith("-"):
        path = os.path.join(".", path)  # never let a filename be parsed as an option

    key = None
    if "://" not in path:
        try:
            st = os.stat(path)
        except OSError:
            log.warning("probe: file not found or unreadable: %s", path)
            return _empty_probe()
        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
        with _probe_lock:
            hit = _probe_cache.get(key)
            if hit is not None:
                _probe_cache.move_to_end(key)
                return dict(hit)

    info = _probe_uncached(path)

    if key is not None and (info["has_video"] or info["has_audio"]):
        with _probe_lock:
            _probe_cache[key] = dict(info)
            while len(_probe_cache) > _PROBE_CACHE_MAX:
                _probe_cache.popitem(last=False)
    return info


def duration(path: str | Path) -> float:
    return probe(path)["duration"]