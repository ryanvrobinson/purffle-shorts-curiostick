"""Text-to-speech engines. Every engine returns audio plus per-word timings (native when the engine
provides them, otherwise from an aligner or an estimate) so captions land exactly on the voice.

  edge        Microsoft neural voices — free, no key, 320+ voices in 75 languages, native word timings (default)
  openai      OpenAI gpt-4o-mini-tts / tts-1-hd (OPENAI_API_KEY)
  elevenlabs  ElevenLabs premium voices, native timings (ELEVENLABS_API_KEY)
  kokoro      Kokoro-82M open-weight model, runs locally (pip install kokoro soundfile)
  coqui       Coqui TTS models, runs locally (pip install coqui-tts)
  system      The OS voice (macOS `say`, Linux espeak-ng, Windows SAPI) — offline
  silent      No voice; for layout previews and tests
"""

from __future__ import annotations

import asyncio
import base64
import logging
import queue
import re
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from . import ffmpeg
from .config import Settings
from .script import MULTI_SPEAKER
from .timing import NO_SPACE_LANGS, Word, estimate_timings, tokenize, transfer_timings
from .utils import PermanentError, http, raise_for_status, with_retries

log = logging.getLogger("purffle")

# Hand-picked, natural-sounding default voices per language (edge-tts). Override with TTS_VOICE.
EDGE_VOICES = {
    "en": "en-US-AndrewMultilingualNeural", "es": "es-MX-JorgeNeural", "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural", "it": "it-IT-DiegoNeural", "pt": "pt-BR-AntonioNeural",
    "hi": "hi-IN-MadhurNeural", "ta": "ta-IN-ValluvarNeural", "te": "te-IN-MohanNeural",
    "bn": "bn-IN-BashkarNeural", "mr": "mr-IN-ManoharNeural", "ur": "ur-PK-AsadNeural",
    "ar": "ar-SA-HamedNeural", "ru": "ru-RU-DmitryNeural", "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural", "zh": "zh-CN-YunxiNeural", "id": "id-ID-ArdiNeural",
    "tr": "tr-TR-AhmetNeural", "nl": "nl-NL-MaartenNeural", "pl": "pl-PL-MarekNeural",
    "vi": "vi-VN-NamMinhNeural", "th": "th-TH-NiwatNeural", "fil": "fil-PH-AngeloNeural",
    "uk": "uk-UA-OstapNeural", "sv": "sv-SE-MattiasNeural",
}
# Second speaker for dialogue and chat formats (a contrasting voice). Override with TTS_VOICE_B.
EDGE_VOICES_B = {
    "en": "en-US-AvaMultilingualNeural", "es": "es-MX-DaliaNeural", "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural", "it": "it-IT-ElsaNeural", "pt": "pt-BR-FranciscaNeural",
    "hi": "hi-IN-SwaraNeural", "ta": "ta-IN-PallaviNeural", "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural", "mr": "mr-IN-AarohiNeural", "ur": "ur-PK-UzmaNeural",
    "ar": "ar-SA-ZariyahNeural", "ru": "ru-RU-SvetlanaNeural", "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural", "zh": "zh-CN-XiaoxiaoNeural", "id": "id-ID-GadisNeural",
    "tr": "tr-TR-EmelNeural", "nl": "nl-NL-ColetteNeural", "pl": "pl-PL-ZofiaNeural",
    "vi": "vi-VN-HoaiMyNeural", "th": "th-TH-PremwadeeNeural", "fil": "fil-PH-BlessicaNeural",
    "uk": "uk-UA-PolinaNeural", "sv": "sv-SE-SofieNeural",
}
SECOND_VOICE = {"openai": "nova", "elevenlabs": "21m00Tcm4TlvDq8ikWAM", "kokoro": "af_heart"}
LINE_GAP = 0.18  # silence between two speakers' lines

KOKORO_LANG = {"en": ("a", "am_michael"), "es": ("e", "em_alex"), "fr": ("f", "ff_siwis"),
               "hi": ("h", "hm_omega"), "it": ("i", "im_nicola"), "ja": ("j", "jm_kumo"),
               "pt": ("p", "pm_alex"), "zh": ("z", "zm_yunxi")}


@dataclass
class Speech:
    audio: Path
    words: list[Word]
    duration: float
    engine: str
    voice: str
    timing: str  # native | aligned | estimated
    line_starts: list[float] | None = None  # multi-voice: when each line starts


CATEGORY_VOICES = {
    "educational": {
        "en": "en-US-AndrewMultilingualNeural",
        "es": "es-MX-JorgeNeural",
        "fr": "fr-FR-HenriNeural",
        "de": "de-DE-ConradNeural",
        "default": "en-US-AndrewMultilingualNeural",
    },
    "storytelling": {
        "en": "en-US-AriaNeural",
        "es": "es-MX-DaliaNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KatjaNeural",
        "default": "en-US-AriaNeural",
    },
    "documentary": {
        "en": "en-US-SteveMorrisNeural",
        "es": "es-MX-JorgeNeural",
        "fr": "fr-FR-HenriNeural",
        "de": "de-DE-ConradNeural",
        "default": "en-US-SteveMorrisNeural",
    },
    "technology": {
        "en": "en-US-GuyNeural",
        "es": "es-MX-JorgeNeural",
        "fr": "fr-FR-DeniseNeural",
        "default": "en-US-GuyNeural",
    },
    "energetic": {
        "en": "en-US-AvaMultilingualNeural",
        "es": "es-MX-DaliaNeural",
        "fr": "fr-FR-DeniseNeural",
        "default": "en-US-AvaMultilingualNeural",
    },
    "calm": {
        "en": "en-US-EthanNeural",
        "es": "es-MX-JorgeNeural",
        "fr": "fr-FR-HenriNeural",
        "default": "en-US-EthanNeural",
    },
}


def _classify_category(settings: Settings, style: str | None = None, topic: str | None = None) -> str:
    """Determine the content category for voice selection."""
    s = settings.style if style is None else style
    t = (topic or "").lower()  # Settings has no 'topic'; callers may pass the script's subject
    if s in MULTI_SPEAKER:
        return "storytelling"
    if any(kw in t for kw in ["science", "technology", "AI", "tech", "computer", "software"]):
        return "technology"
    if any(kw in t for kw in ["history", "civilization", "ancient", "archaeology", "biography"]):
        return "documentary"
    if any(kw in t for kw in ["fact", "explainer", "learn", "did you know", "surprising"]):
        return "educational"
    if any(kw in t for kw in ["story", "tale", "legend", "myth"]):
        return "storytelling"
    if any(kw in t for kw in ["music", "entertainment", "fun", "exciting"]):
        return "energetic"
    if any(kw in t for kw in ["calm", "relax", "peaceful", "meditative"]):
        return "calm"
    return "educational"  # default fallback


def _pick_voice_by_category(settings: Settings, category: str, default: str) -> str:
    """Select a voice based on content category and language."""
    lang = settings.language.split("-")[0]
    category_voices = CATEGORY_VOICES.get(category, CATEGORY_VOICES["educational"])
    voice = category_voices.get(lang, category_voices.get("default"))
    return voice if voice else default


def _pick_voice(settings: Settings, default: str) -> str:
    """Select a voice using category-based intelligence, with --voice override respected.

    - If --voice is explicitly set, use it (honor user override).
    - Otherwise, classify the content category and select an appropriate voice.
    - Selection is deterministic for identical inputs unless variation is requested.
    """
    # If voice is explicitly set via --voice, use it (honor user override)
    if settings.tts_voice:
        voices = [v.strip() for v in settings.tts_voice.split(",") if v.strip()]
        if voices:
            return voices[0]  # single voice or first in list

    # Otherwise, use intelligent category-based selection
    category = _classify_category(settings)
    return _pick_voice_by_category(settings, category, default)


def _rate_multiplier(rate: str) -> float:
    try:
        return 1.0 + float(rate.strip().rstrip("%")) / 100.0
    except ValueError:
        return 1.0


def expand_pronunciations(text: str, pronunciations: dict[str, str],
                          language: str = "en") -> tuple[str, list[int]]:
    """Expand configured terms for the voice only, leaving caption text untouched.

    Returns the spoken text and, for each original whitespace token, how many spoken tokens it became.
    Scripts without spaces are left alone because a term cannot be isolated reliably.
    """
    tokens = text.split()
    if not pronunciations or language.split("-")[0] in NO_SPACE_LANGS:
        return text, [1] * len(tokens)
    spoken_tokens: list[str] = []
    groups: list[int] = []
    for token in tokens:
        expanded = token
        for term, replacement in pronunciations.items():
            expanded = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", replacement, expanded, flags=re.IGNORECASE)
        parts = expanded.split()
        groups.append(len(parts))
        spoken_tokens.extend(parts)
    return " ".join(spoken_tokens), groups


def _group_word_timings(tokens: list[str], groups: list[int],
                        spoken_words: list[Word]) -> list[Word] | None:
    """Fold expanded spoken words back onto the original caption tokens."""
    if len(tokens) != len(groups) or any(n <= 0 for n in groups) or sum(groups) != len(spoken_words):
        return None
    out: list[Word] = []
    k = 0
    for token, count in zip(tokens, groups):
        part = spoken_words[k:k + count]
        k += count
        out.append(Word(token, part[0].start, part[-1].end))
    return out


def _pronounced_timings(tokens: list[str], spoken_tokens: list[str], groups: list[int],
                        native: list[Word], duration: float) -> list[Word]:
    mapped = any(n != 1 for n in groups)
    if not mapped:
        return transfer_timings(tokens, native, duration)
    spoken_words = transfer_timings(spoken_tokens, native, duration)
    grouped = _group_word_timings(tokens, groups, spoken_words)
    return grouped if grouped is not None else transfer_timings(tokens, native, duration)


# ------------------------------------------------------------------------------------------ engines
# edge-tts' own ``stream_sync`` waits on an unbounded queue with no timeout: if its worker thread dies
# (a dropped connection, a socket the server half-closed) the sentinel is never queued and the call
# blocks forever. Drive the public async ``stream()`` ourselves with a deadline so a stalled synthesis
# raises instead of hanging the whole pipeline.
EDGE_STREAM_TIMEOUT = 180.0  # seconds for one synthesis attempt (per with_retries attempt)


def _edge(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    try:
        import edge_tts
    except ImportError as e:
        raise PermanentError("edge-tts is not installed:  pip install edge-tts") from e
    voice = _pick_voice(settings, EDGE_VOICES.get(settings.language, EDGE_VOICES["en"]))
    path = out / "voice.mp3"

    def _do():
        words: list[Word] = []
        comm = edge_tts.Communicate(text, voice, rate=settings.tts_rate, boundary="WordBoundary",
                                    receive_timeout=60)
        items: queue.Queue = queue.Queue()
        done = object()

        def produce() -> None:
            async def consume() -> None:
                async for chunk in comm.stream():
                    items.put(chunk)

            try:
                asyncio.run(consume())
            except BaseException as e:  # noqa: BLE001 — hand the failure to the consumer below
                items.put(e)
            finally:
                items.put(done)

        threading.Thread(target=produce, name="edge-tts", daemon=True).start()
        deadline = time.monotonic() + EDGE_STREAM_TIMEOUT
        with open(path, "wb") as f:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RuntimeError(f"edge-tts produced no complete audio within "
                                       f"{EDGE_STREAM_TIMEOUT:.0f}s (stalled connection)")
                try:
                    item = items.get(timeout=min(left, 2.0))
                except queue.Empty:
                    continue
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                if item["type"] == "audio":
                    f.write(item["data"])
                elif item["type"] == "WordBoundary":
                    start = item["offset"] / 1e7
                    words.append(Word(item["text"], start, start + item["duration"] / 1e7))
        if path.stat().st_size == 0:
            raise RuntimeError("edge-tts returned no audio")
        return words

    try:
        words = with_retries(_do, attempts=3, label=f"edge-tts {voice}")
    except Exception as e:
        status = getattr(e, "status", getattr(e, "status_code", None))
        if status == 403 or "403" in str(e):
            raise RuntimeError(
                "Edge TTS was rejected with HTTP 403 by the upstream service; this may be a temporary "
                "service or network restriction. The system voice can be used instead."
            ) from e
        if isinstance(e, (TimeoutError, asyncio.TimeoutError)) or "timeout" in type(e).__name__.lower():
            raise RuntimeError("Edge TTS timed out while waiting for the upstream service.") from e
        raise RuntimeError(f"Edge TTS synthesis failed ({type(e).__name__}): {e}") from e
    return path, words, voice


def _openai(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    if not settings.openai_api_key:
        raise PermanentError("TTS_ENGINE=openai needs OPENAI_API_KEY")
    model = settings.tts_model or "gpt-4o-mini-tts"
    voice = _pick_voice(settings, "onyx")
    body = {"model": model, "voice": voice, "input": text, "response_format": "mp3"}
    if model.startswith("gpt-4o"):
        body["instructions"] = settings.tts_instructions
    else:
        body["speed"] = max(0.25, min(4.0, _rate_multiplier(settings.tts_rate)))
    path = out / "voice.mp3"

    def _do():
        r = http().post("https://api.openai.com/v1/audio/speech", json=body, timeout=180,
                        headers={"Authorization": f"Bearer {settings.openai_api_key}"})
        raise_for_status(r, "OpenAI TTS")
        path.write_bytes(r.content)

    with_retries(_do, attempts=3, label="OpenAI TTS")
    return path, [], voice


def _elevenlabs(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    if not settings.elevenlabs_api_key:
        raise PermanentError("TTS_ENGINE=elevenlabs needs ELEVENLABS_API_KEY")
    voice = _pick_voice(settings, "pNInz6obpgDQGcFmaJgB")  # "Adam" premade voice; use any voice id
    model = settings.tts_model or "eleven_multilingual_v2"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps?output_format=mp3_44100_128"
    path = out / "voice.mp3"

    def _do():
        r = http().post(url, timeout=180, headers={"xi-api-key": settings.elevenlabs_api_key},
                        json={"text": text, "model_id": model,
                              "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.3}})
        raise_for_status(r, "ElevenLabs TTS")
        return r.json()

    data = with_retries(_do, attempts=3, label="ElevenLabs TTS")
    path.write_bytes(base64.b64decode(data["audio_base64"]))
    al = data.get("alignment") or data.get("normalized_alignment") or {}
    words: list[Word] = []
    cur, cs, prev_e = "", 0.0, 0.0
    for ch, s, e in zip(al.get("characters", []), al.get("character_start_times_seconds", []),
                        al.get("character_end_times_seconds", [])):
        if ch.isspace():
            if cur:
                words.append(Word(cur, cs, prev_e))
            cur = ""
            continue
        if not cur:
            cs = s
        cur += ch
        prev_e = e
    if cur:
        words.append(Word(cur, cs, prev_e))
    return path, words, voice


_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}


def _kokoro(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    try:
        import numpy as np
        from kokoro import KPipeline
    except ImportError as e:
        raise PermanentError("TTS_ENGINE=kokoro needs:  pip install kokoro soundfile") from e
    lang_code, default_voice = KOKORO_LANG.get(settings.language, KOKORO_LANG["en"])
    voice = _pick_voice(settings, default_voice)
    with _model_lock:
        pipe = _model_cache.get(f"kokoro-{lang_code}")
        if pipe is None:
            pipe = _model_cache[f"kokoro-{lang_code}"] = KPipeline(lang_code=lang_code)
        parts = [np.asarray(audio, dtype=np.float32)
                 for _, _, audio in pipe(text, voice=voice, speed=_rate_multiplier(settings.tts_rate))]
    if not parts:
        raise RuntimeError("kokoro produced no audio")
    pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype("<i2")
    path = out / "voice.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())
    return path, [], voice


def _coqui(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    try:
        from TTS.api import TTS
    except ImportError as e:
        raise PermanentError("TTS_ENGINE=coqui needs:  pip install coqui-tts") from e
    model = settings.tts_model or "tts_models/en/ljspeech/vits"
    path = out / "voice.wav"
    with _model_lock:
        tts = _model_cache.get(model)
        if tts is None:
            tts = _model_cache[model] = TTS(model_name=model)
        tts.tts_to_file(text=text, file_path=str(path))
    return path, [], model


def _system(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    txt = out / "voice.txt"
    txt.write_text(text, encoding="utf-8")
    voice = settings.tts_voice or ""
    if sys.platform == "darwin":
        raw = out / "voice.aiff"
        cmd = ["say", "-r", str(int(185 * _rate_multiplier(settings.tts_rate))), "-o", str(raw), "-f", str(txt)]
        if voice:
            cmd[1:1] = ["-v", voice]
    elif sys.platform.startswith("win"):
        raw = out / "voice.wav"
        # Use -Encoding UTF8 for PowerShell to properly read Unicode characters
        ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.SetOutputToWaveFile('{raw}'); $s.Speak([IO.File]::ReadAllText('{txt}', [System.Text.Encoding]::UTF8)); $s.Dispose()")
        cmd = ["powershell", "-NoProfile", "-Command", ps]
    else:
        raw = out / "voice.wav"
        exe = "espeak-ng"
        cmd = [exe, "-v", voice or settings.language, "-s", str(int(165 * _rate_multiplier(settings.tts_rate))),
               "-w", str(raw), "-f", str(txt)]
    try:
        # Ensure subprocess uses UTF-8 encoding for stdout/stderr
        subprocess.run(cmd, check=True, capture_output=True, timeout=300, encoding="utf-8", errors="replace")
    except FileNotFoundError as e:
        raise PermanentError(f"System voice not available ({cmd[0]} missing). On Linux: apt install espeak-ng") from e
    except UnicodeDecodeError as e:
        # Retry without capturing output if encoding fails
        log.warning("Unicode decode error in system TTS, retrying without capture: %s", e)
        subprocess.run(cmd, check=True, timeout=300)
    return raw, [], voice or "system-default"


def _silent(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    seconds = max(2.0, len(tokenize(text)) / 2.6)
    path = out / "voice.wav"
    ffmpeg.run(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{seconds:.2f}", str(path)],
               label="silent-voice")
    return path, [], "silent"


_ENGINE_FUNCS = {"edge": _edge, "openai": _openai, "elevenlabs": _elevenlabs, "kokoro": _kokoro,
                 "coqui": _coqui, "system": _system, "silent": _silent}


# ------------------------------------------------------------------------------------------ aligners
def _align_faster_whisper(audio: Path, language: str) -> list[Word]:
    from faster_whisper import WhisperModel  # optional dependency
    with _model_lock:
        model = _model_cache.get("whisper")
        if model is None:
            model = _model_cache["whisper"] = WhisperModel("base", device="auto", compute_type="int8")
        segments, _ = model.transcribe(str(audio), language=language, word_timestamps=True)
        return [Word(w.word.strip(), w.start, w.end) for seg in segments for w in (seg.words or [])]


def _align_openai(audio: Path, language: str, api_key: str) -> list[Word]:
    with open(audio, "rb") as f:
        r = http().post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": "whisper-1", "response_format": "verbose_json", "language": language,
                  "timestamp_granularities[]": "word"},
            files={"file": (audio.name, f)}, timeout=180)
    raise_for_status(r, "OpenAI alignment")
    return [Word(w["word"], float(w["start"]), float(w["end"])) for w in r.json().get("words", [])]


def _align(audio: Path, settings: Settings) -> tuple[list[Word], str]:
    mode = settings.align
    if mode == "none":
        return [], "estimated"
    lang = settings.language.split("-")[0]
    if mode in ("auto", "faster-whisper"):
        try:
            log.info("Aligning word timings with faster-whisper (the first run downloads the model)")
            return _align_faster_whisper(audio, lang), "aligned"
        except ImportError:
            if mode == "faster-whisper":
                log.warning("ALIGN=faster-whisper but it isn't installed (pip install faster-whisper)")
        except Exception as e:
            log.warning("faster-whisper alignment failed: %s", e)
    if mode == "openai" and settings.openai_api_key:
        try:
            return _align_openai(audio, lang, settings.openai_api_key), "aligned"
        except Exception as e:
            log.warning("OpenAI alignment failed: %s", e)
    return [], "estimated"


def synthesize(text: str, out_dir: Path, settings: Settings) -> Speech:
    """Speak ``text`` and return audio + one timed Word per script token."""
    engine = settings.tts_engine if settings.tts_engine in _ENGINE_FUNCS else "edge"
    out_dir.mkdir(parents=True, exist_ok=True)
    spoken_text, groups = expand_pronunciations(text, settings.tts_pronunciations, settings.language)
    audio, native, voice = _ENGINE_FUNCS[engine](spoken_text, out_dir, settings)
    dur = ffmpeg.duration(audio)
    if dur <= 0:
        raise RuntimeError(f"{engine} produced unreadable audio: {audio}")
    tokens = tokenize(text, settings.language)
    mapped = any(n != 1 for n in groups)
    spoken_tokens = tokenize(spoken_text, settings.language) if mapped else tokens
    if native:
        words, timing = _pronounced_timings(tokens, spoken_tokens, groups, native, dur), "native"
    elif engine == "silent":
        words, timing = estimate_timings(tokens, dur), "estimated"
    else:
        aligned, timing = _align(audio, settings)
        words = (_pronounced_timings(tokens, spoken_tokens, groups, aligned, dur)
                 if aligned else estimate_timings(tokens, dur))
    log.info("Voice: %s/%s, %.1fs, %d words (%s timing)", engine, voice, dur, len(words), timing)
    return Speech(audio=audio, words=words, duration=dur, engine=engine, voice=voice, timing=timing)


def second_voice(settings: Settings) -> str:
    if settings.tts_voice_b:
        return settings.tts_voice_b
    engine = settings.tts_engine
    if engine == "edge":
        return EDGE_VOICES_B.get(settings.language.split("-")[0], EDGE_VOICES_B["en"])
    return SECOND_VOICE.get(engine, "")


def synthesize_lines(lines: list[tuple[str, str]], out_dir: Path, settings: Settings) -> Speech:
    """Voice a conversation: each (speaker, text) line with that speaker's voice, joined into one track.

    Word timings stay exact because every line is timed on its own and shifted by where it lands."""
    import concurrent.futures
    from dataclasses import replace
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_a = _pick_voice(settings, "")  # blank = engine default for speaker A
    voice_b = second_voice(settings)
    # One voice per speaker for the whole video (a comma list in TTS_VOICE must not change mid-talk).
    cfg = {"A": replace(settings, tts_voice=voice_a), "B": replace(settings, tts_voice=voice_b or voice_a)}

    def one(i: int, speaker: str, text: str) -> Speech:
        return synthesize(text, out_dir / f"line{i:02d}", cfg[speaker])

    workers = 4 if settings.tts_engine in ("edge", "openai", "elevenlabs") else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(lambda a: one(*a), [(i, sp, t) for i, (sp, t) in enumerate(lines)]))

    # Normalise every line to the same PCM format and join them with a short pause in between.
    gap = out_dir / "gap.wav"
    ffmpeg.run(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{LINE_GAP:.3f}", str(gap)], label="gap")
    listing, words, starts, t = [], [], [], 0.0
    for i, ((speaker, _), sp) in enumerate(zip(lines, parts)):
        wav = out_dir / f"line{i:02d}.wav"
        # Keep only the spoken part (plus a breath): engines pad each clip with silence, and between two
        # speakers that padding adds up to a second-long dead gap.
        cut0 = max(0.0, sp.words[0].start - 0.05) if sp.words else 0.0
        cut1 = min(sp.duration, sp.words[-1].end + 0.15) if sp.words else sp.duration
        ffmpeg.run(["-i", str(sp.audio), "-af", f"atrim=start={cut0:.3f}:end={cut1:.3f},asetpts=PTS-STARTPTS",
                    "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], label="line")
        dur = ffmpeg.duration(wav)
        starts.append(t)
        words += [Word(w.text, round(w.start - cut0 + t, 3), round(w.end - cut0 + t, 3), speaker) for w in sp.words]
        listing.append(f"file '{wav.name}'")
        t += dur
        if i + 1 < len(parts):
            listing.append(f"file '{gap.name}'")
            t += LINE_GAP
    (out_dir / "lines.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
    audio = out_dir / "voice.wav"
    ffmpeg.run(["-f", "concat", "-safe", "0", "-i", "lines.txt", "-c:a", "pcm_s16le", str(audio.name)],
               label="join voices", cwd=out_dir)
    dur = ffmpeg.duration(audio)
    timing = min((p.timing for p in parts), key=["estimated", "aligned", "native"].index)
    voices = "/".join(dict.fromkeys(p.voice for p in parts))
    log.info("Voices: %s, %.1fs, %d lines, %d words (%s timing)", voices, dur, len(lines), len(words), timing)
    return Speech(audio=audio, words=words, duration=dur, engine=parts[0].engine, voice=voices, timing=timing,
                  line_starts=starts)


def list_edge_voices(language: str = "") -> list[dict]:
    import asyncio

    import edge_tts
    voices = asyncio.run(edge_tts.list_voices())
    if language:
        voices = [v for v in voices if v["Locale"].lower().startswith(language.lower())]
    return sorted(voices, key=lambda v: v["ShortName"])
