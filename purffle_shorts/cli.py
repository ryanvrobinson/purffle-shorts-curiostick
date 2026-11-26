"""Command line: `purffle-shorts <command>` or `python -m purffle_shorts <command>`."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from . import __version__
from .config import (
    ASPECTS,
    MAX_TARGET_SECONDS,
    MIN_TARGET_SECONDS,
    ConfigError,
    Settings,
    aspect_resolution,
    clamp_target_seconds,
    load_env,
    parse_resolution,
)

log = logging.getLogger("purffle")


def setup_logging(settings: Settings, verbose: bool = False, stream=None) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(stream or sys.stdout)
    console.setFormatter(fmt)
    # Force UTF-8 encoding on Windows
    if sys.platform.startswith("win"):
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass
        # Also set the console code page to UTF-8
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    logs = settings.data_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    file = logging.handlers.RotatingFileHandler(logs / "purffle.log", maxBytes=5_000_000, backupCount=3,
                                                encoding="utf-8")
    file.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(threadName)s %(message)s"))
    root.addHandler(console)
    root.addHandler(file)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for noisy in ("urllib3", "googleapiclient", "google", "httpx", "anthropic", "PIL", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("content & models")
    g.add_argument("--topic", help="make a video about exactly this")
    g.add_argument("--source", choices=["niche", "trending", "wikipedia", "reddit", "rss", "file", "queue"],
                   help="topic source")
    g.add_argument("--niche", action="append", help="niche to pick topics from (repeatable)")
    g.add_argument("--style", help="facts|story|listicle|myth|quiz|motivational|news|explainer|dialogue|chat|reddit|auto")
    g.add_argument("--lang", dest="language", help="language code, e.g. en, es, hi, ta, fr, ja")
    g.add_argument("--duration", dest="target_seconds", type=int,
                   help=f"target length in seconds ({MIN_TARGET_SECONDS}-{MAX_TARGET_SECONDS})")
    g.add_argument("--provider", dest="llm_provider", help="LLM provider (see `providers`)")
    g.add_argument("--model", dest="llm_model", help="LLM model name")
    g.add_argument("--tts", dest="tts_engine", help="edge|openai|elevenlabs|kokoro|coqui|system|silent")
    g.add_argument("--voice", dest="tts_voice", help="voice name/id (comma list = rotate)")
    g.add_argument("--voice-b", dest="tts_voice_b", help="second speaker's voice (dialogue and chat formats)")
    g.add_argument("--no-review", action="store_true", help="skip the Script Doctor's second pass")
    g.add_argument("--quality-threshold", dest="script_quality_threshold", type=int,
                   help="minimum Script Doctor score; rewrite mode stops if one bounded retry still misses (0-100)")
    g.add_argument("--also-lang", help="also make translated copies, e.g. es,hi (same footage)")
    g.add_argument("--visuals", help="comma list: pexels,pixabay,local,openai-images,pollinations,agnes")
    g.add_argument("--visual-mode", choices=["auto", "hybrid", "production"],
                   help="auto=stock-first with AI fallback; hybrid=alternate stock clips and AI images")
    g.add_argument("--hybrid-image-every", type=int,
                   help="hybrid mode: make every Nth scene an AI image (min 2, default 2)")
    g.add_argument("--visual-choice", choices=["stock-first", "ai-first", "suitability", "agnes-only"],
                   help="how to pick stock vs AI per scene: stock-first (default), ai-first, suitability (filmable->stock, concept->AI), agnes-only (Agnes for every scene; error if unavailable)")
    g.add_argument("--research", action="store_true", help="enable claim-based fact verification for the script")
    g.add_argument("--verify-facts", action="store_true", dest="research", help="alias for --research")
    g.add_argument("--storyboard-only", action="store_true", help="write the script and stop before rendering (draft for review)")
    r = p.add_argument_group("look & output")
    r.add_argument("--caption-style", help="kinetic|bold|boxed|neon|clean|karaoke|minimal")
    r.add_argument("--caption-position", help="upper|center|lower")
    r.add_argument("--caption-exit", help="auto|none|dust|dissolve")
    r.add_argument("--caption-exit-seconds", type=float, help="exit animation length (default 0.18)")
    r.add_argument("--caption-reduced-effects", action="store_true", default=None,
                   help="disable dust and pop-in motion (lower CPU)")
    r.add_argument("--grade", dest="color_grade", help="none|vivid|cinematic|warm|cool|bw")
    r.add_argument("--transition", help="random|none|fade|slideup|circleopen|...")
    r.add_argument("--aspect", choices=list(ASPECTS), help="9:16 Short/Reel, 16:9 YouTube, 1:1 or 4:5 feed")
    r.add_argument("--resolution", help="e.g. 1080x1920 or 720x1280")
    r.add_argument("--fps", type=int)
    r.add_argument("--encoder", dest="video_encoder", help="libx264|h264_videotoolbox|h264_nvenc|auto")
    r.add_argument("--no-music", action="store_true", help="skip background music")
    u = p.add_argument_group("publishing")
    u.add_argument("--no-upload", action="store_true", help="render only, keep the MP4")
    u.add_argument("--privacy", help="public|unlisted|private")
    u.add_argument("--publish-times", help="schedule slots, e.g. 09:00,14:00,19:00")
    p.add_argument("--env-file", help="load settings from this file (one per channel)")
    p.add_argument("-v", "--verbose", action="store_true")


def build_settings(args: argparse.Namespace) -> Settings:
    load_env(getattr(args, "env_file", None))
    s = Settings.from_env()
    ov = {k: getattr(args, k, None) for k in (
        "style", "language", "target_seconds", "llm_provider", "llm_model", "tts_engine", "tts_voice",
        "tts_voice_b", "script_quality_threshold", "caption_style", "caption_position", "caption_exit",
        "caption_exit_seconds", "caption_reduced_effects", "color_grade", "transition", "visual_mode",
        "hybrid_image_every", "fps", "video_encoder", "privacy", "visual_choice", "research")}
    if getattr(args, "niche", None):
        ov["niches"] = args.niche
    if getattr(args, "visuals", None):
        ov["visual_sources"] = [v.strip().lower() for v in args.visuals.split(",") if v.strip()]
    if getattr(args, "hybrid_image_every", None) is not None and args.hybrid_image_every < 2:
        raise ConfigError("--hybrid-image-every must be at least 2 (hybrid means both videos and images)")
    if getattr(args, "aspect", None):
        ov["resolution"] = aspect_resolution(args.aspect)
    if getattr(args, "resolution", None):
        ov["resolution"] = parse_resolution(args.resolution)
    if getattr(args, "no_review", False):
        ov["script_review"] = "off"
    if getattr(args, "also_lang", None):
        ov["also_languages"] = [x.strip().lower() for x in args.also_lang.split(",") if x.strip()]
    if getattr(args, "no_music", False):
        ov["music_volume"] = 0.0
    if getattr(args, "no_upload", False):
        ov["upload"] = False
    if getattr(args, "publish_times", None):
        ov["publish_times"] = [t.strip() for t in args.publish_times.split(",") if t.strip()]
    s = s.with_overrides(**ov)
    if s.target_seconds:
        s.target_seconds = clamp_target_seconds(s.target_seconds)
    if not 0 <= s.script_quality_threshold <= 100:
        raise ConfigError("--quality-threshold must be between 0 and 100")
    return s


# ------------------------------------------------------------------------------------------ commands
def cmd_run(args, s: Settings) -> int:
    from .pipeline import Studio
    # Generation/automation must not publish future videos without an individual review.
    s = s.with_overrides(upload=False)
    if args.batch:
        s.batch_size = args.batch
    if args.workers:
        s.workers = args.workers
    if args.delay is not None:
        s.batch_delay = args.delay
    produced = Studio(s).autopilot(count=args.count, once=args.once, topic=args.topic, source=args.source)
    return 0 if produced or not (args.count or args.once) else 1


def cmd_make(args, s: Settings) -> int:
    from .pipeline import Studio
    # --make always creates a reviewable draft; publishing is a separate explicit command.
    s = s.with_overrides(upload=False)
    studio = Studio(s)
    ok = 0
    for _ in range(max(1, args.count or 1)):
        if getattr(args, "storyboard_only", False):
            # Storyboard only: draft the script (with research verification if enabled) and stop.
            script, pick, writer = studio.draft(
                args.topic, source=args.source, style=s.style if s.style != "auto" else None,
                url=args.url, document=args.from_file)
            print(f"\nStoryboard: {script.title}")
            print(f"  Topic: {script.topic}  Style: {script.style}  Scenes: {len(script.scenes)}")
            print(f"  Hook: {script.hook_text}")
            print(f"  Score: {script.score}/100" if script.score is not None else "  Score: not reviewed")
            for i, sc in enumerate(script.scenes, 1):
                print(f"\n  Scene {i} ({sc.speaker}): {sc.narration}")
                print(f"    Visual: {sc.visual_description or '(not set)'}  Source: {sc.visual_source or '(not set)'}")
                print(f"    Search: {sc.search_query}  Image: {sc.image_prompt}")
            ok = 1
        else:
            r = studio.make(args.topic, source=args.source, script_file=args.script, url=args.url,
                            document=args.from_file)
            for x in [r, *r.variants]:
                ok += x.ok
                print_result(x)
    return 0 if ok else 1


def print_result(r) -> None:
    if not r.ok:
        print(f"\nâœ˜ {r.language + ' ' if r.language else ''}failed: {r.error}")
        return
    score = f"\n  retention score: {r.score}/100" if r.score is not None else ""
    print(f"\nâœ” {r.title}\n  folder: {r.folder}\n  status: {r.status}{score}"
          + (f"\n  https://youtube.com/shorts/{r.youtube_id}" if r.youtube_id else ""))


def cmd_demo(args, s: Settings) -> int:
    from .pipeline import Studio
    s = s.with_overrides(offline=True, upload=False)
    print("Demo: offline script + free neural voice + generated backgrounds. No API keys used.\n")
    r = Studio(s).make(args.topic, style=s.style if s.style != "auto" else None)
    if r.ok:
        print(f"\nâœ” Demo video: {r.video}\n  (cover.jpg, captions.srt and metadata.json are next to it)")
        return 0
    print(f"\nâœ˜ Demo failed: {r.error}")
    return 1


def cmd_upload(args, s: Settings) -> int:
    from .pipeline import Studio
    studio = Studio(s)
    if args.pending:
        records = studio.history.pending_uploads(("queued", "rendered"))
        if not records:
            print("No rendered or queued videos are waiting for review.")
            return 0
        completed = 0
        for rec in records:
            reviewed = _review_upload(
                rec, s, studio.history.scheduled_times(), studio.history, dry_run=args.dry_run)
            if not reviewed:
                print(f"Skipped video #{rec['id']}.")
                continue
            if args.dry_run:
                completed += 1
                print(f"Dry run passed for video #{rec['id']}; no approval or upload was made.")
                continue
            status = studio.upload_record(rec["id"])
            if status in ("uploaded", "scheduled"):
                completed += 1
            elif status == "queued":
                print(f"Video #{rec['id']} remains queued; the upload limit has been reached.")
            elif status == "approval-required":
                print(f"Video #{rec['id']} lacks a saved approval; it was not uploaded.")
            else:
                print(f"Video #{rec['id']} was not uploaded (status: {status}).")
        if args.dry_run:
            print(f"Dry run passed for {completed} of {len(records)} video(s); nothing was uploaded or scheduled.")
        else:
            print(f"Uploaded or scheduled {completed} video(s). Remaining today: {max(0, studio.quota_left())}")
        return 0 if completed or not records else 1
    if args.id:
        rec = studio.history.get(args.id)
        if not rec:
            print(f"Video #{args.id} was not found.")
            return 1
        if not _review_upload(rec, s, studio.history.scheduled_times(), studio.history, dry_run=args.dry_run):
            print("Upload cancelled.")
            return 1
        if args.dry_run:
            print(f"Dry run passed for video #{args.id}; no approval or upload was made.")
            return 0
        status = studio.upload_record(args.id)
        if status == "approval-required":
            print("Upload blocked: approval could not be verified.")
        return 0 if status in ("uploaded", "scheduled") else 1
    print("Use --pending (everything not yet uploaded) or --id N (see `history`).")
    return 2


def _review_upload(rec: dict, s: Settings, scheduled_times: list[str] | None = None, history=None,
                   dry_run: bool = False) -> bool:
    """Show a per-video publication summary and require an explicit confirmation."""
    import json

    from . import youtube
    from .quality import run_quality_checks

    folder = Path(rec["folder"]) if rec.get("folder") else None
    video = Path(rec["video_path"]) if rec.get("video_path") else None
    print(f"\n--- Review video #{rec['id']} before YouTube publishing ---")
    if rec.get("youtube_id") or rec.get("status") in ("uploaded", "scheduled"):
        print("Failed check: this record already has a YouTube upload; refusing to create a duplicate.")
        return False
    print(f"Title: {rec.get('title') or '(missing)'}")
    print(f"File: {video or '(missing)'}")
    print(f"Account: authorized account in {s.token_file} (verify the account in your browser)")
    print(f"Visibility: {s.privacy}")
    publish_at = youtube.next_publish_slot(s, scheduled_times or [], None)
    print(f"Publication: {publish_at.isoformat() if publish_at else 'immediately after upload'}")

    metadata = {}
    if folder and (folder / "metadata.json").exists():
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    print(f"Thumbnail concept: {metadata.get('thumbnail_concept', 'not generated')}")
    print(f"Description:\n{rec.get('description') or '(missing)'}")
    print(f"Hashtags/tags: {rec.get('tags') or '[]'}")
    print(f"Topic source: {metadata.get('source') or rec.get('source') or 'not recorded'}")
    research = metadata.get("research") or {}
    source_references = research.get("references") or metadata.get("source_references") or []
    print(f"Source references: {json.dumps(source_references, ensure_ascii=False) if source_references else 'not recorded'}")
    for warning in research.get("warnings", []):
        print(f"Factual verification warning: {warning}")
    print(f"Thumbnail/preview: {folder / 'cover.jpg' if folder else '(missing)'}")

    script_path = folder / "script.json" if folder else None
    if script_path and script_path.exists():
        script = json.loads(script_path.read_text(encoding="utf-8"))
        print("Narration:")
        for scene in script.get("scenes", []):
            print(f"  - {scene.get('narration', '')}")
    else:
        print("Narration: unavailable (script.json not found)")
    captions = folder / "captions.srt" if folder else None
    if captions and captions.exists():
        print(f"Captions ({captions}):\n{captions.read_text(encoding='utf-8')}")
    else:
        print("Captions: not found")

    visual_sources = sorted({str(item.get("source", "unknown")) for item in metadata.get("visuals", [])
                             if isinstance(item, dict)})
    print(f"Visual asset sources: {', '.join(visual_sources) if visual_sources else 'not recorded'}")
    print("Factual warning: AI review is not independent fact-checking; verify factual claims and citations.")
    if visual_sources:
        print("Rights warning: confirm each asset's license and permitted use before publishing.")
    else:
        print("Rights warning: asset provenance is incomplete; do not publish until verified.")

    if not video or not folder:
        print("Failed check: video path or project folder is missing.")
        return False
    report = run_quality_checks(video, folder, s)
    for check in report.checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.message}")
    if not report.passed:
        print(f"Publishing blocked: {report.summary}")
        return False

    if dry_run:
        print("Dry-run checks passed; no approval, upload, or schedule was created.")
        return True

    try:
        answer = input("Type PUBLISH to approve this video for YouTube now: ")
    except EOFError:
        return False
    if answer.strip() != "PUBLISH":
        return False
    if history is None:
        print("Failed check: approval cannot be persisted; refusing to publish.")
        return False

    from .history import now_iso

    approved_at = now_iso()
    approval = {
        "approved_at": approved_at,
        "platform": "youtube",
        "video_path": str(video),
        "title": rec.get("title"),
        "description": rec.get("description"),
        "tags": rec.get("tags"),
        "visibility": s.privacy,
        "publish_at": publish_at.isoformat() if publish_at else None,
        "source_references": source_references,
    }
    if not history.approve_video(int(rec["id"]), approved_at, approval):
        print("Failed check: approval could not be saved for this video.")
        return False
    metadata["approval_status"] = "approved"
    metadata["approval"] = approval
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Approval recorded at {approved_at}.")
    return True


def cmd_auth(args, s: Settings) -> int:
    from . import youtube

    print("\nYouTube OAuth authorization")
    print("=" * 50)
    print("1. A browser window will open to Google's authorization page.")
    print("2. Sign in to the YouTube channel you want to upload to.")
    print("3. Grant permission for PurffleShorts to upload videos on your behalf.")
    print("4. After authorization, the token is saved locally as token.json.")
    print()
    print("After this command completes, copy the contents of token.json into the")
    print("YOUTUBE_TOKEN_JSON GitHub Actions secret so unattended runs can upload.")
    print("Never paste token contents into chat or commit them to version control.")
    print()

    try:
        creds = youtube.get_credentials(s, interactive=True, extra_scopes=youtube.auth_scopes(s))
    except youtube.NotAuthorized as e:
        print(f"\nAuthorization failed: {e}")
        return 1

    # Verify refresh token is present (required for unattended GitHub Actions runs)
    if not creds.refresh_token:
        print("\nAuthorization succeeded but no refresh token was issued.")
        print("Unattended GitHub Actions uploads require a refresh token.")
        print("Revoke this client in your Google Account permissions, then run 'purffle-shorts auth' again.")
        return 1

    print(f"\nYouTube authorized. Token saved to {s.token_file}")
    print("Update the YOUTUBE_TOKEN_JSON GitHub Actions secret with the new token.json contents.")
    return 0


def cmd_history(args, s: Settings) -> int:
    from .history import History
    h = History(s.data_path / "history.db")
    rows = h.recent(args.limit)
    if not rows:
        print("No videos yet.")
        return 0
    print(f"{'ID':>4}  {'CREATED':19}  {'STATUS':9}  TITLE")
    for r in rows:
        link = f"  https://youtube.com/shorts/{r['youtube_id']}" if r["youtube_id"] else ""
        print(f"{r['id']:>4}  {r['created_at'][:19]:19}  {r['status']:9}  {r['title']}{link}")
    print("\n" + ", ".join(f"{k}: {v}" for k, v in h.stats().items()))
    return 0


def cmd_providers(args, s: Settings) -> int:
    from .llm import PRESETS, _key_for, ollama_models, ollama_url
    from .utils import http

    def running(name: str) -> bool:
        if name == "ollama":
            return ollama_models(ollama_url()) is not None
        try:
            return http().get(PRESETS[name].base_url + "/models", timeout=1.5).ok
        except Exception:
            return False

    print(f"{'PROVIDER':11} {'READY':8} {'DEFAULT MODEL':42} KEY")
    for name, p in PRESETS.items():
        ready = ("running" if running(name) else "off") if p.local else ("yes" if _key_for(p, s) else "-")
        keys = " / ".join(p.key_envs) or "(none)"
        print(f"{name:11} {ready:8} {p.default_model or '(LLM_MODEL)':42} {keys}")
    print("\nPick with LLM_PROVIDER / --provider and LLM_MODEL / --model. Any model the provider offers works.")
    return 0


def cmd_voices(args, s: Settings) -> int:
    from .tts import list_edge_voices
    voices = list_edge_voices(args.lang or s.language)
    for v in voices:
        print(f"{v['ShortName']:40} {v['Gender']:7} {v['Locale']}")
    print(f"\n{len(voices)} voices. Use one with TTS_VOICE=<name> or --voice <name>.")
    return 0


def cmd_doctor(args, s: Settings) -> int:
    from .doctor import checks
    print(f"PurffleShorts {__version__} â€” environment check\n")
    results = checks(s)
    for c in results:
        mark = {True: "âœ”", False: "âœ˜", None: "â€¢"}[c["ok"]]
        print(f" {mark} {c['label']}{': ' + c['detail'] if c['detail'] else ''}")
    ok = all(c["ok"] is not False for c in results)
    print("\nAll good." if ok else "\nFix the âœ˜ items above, then run again.")
    return 0 if ok else 1


def cmd_studio(args, s: Settings) -> int:
    from .studio import serve
    serve(s, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_plan(args, s: Settings) -> int:
    from . import analytics
    from .pipeline import Studio
    from .planner import plan_ideas
    studio = Studio(s)
    insights = analytics.insights(studio.history) if s.learn_from_stats else ""
    ideas = plan_ideas(studio.llm, s, studio.history, args.count, args.niche, insights)
    for i in ideas:
        print(f"{i['id']:>4}  {i['style'] or 'auto':12} {i['subject']}\n      {i['notes']}")
    queued = len(studio.history.ideas("pending"))
    print(f"\n{len(ideas)} idea(s) added. {queued} waiting in the queue; `run` and `make` use them first.")
    return 0 if ideas else 1


def cmd_ideas(args, s: Settings) -> int:
    from .history import History
    h = History(s.data_path / "history.db")
    if args.add:
        print(f"Added idea #{h.add_idea(args.add, args.style or '')}")
        return 0
    if args.remove:
        h.delete_idea(args.remove)
        print(f"Removed idea #{args.remove}")
        return 0
    if args.clear:
        for i in h.ideas("pending"):
            h.delete_idea(i["id"])
        print("Queue cleared.")
        return 0
    rows = h.ideas("pending")
    if not rows:
        print("The idea queue is empty. Fill it with:  purffle-shorts plan --count 10")
        return 0
    for i in rows:
        print(f"{i['id']:>4}  {i['style'] or 'auto':12} {i['subject']}" + (f"\n      {i['notes']}" if i["notes"] else ""))
    return 0


def cmd_clip(args, s: Settings) -> int:
    from .clipper import make_clips
    from .pipeline import Studio
    results = make_clips(Studio(s), args.source, count=args.count, crop=args.crop,
                         min_seconds=args.min_seconds, max_seconds=args.max_seconds, upload=False)
    for r in results:
        print_result(r)
    return 0 if any(r.ok for r in results) else 1


def cmd_stats(args, s: Settings) -> int:
    from .analytics import sync_stats
    from .history import History
    h = History(s.data_path / "history.db")
    n = sync_stats(s, h)
    rows = [r for r in h.recent(500) if r.get("views") is not None]
    if not rows:
        print("No stats yet." + ("" if n else " Upload some videos first, and set YOUTUBE_API_KEY or run `auth` "
                                         "again with LEARN_FROM_STATS=true."))
        return 0 if n or not h.live_video_ids() else 1
    rows.sort(key=lambda r: r["views"], reverse=True)
    print(f"{'VIEWS':>9} {'LIKES':>7} {'SCORE':>5}  TITLE")
    for r in rows[:args.limit]:
        score = r["score"] if r.get("score") is not None else "-"
        print(f"{r['views']:>9,} {r['likes'] or 0:>7,} {score:>5}  {r['title']}")
    return 0


def cmd_analytics_import(args, s: Settings) -> int:
    from .analytics import import_csv
    from .history import History

    history = History(s.data_path / "history.db")
    result = import_csv(args.csv_file, history)
    print(f"Imported analytics for {result.imported} video(s); skipped {result.skipped} row(s).")
    return 0 if result.imported else 1


def cmd_mcp(args, s: Settings) -> int:
    from .mcp_server import Server
    Server(s).serve()
    return 0


def cmd_daily(args, s: Settings) -> int:
    """Run the daily automation: one video per day with quality gate and upload safeguards.

    With --slot, runs one publishing slot instead: generate -> quality gate ->
    (with --publish) schedule a private YouTube video for the slot's IST time.
    Without --publish the slot run is a dry run that never touches YouTube.
    """
    from .automation import DailyAutomation, SLOT_LABELS, SlotAutomation

    if args.slot is not None:
        automation = SlotAutomation(
            s, slot=args.slot, run_date=args.date,
            dry_run=not args.publish, skip_quality=args.skip_quality,
        )
        result = automation.run_slot()
        label = SLOT_LABELS[args.slot]
        print(f"\nCurioStick slot {args.slot} ({label}) for {result.run_date}")
        if result.publish_at_utc:
            print(f"  Publish target: {result.publish_at_ist} = {result.publish_at_utc} UTC")
        if result.dry_run:
            print("  Dry run: no upload was made; use --publish to schedule for real.")
        if result.skipped:
            print(f"  Skipped: {result.skip_reason}")
        if result.uploaded:
            print(f"  Scheduled on YouTube: https://youtube.com/shorts/{result.youtube_id}")
        if result.error:
            print(f"  Error: {result.error}")
            return 1
        return 0

    automation = DailyAutomation(
        s,
        dry_run=args.dry_run,
        skip_quality=args.skip_quality,
        skip_upload=True,
        skip_instagram=not args.instagram,
    )
    result = automation.run_once()

    if result.ok:
        if result.video_result:
            print(f"âœ… Daily run completed: {result.video_result.title}")
            if result.uploaded:
                print("  â†’ Uploaded to YouTube")
            if result.instagram_published:
                print("  â†’ Published to Instagram Reels")
        else:
            print(f"âœ… Daily run: {result.error}")
        return 0
    else:
        print(f"âŒ Daily run failed: {result.error}")
        return 1


def cmd_recover(args, s: Settings) -> int:
    """Recover failed slot runs from the persistent recovery queue."""
    from .automation import SlotAutomation
    from .history import History

    history = History(s.data_path / "history.db")
    pending = history.pending_recoveries(limit=100)

    if args.slot is not None:
        pending = [r for r in pending if r["slot"] == args.slot]
    if args.date is not None:
        pending = [r for r in pending if r["run_date"] == args.date]

    if not pending:
        print("No recoverable failed jobs in the queue.")
        return 0

    print(f"Found {len(pending)} recoverable failed job(s):")
    for r in pending:
        print(f"  #{r['id']}  {r['run_date']} slot-{r['slot']}  stage={r['stage']}  "
              f"attempt={r['retry_count']}/{r['max_retries']}  "
              f"recoverable={'yes' if r['recoverable'] else 'no'}")
        if r["error"]:
            print(f"      error: {r['error'][:120]}")

    if args.dry_run:
        print(
            "\\nDry run: would attempt "
            f"{len(pending)} job(s). "
            "Use without --dry-run to execute."
        )
        return 0

    recovered = 0
    for r in pending:
        automation = SlotAutomation(
            s, slot=r["slot"], run_date=r["run_date"], recover=True,
            history=history,
        )
        result = automation.run_slot()
        if result.ok:
            recovered += 1
            print(f"  #{r['id']} recovered: {result.key} -> {result.status or 'ok'}")
        else:
            print(f"  #{r['id']} still failed: {result.error}")

    print(
        "\\nRecovered "
        f"{recovered} of {len(pending)} job(s)."
    )
    return 0 if recovered == len(pending) else 1


def cmd_produce(args, s: Settings) -> int:
    """Create and validate one persistent 10-Short production batch."""
    from .production import ProductionManifest

    manifest = ProductionManifest(s, start_date=args.start_date)
    print(f"Production manifest: {manifest.path}")
    print(f"Generating batch {args.batch} (Shorts {((args.batch - 1) * 10 + 1):03d}-"
          f"{args.batch * 10:03d})")

    def progress(stage: str, percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {stage}: {message}")

    results = manifest.generate_batch(args.batch, progress=progress)
    passed = sum(1 for row in results if row["ok"])
    failed = len(results) - passed
    print(f"\nBatch {args.batch}: {passed} passed, {failed} failed")
    for row in results:
        if row["ok"]:
            print(f"  CS: {row['content_id']} -> {row.get('video', 'already completed')}")
        else:
            print(f"  CS: {row['content_id']} -> FAILED: {row['error']}")
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="purffle-shorts", description="Autonomous AI YouTube Shorts studio.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="autopilot: make (and upload) Shorts on a loop")
    _common(r)
    r.add_argument("--count", type=int, help="stop after N successful videos")
    r.add_argument("--once", action="store_true", help="run a single batch, then exit")
    r.add_argument("--batch", type=int, help="videos per batch")
    r.add_argument("--workers", type=int, help="videos made in parallel")
    r.add_argument("--delay", type=int, help="seconds between batches")
    r.set_defaults(func=cmd_run)

    daily = sub.add_parser("daily", help="run the daily automation (one video per day with quality gate)")
    _common(daily)
    daily.add_argument("--dry-run", action="store_true", help="verify config only, no generation/upload")
    daily.add_argument("--skip-quality", action="store_true", help="skip quality gate checks")
    daily.add_argument("--skip-upload", action="store_true",
                       help="deprecated; daily runs always stop at review and never publish")
    daily.add_argument("--slot", type=int, choices=[1, 2, 3],
                       help="publishing slot: 1 = 09:00 IST, 2 = 14:00 IST, 3 = 19:00 IST")
    daily.add_argument("--date", metavar="YYYY-MM-DD",
                       help="Asia/Kolkata date for the slot (default: today in --timezone)")
    daily.add_argument("--publish", action="store_true",
                       help="actually schedule the video on YouTube; without it the run is dry-run only")
    daily.add_argument("--instagram", action="store_true", help="enable Instagram Reels publishing")
    daily.set_defaults(func=cmd_daily)

    m = sub.add_parser("make", help="make one video now")
    _common(m)
    m.add_argument("--count", type=int, default=1)
    m.add_argument("--script", metavar="FILE",
                   help="render your own script JSON (e.g. an edited script.json) instead of asking the LLM")
    m.add_argument("--url", help="make the video from a web page (article, blog post, docs)")
    m.add_argument("--from-file", metavar="FILE", help="make the video from a .txt/.md/.html document")
    m.set_defaults(func=cmd_make)

    c = sub.add_parser("clip", help="cut a long video (file or URL) into captioned Shorts of its best moments")
    _common(c)
    c.add_argument("source", help="video file, or a URL (needs: pip install yt-dlp)")
    c.add_argument("--count", type=int, default=3, help="how many clips (default 3)")
    c.add_argument("--crop", choices=["center", "blur"], default="center",
                   help="center = fill the frame; blur = whole picture over a blurred copy")
    c.add_argument("--min-seconds", type=int, default=20)
    c.add_argument("--max-seconds", type=int, default=58)
    c.set_defaults(func=cmd_clip)

    pl = sub.add_parser("plan", help="let the AI plan a batch of video ideas into the queue")
    _common(pl)
    pl.add_argument("--count", type=int, default=10)
    pl.set_defaults(func=cmd_plan)

    ide = sub.add_parser("ideas", help="show or edit the idea queue")
    ide.add_argument("--add", metavar="SUBJECT", help="queue an idea")
    ide.add_argument("--style", help="format for --add")
    ide.add_argument("--remove", type=int, metavar="ID")
    ide.add_argument("--clear", action="store_true", help="remove every pending idea")
    ide.add_argument("--env-file")
    ide.set_defaults(func=cmd_ideas)

    stt = sub.add_parser("stats", help="refresh and show view counts of your uploaded videos")
    stt.add_argument("--limit", type=int, default=25)
    stt.add_argument("--env-file")
    stt.add_argument("-v", "--verbose", action="store_true")
    stt.set_defaults(func=cmd_stats)

    ai = sub.add_parser("analytics-import", help="import a YouTube Studio analytics CSV export")
    ai.add_argument("csv_file", metavar="CSV_FILE", help="CSV with a Video ID column and available metrics")
    ai.add_argument("--env-file")
    ai.set_defaults(func=cmd_analytics_import)

    mc = sub.add_parser("mcp", help="run as an MCP server (stdio) for Claude Desktop, Claude Code, Cursor...")
    mc.add_argument("--env-file")
    mc.add_argument("-v", "--verbose", action="store_true")
    mc.set_defaults(func=cmd_mcp)

    rec = sub.add_parser("recover", help="recover failed slot runs from the recovery queue")
    rec.add_argument("--slot", type=int, choices=[1, 2, 3],
                     help="only recover this slot")
    rec.add_argument("--date", metavar="YYYY-MM-DD",
                     help="only recover this Asia/Kolkata date")
    rec.add_argument("--dry-run", action="store_true",
                     help="list recoverable jobs without attempting them")
    rec.set_defaults(func=cmd_recover)

    prod = sub.add_parser("produce", help="generate and validate one 10-Short CurioStick batch")
    prod.add_argument("--batch", type=int, choices=[1, 2, 3], required=True)
    prod.add_argument("--start-date", metavar="YYYY-MM-DD",
                      help="first production day (Asia/Kolkata)")
    prod.set_defaults(func=cmd_produce)

    d = sub.add_parser("demo", help="render a sample Short with no API keys")
    _common(d)
    d.set_defaults(func=cmd_demo)

    up = sub.add_parser("upload", help="upload rendered or queued videos")
    _common(up)
    up.add_argument("--pending", action="store_true", help="upload everything not yet on YouTube")
    up.add_argument("--id", type=int, help="upload one video by history id")
    up.add_argument("--dry-run", action="store_true", help="run review and quality checks without approval or upload")
    up.set_defaults(func=cmd_upload)

    for name, func, helptext in [("auth", cmd_auth, "connect your YouTube channel (OAuth)"),
                                 ("doctor", cmd_doctor, "check ffmpeg, the LLM, voices, keys and YouTube setup"),
                                 ("providers", cmd_providers, "list supported LLM providers")]:
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--env-file")
        sp.add_argument("-v", "--verbose", action="store_true")
        sp.set_defaults(func=func)

    h = sub.add_parser("history", help="list videos made so far")
    h.add_argument("--limit", type=int, default=25)
    h.add_argument("--env-file")
    h.set_defaults(func=cmd_history)

    v = sub.add_parser("voices", help="list free neural voices (edge-tts)")
    v.add_argument("--lang", help="filter by language/locale, e.g. en, en-GB, hi")
    v.add_argument("--env-file")
    v.set_defaults(func=cmd_voices)

    st = sub.add_parser("studio", help="local web dashboard")
    _common(st)
    st.add_argument("--host", default="127.0.0.1")
    st.add_argument("--port", type=int, default=8765)
    st.add_argument("--no-browser", action="store_true")
    st.set_defaults(func=cmd_studio)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    home = os.getenv("PURFFLE_HOME")
    if home:
        os.chdir(Path(home).expanduser())
    try:
        s = build_settings(args)
    except ConfigError as e:
        print(f"Setting problem: {e}", file=sys.stderr)
        return 2
    # MCP speaks JSON-RPC on stdout, so its logs go to stderr.
    setup_logging(s, getattr(args, "verbose", False), sys.stderr if args.command == "mcp" else None)
    try:
        return args.func(args, s)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except Exception as e:  # clean one-line errors for config problems
        from .llm import LLMError
        from .youtube import NotAuthorized
        if isinstance(e, (LLMError, NotAuthorized)):
            log.error("%s", e)
            return 2
        raise
