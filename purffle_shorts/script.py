"""AI scriptwriting: one LLM call returns the whole Short — hook, scenes with per-scene footage
queries, title, description, hashtags and tags — so everything describes the same video.

A second call, the Script Doctor, scores the draft for retention and rewrites what is weak. Scripts can
also be translated for multi-language channels, and two-speaker formats (dialogue, chat) carry a
speaker per scene so each line gets its own voice."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .config import Settings
from .llm import LLMError
from .utils import clean_narration, strip_emoji, truncate_words

log = logging.getLogger("purffle")

LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "mr": "Marathi", "ur": "Urdu",
    "ar": "Arabic", "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese (Mandarin)",
    "id": "Indonesian", "tr": "Turkish", "nl": "Dutch", "pl": "Polish", "vi": "Vietnamese", "th": "Thai",
    "fil": "Filipino", "uk": "Ukrainian", "sv": "Swedish",
}

STYLES = {
    "facts": "Rapid-fire surprising facts about one specific subject, each one more surprising than the last.",
    "story": "A tight true story: a hook that opens a loop, rising tension, then a satisfying twist or payoff.",
    "listicle": "A top-3 countdown (number 3, number 2, number 1) where number 1 is the most surprising.",
    "myth": "Myth vs. fact: state a belief most people hold, then bust it with clear evidence.",
    "quiz": "Open with a question the viewer will try to answer, build suspense, reveal the answer at the end.",
    "motivational": "A punchy real-life micro-story or principle that ends with a memorable one-line takeaway.",
    "news": "Explain a current story neutrally: what happened, why it matters, what happens next.",
    "explainer": "Explain one idea so a 12-year-old gets it, using one vivid analogy.",
    "dialogue": "A fast back-and-forth between two people, A and B. A is curious or skeptical and asks; B "
                "answers with surprising, specific facts. The last line is a punchline.",
    "chat": "A text-message conversation between A and B that tells a gripping story in real time, with "
            "a twist near the end. Written exactly like real texts: short, casual, emotional.",
    "reddit": "A first-person story told like a viral Reddit post (a confession, an 'am I wrong?' or a 'today I "
              "messed up'). Scene 1 is the post's title, read aloud; the rest is the story in casual first "
              "person, building to a twist or an update at the end.",
}
MULTI_SPEAKER = {"dialogue", "chat"}
CARD_STYLES = {"reddit"}                  # single narrator, but the cast names the community and the poster
AUTO_STYLES = ["facts", "facts", "story", "story", "listicle", "myth", "quiz", "explainer"]
DEFAULT_CAST = {"dialogue": ["Alex", "Sam"], "chat": ["Unknown", "Me"], "reddit": ["r/stories", "u/throwaway"]}

# YouTube video category ids
CATEGORIES = {
    "education": "27", "science": "28", "entertainment": "24", "people": "22", "howto": "26",
    "news": "25", "gaming": "20", "autos": "2", "sports": "17", "travel": "19", "comedy": "23",
    "film": "1", "music": "10", "pets": "15",
}

SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker": {"type": "string", "enum": ["A", "B"]},
        "narration": {"type": "string"},
        "search_query": {"type": "string"},
        "image_prompt": {"type": "string"},
        "duration": {"type": "number"},
        "visual_description": {"type": "string"},
        "visual_source": {"type": "string"},
        "attribution": {"type": "string"},
    },
    "required": ["speaker", "narration", "search_query", "image_prompt"],
    "additionalProperties": False,
}

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "title": {"type": "string"},
        "hook_text": {"type": "string"},
        "angle": {"type": "string"},
        "hooks": {"type": "array", "items": {"type": "string"}},
        "selected_hook": {"type": "string"},
        "cast": {"type": "array", "items": {"type": "string"}},
        "scenes": {"type": "array", "items": SCENE_SCHEMA},
        "description": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["topic", "title", "hook_text", "cast", "scenes", "description", "hashtags", "tags", "category"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "blocking_issues": {"type": "array", "items": {"type": "string"}},
        "advisory_issues": {"type": "array", "items": {"type": "string"}},
        "script": SCRIPT_SCHEMA,
    },
    "required": ["score", "issues", "script"],
    "additionalProperties": False,
}
SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "blocking_issues": {"type": "array", "items": {"type": "string"}},
        "advisory_issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "issues"],
    "additionalProperties": False,
}

WORDS_PER_SECOND = 2.6


@dataclass
class Scene:
    narration: str
    search_query: str
    image_prompt: str = ""
    speaker: str = "A"
    duration: float | None = None        # storyboard scene duration (seconds); filled at render
    visual_description: str = ""         # what should be shown in this scene
    visual_source: str = ""              # preferred source hint: stock|ai|agency
    attribution: str = ""                # filled by the media layer from actual asset metadata


@dataclass
class Script:
    topic: str
    title: str
    hook_text: str
    scenes: list[Scene]
    description: str
    hashtags: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    category: str = "education"
    style: str = "facts"
    language: str = "en"
    cast: list[str] = field(default_factory=list)
    score: int | None = None           # Script Doctor retention score (0-100)
    review: list[str] = field(default_factory=list)  # everything the Script Doctor found
    blocking: list[str] = field(default_factory=list)  # factual/safety problems that must stop a render
    angle: str = ""                    # the specific, defensible angle chosen for this topic
    hooks: list[str] = field(default_factory=list)  # three alternative hooks the writer considered
    selected_hook: str = ""            # the hook that made it into the final script
    sources: list[str] = field(default_factory=list)  # source references when research material was supplied
    verification_report: dict | None = None  # claim-based fact verification report (Phase 2)

    @property
    def narration(self) -> str:
        return " ".join(s.narration for s in self.scenes)

    @property
    def category_id(self) -> str:
        return CATEGORIES.get(self.category, "27")

    @property
    def multi_speaker(self) -> bool:
        return any(s.speaker == "B" for s in self.scenes)

    def speaker_name(self, speaker: str) -> str:
        i = 1 if speaker == "B" else 0
        return self.cast[i] if i < len(self.cast) else speaker

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Script:
        known = {f.name for f in fields(cls)}
        scene_keys = {f.name for f in fields(Scene)}
        d = {k: v for k, v in dict(d).items() if k in known}
        d["scenes"] = [Scene(**{k: v for k, v in s.items() if k in scene_keys}) for s in d.get("scenes", [])]
        return cls(**d)


def pick_style(settings: Settings, source: str) -> str:
    if settings.style in STYLES:
        return settings.style
    if source == "trending":
        return "news"
    if source in ("wikipedia", "reddit", "rss"):
        return "story"
    return random.choice(AUTO_STYLES)


def scene_plan(style: str, target_seconds: int) -> tuple[int, int]:
    """(words, scenes) for a target length, favoring a visual beat every few seconds."""
    words = int(target_seconds * WORDS_PER_SECOND)
    if style == "chat":
        return words, max(6, min(18, round(target_seconds / 2.8)))
    if style == "dialogue":
        return words, max(6, min(14, round(target_seconds / 3.5)))
    if style == "reddit":
        return words, max(5, min(12, round(target_seconds / 4.5)))
    return words, max(4, min(10, round(target_seconds / 3.5)))


def _format_rules(style: str) -> str:
    if style == "dialogue":
        return ("- Each scene is ONE line spoken by ONE person. Alternate speakers: A, B, A, B...\n"
                "- cast: two short first names, [name of A, name of B]. Nobody says their own name.\n")
    if style == "chat":
        return ("- Each scene is ONE text message (max 18 words) from A or B. Messages may be 1-3 words. "
                "Consecutive messages from the same person are fine.\n"
                "- cast: [contact name shown at the top of the chat (e.g. Mom, Unknown Number, Jake), \"Me\"]. "
                "B is \"Me\", the phone's owner.\n"
                "- No emojis. The story must be fiction that is clearly plausible, never about real people.\n")
    if style == "reddit":
        return ("- speaker is always \"A\": the poster, telling it in the first person.\n"
                "- Scene 1 narration is ONLY the post title: a first-person question or confession, max 16 words. "
                "Spell things out for the voice (\"Am I wrong for\", not \"AITA\"; no abbreviations like TIFU).\n"
                "- cast: [a fitting community name starting with r/ (e.g. r/confessions, r/pettyrevenge), "
                "a made-up username starting with u/]. hook_text is not shown for this format.\n"
                "- Fiction written like a real post: invent every person, never name real people. No emojis.\n")
    return "- speaker is always \"A\" (one narrator). cast: [\"Narrator\"].\n"


def build_prompt(subject: str, style: str, settings: Settings, avoid: list[str], context: str = "",
                 insights: str = "") -> tuple[str, str]:
    lang = LANGUAGES.get(settings.language, settings.language)
    words, n_scenes = scene_plan(style, settings.target_seconds)
    system = (
        "You are a top YouTube Shorts scriptwriter and editor. You write fast, factual, highly "
        "re-watchable vertical videos. You reply with a single JSON object and nothing else."
    )
    avoid_block = ""
    if avoid:
        avoid_block = "Do NOT repeat any of these recent videos:\n" + "\n".join(f"- {a}" for a in avoid[:40]) + "\n\n"
    ctx = f"Background information (use it, don't invent beyond it):\n{context.strip()}\n\n" if context else ""
    ins = f"{insights.strip()}\n\n" if insights else ""
    src_rule = (
        "- Ground claims in the Background information above; list the source references you relied on\n"
        "  in \"sources\". Do not invent sources or factual claims. If reliable material is unavailable,\n"
        "  mark the claims that need verification instead of presenting them as verified.\n\n"
    ) if context else ""
    user = f"""Write one YouTube Short.

Subject: {subject}
If the subject is broad, choose ONE specific, surprising, lesser-known topic inside it and name it in
"angle" — the one defensible take this video will argue or show.
Favor a story with clear visual evidence, an immediately understandable mystery, a verifiable surprise,
and a concrete reveal. Avoid repeating recent subjects or relying on misleading mystery framing.
Format: {STYLES[style]}
Audience: {settings.audience}
Language: {lang} for narration, title, hook_text, cast and description. search_query and image_prompt are ALWAYS English.

{ctx}{ins}{avoid_block}Rules:
- Total narration between {int(words * 0.9)} and {int(words * 1.1)} words (~{settings.target_seconds} seconds spoken), split into exactly {n_scenes} scenes.
{_format_rules(style)}- Open with the most surprising well-established fact the story can support, or a precise curiosity gap if no such fact is available. No greeting, "Did you know?", or throat-clearing.
- The first narration and first visual must show the same concrete subject; name the subject or its consequence immediately, not a vague phrase such as "a cosmic transformation."
- Keep every scientific claim supportable. Never invent quotations, statistics, discoveries, or certainty; frame hypotheticals as hypothetical.
- Introduce the mystery early, escalate with one useful new detail per scene, and resolve it with a satisfying final payoff. Close the loop to the opening only when natural.
- Every line earns its place. Aim for a new visual/story beat every 2–3 seconds when it helps; don't split a thought just to force a cut.
- Propose THREE substantially different hooks in "hooks" (different angles, under 10 words each), then set "selected_hook" to the strongest one — the one Scene 1's narration delivers on.
- The last scene lands the payoff. Omit generic outros and calls to action; add a brief, natural prompt (max 6 words) only if it does not interrupt the story.
- Narration is natural spoken text by a voice: no emojis, hashtags, labels, brackets or stage directions.
- search_query: 1-4 English words describing concrete, filmable stock footage for that scene (e.g. "octopus underwater", "old library books"). No abstract words.
- image_prompt: one vivid English sentence for the exact subject/action in this scene, no text or letters in the image. Never substitute unrelated generic mystery imagery.
- visual_description: identify the actual place, geological/biological/astronomical subject, and visible details for this scene in one sentence. Do not depict an unverified event as a real photograph.
- title: max 60 characters, curiosity-driven but honest, no hashtags, no quotes.
- hook_text: max 6 words shown on screen during the first seconds.
- description: a concise, accurate summary and one relevant question. No hashtags.
- hashtags: 3-5 relevant hashtags. tags: 8-15 search keywords.
- category: one of {", ".join(CATEGORIES)}.
{src_rule}Return JSON with keys: topic, title, hook_text, angle, hooks (exactly three strings), selected_hook, cast, scenes (array of {{speaker, narration, search_query, image_prompt, visual_description}}), description, hashtags, tags, category, sources (when background information was supplied).
- HARD LIMIT: Narration total must stay within {int(words * 0.9)}–{int(words * 1.1)} words across exactly {n_scenes} scenes. No filler, no padding, no extra scenes.
"""
    return system, user


def _clean_tag(tag: str) -> str:
    return re.sub(r"[<>\"#]", "", strip_emoji(str(tag))).strip()


def _clean_hashtag(tag: str) -> str:
    t = re.sub(r"[^\w]", "", strip_emoji(str(tag)), flags=re.UNICODE)
    return f"#{t}" if t else ""


def _speaker(value) -> str:
    return "B" if str(value or "A").strip().upper().startswith("B") else "A"


def _handle(name: str, prefix: str) -> str:
    """'confessions' -> 'r/confessions'; spaces become underscores, as in real handles."""
    core = re.sub(r"^/?[ru]/", "", name.strip(), flags=re.I)
    core = re.sub(r"\s+", "_", core).strip("_/") or "anonymous"
    return prefix + core[:22]


def normalize(data: dict, subject: str, style: str, language: str) -> Script:
    multi = style in MULTI_SPEAKER
    raw_scenes = data.get("scenes") or []
    scenes: list[Scene] = []
    for s in raw_scenes:
        if isinstance(s, str):
            s = {"narration": s}
        narration = clean_narration(str(s.get("narration", "")))
        if not narration:
            continue
        query = _clean_tag(s.get("search_query") or "") or subject
        raw_duration = s.get("duration")
        duration = float(raw_duration) if raw_duration is not None else None
        scenes.append(Scene(narration=narration, search_query=truncate_words(query, 60),
                            image_prompt=str(s.get("image_prompt") or query).strip(),
                            speaker=_speaker(s.get("speaker")) if multi else "A",
                            duration=duration,
                            visual_description=str(s.get("visual_description") or "").strip(),
                            visual_source=str(s.get("visual_source") or "").strip(),
                            attribution=str(s.get("attribution") or "").strip()))
    if not scenes:
        raise ValueError("script has no narration")
    limit = 20 if style == "chat" else 12
    while len(scenes) > limit:  # merge the shortest same-speaker neighbours to keep cuts watchable
        pairs = [k for k in range(len(scenes) - 1) if scenes[k].speaker == scenes[k + 1].speaker]
        if not pairs:
            break
        i = min(pairs, key=lambda k: len(scenes[k].narration) + len(scenes[k + 1].narration))
        a, b = scenes[i], scenes.pop(i + 1)
        a.narration = f"{a.narration} {b.narration}"

    title = re.sub(r"#\w+", "", strip_emoji(str(data.get("title") or subject)))
    title = re.sub(r"[<>\"“”]", "", title).strip(" '-:")
    title = truncate_words(title or subject.title(), 90)

    hook = re.sub(r"[<>\"“”]", "", strip_emoji(str(data.get("hook_text") or ""))).strip()
    hook = truncate_words(hook, 48) if hook else truncate_words(title, 40)

    hashtags, seen = ["#shorts"], {"#shorts"}
    for h in data.get("hashtags") or []:
        h = _clean_hashtag(h)
        if h and h.lower() not in seen:
            seen.add(h.lower())
            hashtags.append(h)
    hashtags = hashtags[:6]

    tags, total, seen_t = [], 0, set()
    for t in list(data.get("tags") or []) + [subject, "shorts"]:
        t = truncate_words(_clean_tag(t), 60)
        cost = len(t) + (2 if " " in t else 0) + 1
        if t and t.lower() not in seen_t and total + cost <= 450:  # YouTube's tag limit is 500 chars
            seen_t.add(t.lower())
            tags.append(t)
            total += cost

    category = str(data.get("category") or "education").lower()
    if category not in CATEGORIES:
        category = "education"

    cast = [truncate_words(re.sub(r"[<>\"{}\[\]]", "", strip_emoji(str(c))).strip(), 24)
            for c in (data.get("cast") or []) if str(c).strip()]
    if multi or style in CARD_STYLES:
        defaults = DEFAULT_CAST[style]
        cast = (cast + defaults[len(cast):])[:2] if len(cast) < 2 else cast[:2]
        if style == "reddit":
            cast = [_handle(cast[0], "r/"), _handle(cast[1], "u/")]
    else:
        cast = []

    description = re.sub(r"(?<!\w)#\w+", "", strip_emoji(str(data.get("description") or ""))).strip()
    score = data.get("score")
    return Script(
        topic=str(data.get("topic") or subject).strip(),
        title=title,
        hook_text=hook,
        scenes=scenes,
        description=description,
        hashtags=hashtags,
        tags=tags,
        category=category,
        style=style,
        language=language,
        cast=cast,
        score=int(score) if isinstance(score, (int, float)) else None,
        review=[str(x) for x in data.get("review") or []][:8],
        blocking=[str(x) for x in data.get("blocking") or data.get("blocking_issues") or []][:5],
        angle=str(data.get("angle") or "").strip(),
        hooks=[str(h).strip() for h in (data.get("hooks") or []) if str(h).strip()][:3],
        selected_hook=str(data.get("selected_hook") or "").strip(),
        sources=[str(s).strip() for s in (data.get("sources") or []) if str(s).strip()],
        verification_report=data.get("verification_report"),
    )


def word_count(text: str, language: str = "en") -> int:
    if language.split("-")[0] in ("zh", "ja"):
        return round(len(re.sub(r"\s", "", text)) / 2.2)  # ~characters per spoken "word" beat
    return len(text.split())


def estimate_narration_duration(word_count: int, language: str = "en",
                                words_per_second: float = WORDS_PER_SECOND,
                                end_pad: float = 0.6) -> float:
    """Estimate spoken duration for a given word count.

    Accounts for speaking rate and the end pad added at render time.
    Returns seconds.
    """
    speak_time = word_count / words_per_second
    return speak_time + end_pad


def write_script(llm, subject: str, settings: Settings, *, source: str = "niche",
                 avoid: list[str] | None = None, context: str = "", style: str | None = None,
                 insights: str = "") -> Script:
    style = style if style in STYLES else pick_style(settings, source)
    if llm is None:
        return offline_script(subject, settings, style)
    system, user = build_prompt(subject, style, settings, avoid or [], context, insights)
    words, n_scenes = scene_plan(style, settings.target_seconds)
    lo_words = int(words * 0.5)  # absolute floor: at least half the target
    hi_words = int(words * 1.2)  # ceiling: ~20% over target keeps total duration within limits
    min_scenes = max(1, n_scenes // 2)

    for attempt in range(2):
        try:
            # First attempt with a generous token budget; retry with even more
            # if the script comes back truncated.
            max_tokens = 10000 if attempt == 0 else 14000
            data = llm.complete_json(system, user, SCRIPT_SCHEMA, max_tokens=max_tokens)
            script = normalize(data, subject, style, settings.language)
        except Exception as e:
            if attempt == 0:
                log.warning("Initial script generation failed, retrying: %s", e)
                continue
            raise

        actual_words = word_count(script.narration, settings.language)
        if actual_words < lo_words or len(script.scenes) < min_scenes:
            if attempt == 0:
                log.warning(
                    "Initial script too short (%d words, %d scenes; target ~%d words, %d scenes), retrying",
                    actual_words, len(script.scenes), words, n_scenes,
                )
                continue
            raise LLMError(
                f"Script generation produced an incomplete script: "
                f"{actual_words} words, {len(script.scenes)} scenes "
                f"(target ~{words} words, {n_scenes} scenes)"
            )
        if actual_words > hi_words:
            if attempt == 0:
                log.warning(
                    "Initial script too long (%d words; target ~%d, max %d), "
                    "retrying with a more concise prompt",
                    actual_words, words, hi_words,
                )
                user = user + (
                    f"\n\nIMPORTANT: The narration must be concise — target about {words} words "
                    f"total (maximum {hi_words}). Trim every scene to the essential facts; "
                    "do not add padding or extra scenes."
                )
                continue
            raise LLMError(
                f"Script generation produced an overlong script: "
                f"{actual_words} words, {len(script.scenes)} scenes "
                f"(target ~{words} words, max {hi_words} words)"
            )

        log.info("Script (%s, %d words, %d scenes): %s", style,
                 word_count(script.narration, settings.language),
                 len(script.scenes), script.title)
        return script

    raise LLMError("Script generation failed after retries")


# --------------------------------------------------------------------------------------------------
# Script Doctor — a second pass that scores the draft for retention and fixes what is weak.
# --------------------------------------------------------------------------------------------------
def review_prompt(script: Script, settings: Settings, rewrite: bool,
                  feedback: list[str] | None = None) -> tuple[str, str]:
    words, n_scenes = scene_plan(script.style, settings.target_seconds)
    lang = LANGUAGES.get(script.language, script.language)
    lo, hi = int(0.75 * words), int(1.3 * words)
    draft = {k: v for k, v in script.to_dict().items()
             if k not in ("style", "language", "score", "review", "blocking")}
    system = ("You are a ruthless YouTube Shorts retention editor. You know that viewers decide in 1-2 "
              "seconds and swipe away at any slow moment. You reply with a single JSON object and nothing else.")
    task = (f"""Then rewrite it into the strongest version you can. Keep the same subject, format, language ({lang}),
    the same JSON structure and between {lo} and {hi} words (target {words}) in {n_scenes} scenes. Do not pad or cut
    scenes to hit the count; keep the same information density. Preserve scene order and a
    coherent progression from hook to payoff. Make the opening a specific, scientifically
    defensible question or consequence. Fix ONLY the issues listed above — do not rewrite scenes that are
already good. Sharpen the hook, cut filler, make vague lines concrete, remove any claim that may not be true,
keep the payoff for the end. Do not invent facts, statistics, studies, or certainty. Put the rewritten script
in "script". Format rules to keep:
{_format_rules(script.style)}""" if rewrite else
            'Return only "score" and "issues".')
    parts = [
        f"""Review this YouTube Short script ({STYLES[script.style]}).

{json.dumps(draft, ensure_ascii=False, indent=1)}""",
    ]
    findings = []
    if script.score is not None:
        findings.append(f"previous Script Doctor score: {script.score}/100")
    if script.review:
        findings.append(f"previous issues: {'; '.join(script.review)}")
    if findings:
        parts.append("Previous Script Doctor findings on this draft:\n- " + "\n- ".join(findings))
    if feedback:
        parts.append("Previous attempt feedback:\n- " + "\n- ".join(feedback))
    blocking = list(dict.fromkeys(script.blocking or []))
    if blocking:
        parts.append("BLOCKING ISSUES THAT MUST BE FIXED IN THE REWRITE:\n- " + "\n- ".join(blocking[:5]))
    parts.append(
        f"""

Score it 0-100 for how well it will hold viewers to the end: hook strength in the first 2 seconds (most
important), curiosity gaps, pacing, specificity, payoff, and accuracy risk. Be strict: 85+ is rare.
List the concrete problems in "issues" (max 5, short sentences, most important first).
Put any problem that could make the video false, unsafe, self-contradictory or unusable in
"blocking_issues" (for example invented statistics, a study or quote nobody can verify, dangerous advice,
a claim that contradicts another scene, or a missing payoff). Put pacing/style notes in "advisory_issues".
Every issue must also appear in "issues"; if there are no blocking issues, return an empty list.
{task}"""
    )
    user = "\n\n".join(parts)
    return system, user


class ScriptQualityError(RuntimeError):
    """The bounded Script Doctor rewrite did not meet the configured minimum score."""


# --------------------------------------------------------------------------------------
# Script Doctor 2.0 — deterministic gates + a 100-point rubric + bounded targeted rewrites.
# --------------------------------------------------------------------------------------

DOCTOR_TARGET_SCORE = 90          # acceptance target: a script must score 90/100 to be rendered
DOCTOR_MAX_REWRITES = 2           # at most two targeted rewrite attempts per draft

RUBRIC: tuple[tuple[str, int, str], ...] = (
    ("hook_strength", 20, "hook strength in the first two seconds — specific, scroll-stopping, honest"),
    ("storytelling_curiosity", 20, "storytelling and curiosity gaps — information revealed progressively"),
    ("retention_pacing", 20, "retention and pacing — no filler; every line earns its place"),
    ("factual_accuracy", 15, "factual accuracy and evidence — only well-established claims"),
    ("originality", 10, "originality — a specific angle, not a generic treatment"),
    ("narration_quality", 10, "narration quality — natural, voice-friendly spoken text"),
    ("ending_payoff", 5, "ending and payoff — a satisfying close, not an abrupt stop"),
)


def doctor_structural_issues(script: Script, settings: Settings) -> list[str]:
    """Deterministic checks that need no LLM. Any one of them blocks the script.

    Covers: missing or empty narration, invalid story structure, missing title/hook,
    excessive repetition. Factual accuracy, misleading hooks and payoff quality can
    only be judged semantically, so they belong to the rubric review (blocking_issues)
    rather than here.
    """
    issues: list[str] = []
    if not script.scenes:
        return ["Invalid script structure: the script has no scenes."]
    for i, scene in enumerate(script.scenes):
        if not scene.narration or not scene.narration.strip():
            issues.append(f"Scene {i + 1} has no narration.")
    narration = (script.narration or "").strip()
    if not narration:
        issues.append("Missing or empty narration: every scene needs spoken text.")
    if not script.title.strip():
        issues.append("Missing title.")
    if not script.hook_text.strip():
        issues.append("Missing hook_text: the first seconds need an on-screen hook.")
    seen: set[str] = set()
    for scene in script.scenes:
        line = clean_narration(scene.narration).strip().lower()
        if line:
            if line in seen:
                issues.append("Excessive repetition: identical narration appears in more than one scene.")
                break
            seen.add(line)
    return issues


def doctor_schema(include_script: bool) -> dict:
    props: dict[str, Any] = {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "breakdown": {
            "type": "object",
            "properties": {name: {"type": "integer", "minimum": 0, "maximum": pts}
                           for name, pts, _ in RUBRIC},
            "required": [name for name, _, _ in RUBRIC],
        },
        "issues": {"type": "array", "items": {"type": "string"}},
        "blocking_issues": {"type": "array", "items": {"type": "string"}},
        "advisory_issues": {"type": "array", "items": {"type": "string"}},
    }
    required = ["score", "issues"]
    if include_script:
        props["script"] = SCRIPT_SCHEMA
        required.append("script")
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def doctor_prompt(script: Script, settings: Settings, rewrite: bool,
                  feedback: list[str] | None = None) -> tuple[str, str]:
    words, n_scenes = scene_plan(script.style, settings.target_seconds)
    lang = LANGUAGES.get(script.language, script.language)
    lo, hi = int(0.75 * words), int(1.3 * words)
    draft = {k: v for k, v in script.to_dict().items()
             if k not in ("style", "language", "score", "review", "blocking")}
    rubric = "\n".join(f"  - {name}: 0-{points} — {what}."
                       for name, points, what in RUBRIC)
    system = ("You are a ruthless YouTube Shorts retention editor. Viewers decide in 1-2 seconds and "
              "swipe away at any slow moment. Score honestly against the rubric: a high score never "
              "excuses a blocking flaw. Reply with a single JSON object and nothing else.")
    task = (
        f"Then rewrite the script to fix every blocking issue listed above. Keep the same subject, format, "
        f"language ({lang}), the same JSON structure and between {lo} and {hi} words (target {words}) "
        f"in exactly {n_scenes} scenes. Do not pad or cut scenes to hit the count; keep the same "
        f"information density. Preserve scene order and a coherent progression from hook to payoff. "
        f"Make the opening a specific, scientifically defensible question or consequence. Sharpen the "
        f"hook, cut filler, make vague lines concrete, remove any claim that may not be true, keep "
        f"the payoff for the end, and end with a short call to action (max 8 words). Do not invent "
        f"facts, statistics, studies, or certainty. Put the COMPLETE rewritten script in \"script\" "
        f"— the response must include every required field and must not be truncated.\n"
        f"Format rules to keep:\n" + _format_rules(script.style)
    ) if rewrite else 'Return only "score", "breakdown", "issues" and "blocking_issues".'
    parts = [
        f"""Review this YouTube Short script ({STYLES[script.style]}).

{json.dumps(draft, ensure_ascii=False, indent=1)}""",
    ]
    findings = []
    if script.score is not None:
        findings.append(f"previous Script Doctor score: {script.score}/100")
    if script.review:
        findings.append(f"previous issues: {'; '.join(script.review)}")
    if findings:
        parts.append("Previous Script Doctor findings on this draft:\n- " + "\n- ".join(findings))
    if feedback:
        parts.append("Previous attempt feedback:\n- " + "\n- ".join(feedback))
    blocking = list(dict.fromkeys(script.blocking or []))
    if blocking:
        parts.append("BLOCKING ISSUES THAT MUST BE FIXED IN THE REWRITE:\n- " + "\n- ".join(blocking[:5]))
    parts.append(
        f"""\nScore it 0-100 and break the score down:{rubric}
Target acceptance is {DOCTOR_TARGET_SCORE}/100; be strict.

List concrete problems in "issues" (max 5, short sentences, most important first).
Put any problem that could make the video false, unsafe, self-contradictory or unusable in
"blocking_issues" — for example: missing or empty narration; unsupported factual claims presented
as established facts; a misleading hook; broken story structure; excessive repetition; a missing
promised payoff; invalid script structure. Blocking issues always reject the script, whatever its
score. Put pacing/style notes in "advisory_issues". Every blocking issue must also appear in
"issues"; if there are no blocking issues, return an empty list.

Remember: an AI-generated score is not proof of factual accuracy. When the supplied material does
not support a claim, flag it instead of scoring it as verified.
{task}"""
    )
    return system, "\n\n".join(parts)


def review_script(llm, script: Script, settings: Settings) -> Script:
    """Score the script and (in rewrite mode) replace it with an improved version.

    Each round scores the current draft against the 100-point Script Doctor 2.0 rubric and,
    in rewrite mode, asks for a targeted rewrite. A rewrite is adopted only when it has valid
    narration and structure, lands in the configured word window, does not lose quality, and
    resolves the blocking issues. Invalid rewrites are discarded and the current draft is kept.
    At most DOCTOR_MAX_REWRITES rewrite rounds happen; if the draft still fails, production
    stops with a clear ScriptQualityError and the last valid draft is never replaced by an
    invalid one.
    """
    mode = settings.script_review
    if llm is None or mode == "off":
        return script
    rewrite = mode == "rewrite"
    threshold = settings.script_quality_threshold
    attempts = DOCTOR_MAX_REWRITES if rewrite else 1
    current = script
    feedback: list[str] = []
    for attempt in range(attempts):
        system, user = review_prompt(current, settings, rewrite, feedback=feedback)
        try:
            data = llm.complete_json(system, user, doctor_schema(include_script=rewrite),
                                     max_tokens=7000)
        except Exception as e:
            log.warning("Script Doctor attempt %d/%d skipped: %s", attempt + 1, attempts, e)
            if current.blocking:
                raise ScriptQualityError(
                    "Script Doctor found blocking factual/safety problems and could not verify a fix: "
                    + "; ".join(current.blocking)) from e
            return current
        if not isinstance(data, dict):
            log.warning("Script Doctor returned an unusable response; keeping the current script")
            if current.blocking:
                raise ScriptQualityError(
                    "Script Doctor found blocking factual/safety problems and returned no usable "
                    "verification: " + "; ".join(current.blocking))
            return current
        try:
            score = max(0, min(100, int(data.get("score"))))
        except (TypeError, ValueError):
            score = None
        issues = [str(i).strip() for i in data.get("issues") or [] if str(i).strip()][:5]
        blocking_issues = [
            str(i).strip()
            for i in (data.get("blocking_issues") or data.get("blocking") or [])
            if str(i).strip()
        ][:5]
        structural = doctor_structural_issues(current, settings)
        combined_blocking = list(dict.fromkeys(structural + blocking_issues))

        result = current
        if rewrite and isinstance(data.get("script"), dict):
            try:
                better = normalize(data["script"], current.topic, current.style, current.language)
            except ValueError as e:
                # The rewrite dropped required narration (or is otherwise un-normalizable):
                # record targeted feedback so the next round restores it, and keep the
                # current draft untouched.
                feedback.append(
                    "The last rewrite had no usable narration; every scene needs narration "
                    "so the video can be voiced. Keep all scenes and their narration intact, "
                    "and do not delete any scene's text."
                )
                log.warning("Script Doctor rewrite ignored: %s", e)
            else:
                target = scene_plan(current.style, settings.target_seconds)[0]
                n = word_count(better.narration, current.language)
                previous_score = current.score
                in_range = 0.75 * target <= n <= 1.3 * target
                score_gain = score is None or previous_score is None or score >= previous_score
                candidate_structural = doctor_structural_issues(better, settings)
                blocking_resolved = not blocking_issues
                if in_range and score_gain and blocking_resolved and not candidate_structural:
                    result = better
                else:
                    if not in_range:
                        feedback.append(
                            f"The last rewrite had {n} words; the target is between "
                            f"{int(0.75 * target)} and {int(1.3 * target)} words (target {target}). "
                            "Expand it while preserving every fact, the scene order, and the "
                            "hook-to-payoff progression — do not cut content to hit a shorter count."
                        )
                    elif previous_score is not None and score is not None and score < previous_score:
                        feedback.append(
                            f"The last rewrite scored {score}, below the current draft's "
                            f"{previous_score}. Keep what works and only fix the listed issues."
                        )
                    elif blocking_issues:
                        feedback.append(
                            "The last rewrite still has unresolved blocking issues: "
                            + "; ".join(blocking_issues)
                            + ". Fix those issues before resubmitting."
                        )
                    elif candidate_structural:
                        feedback.append(
                            "The last rewrite failed the structure check: "
                            + "; ".join(candidate_structural)
                        )
                    log.warning(
                        "Script Doctor rewrite ignored: %d words for a %d-word target or unresolved issues",
                        n,
                        target,
                    )

        score = score if score is not None else current.score
        result.score, result.review = score, issues
        result.blocking = combined_blocking

        if combined_blocking:
            if rewrite and attempt + 1 < attempts:
                log.warning(
                    "Script Doctor found blocking issues; requesting another bounded rewrite: %s",
                    "; ".join(combined_blocking),
                )
                current = result
                continue
            if rewrite:
                raise ScriptQualityError(
                    "Script Doctor found unresolved blocking factual/safety/usability issues: "
                    + "; ".join(combined_blocking))
            # Score-only mode reports but never stops production.
            log.warning("Script Doctor blocking issues on the draft: %s", "; ".join(combined_blocking))

        if score is not None and score < threshold:
            if rewrite and attempt + 1 < attempts:
                log.warning("Script Doctor score %d is below the %d-point threshold; requesting "
                            "another bounded rewrite", score, threshold)
                current = result
                continue
            if rewrite:
                raise ScriptQualityError(
                    f"Script Doctor score {score} remains below the {threshold}-point threshold "
                    f"after {attempts} bounded rewrite(s); rendering was stopped.")
            log.warning("Script Doctor score %d is below the %d-point threshold", score, threshold)
        changed = "rewritten" if result is not current else "kept"
        log.info("Script Doctor: %s/100, %s%s", score if score is not None else "?", changed,
                 f" — {issues[0]}" if issues else "")
        return result
    return current


# --------------------------------------------------------------------------------------------------
# Translation — one idea, many languages, same footage.
# --------------------------------------------------------------------------------------------------
def translate_script(llm, script: Script, language: str) -> Script:
    lang = LANGUAGES.get(language, language)
    src = {k: v for k, v in script.to_dict().items()
           if k not in ("style", "language", "score", "review", "blocking")}
    system = ("You are a native-level translator and YouTube Shorts adapter. You reply with a single JSON "
              "object and nothing else.")
    user = f"""Adapt this YouTube Short into {lang}.

{json.dumps(src, ensure_ascii=False, indent=1)}

Rules:
- Translate title, hook_text, cast, description and every scene's narration into natural spoken {lang},
  the way a native creator would say it (not word for word). Keep numbers and names accurate.
- Keep EXACTLY {len(script.scenes)} scenes in the same order, with the same speaker for each.
- Keep search_query and image_prompt unchanged (English).
- Hashtags and tags in {lang} where people search in {lang}, otherwise English.
Return the same JSON structure."""
    data = llm.complete_json(system, user, SCRIPT_SCHEMA)
    out = normalize(data, script.topic, script.style, language)
    if len(out.scenes) != len(script.scenes):
        raise ValueError(f"translation to {language} has {len(out.scenes)} scenes, expected {len(script.scenes)}")
    for mine, theirs in zip(out.scenes, script.scenes):
        mine.search_query, mine.image_prompt, mine.speaker = theirs.search_query, theirs.image_prompt, theirs.speaker
    out.category = script.category
    if script.style in CARD_STYLES:
        out.cast = list(script.cast)  # r/community and u/name are handles, not words to translate
    return out


def load_script(path: str | Path, settings: Settings) -> Script:
    """Read a script you wrote or edited yourself (e.g. the ``script.json`` of an earlier video) and
    clean it up exactly like an AI-written one, so it can be rendered without calling an LLM."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object with 'title' and 'scenes'")
    return script_from_data(data, settings, fallback_subject=Path(path).stem)


def script_from_data(data: dict, settings: Settings, fallback_subject: str = "my video") -> Script:
    subject = str(data.get("topic") or data.get("title") or fallback_subject)
    style = data.get("style") if data.get("style") in STYLES else "facts"
    if style not in MULTI_SPEAKER and any(_speaker(s.get("speaker")) == "B" for s in data.get("scenes") or []
                                          if isinstance(s, dict)):
        style = "dialogue"
    language = str(data.get("language") or settings.language)
    return normalize(data, subject, style, language)


# --------------------------------------------------------------------------------------------------
# Offline writer — lets `demo` produce a real video with no API keys at all.
# --------------------------------------------------------------------------------------------------
_DEMO = {
    "title": "Octopuses Are Basically Aliens",
    "hook_text": "THIS ANIMAL HAS 3 HEARTS",
    "scenes": [
        ("An octopus has three hearts, and one of them stops when it swims.", "octopus swimming"),
        ("Its blood is blue, because it carries oxygen with copper instead of iron.", "octopus blood blue"),
        ("Most of its neurons are not in its head. They are spread through its eight arms.", "octopus tentacles"),
        ("Each arm can taste what it touches, using thousands of sensors in its suckers.", "octopus suckers closeup"),
        ("And it changes color in a split second to vanish against the reef.", "octopus camouflage color change"),
        ("Which fact surprised you most? Follow for more.", "octopus"),
    ],
    "description": "Three hearts, blue blood and arms that can taste. The octopus might be the strangest "
                   "animal on Earth. Which fact surprised you most?",
    "hashtags": ["#octopus", "#ocean", "#animals", "#facts"],
    "tags": ["octopus facts", "ocean animals", "marine biology", "weird animals", "animal facts"],
    "category": "education",
}

_DEMO_DIALOGUE = [
    ("A", "Wait, octopuses have three hearts?", "octopus swimming"),
    ("B", "Three. And one of them stops beating every time it swims.", "octopus swimming underwater"),
    ("A", "That can't be healthy.", "octopus heart anatomy"),
    ("B", "It's why they'd rather crawl. Swimming wears them out.", "octopus crawling"),
    ("A", "Okay, but why is their blood blue?", "octopus blood blue"),
    ("B", "Copper. Their blood carries oxygen with copper instead of iron.", "copper blood cells"),
    ("A", "So they're basically aliens.", "octopus alien-like"),
    ("B", "Pretty much. Follow for more ocean weirdness.", "octopus"),
]

_DEMO_REDDIT = [
    ("Am I wrong for feeding my neighbor's cat every night for a whole year?", "orange cat window"),
    ("Every evening at seven, this orange cat sat outside my kitchen door and cried like it hadn't eaten in days.",
     "cat door"),
    ("So I bought him food. Good food. It became our little routine.", "cat eating"),
    ("Last week my neighbor knocked and asked why her cat was getting so fat.", "front door"),
    ("Turns out she feeds him at six. The family across the street feeds him at eight.", "suburban street"),
    ("He has been eating three dinners a night, at three different houses.", "fat cat sleeping"),
    ("We made a group chat. He is on a diet now, and he is furious about it.", "grumpy cat"),
    ("Would you have kept feeding him? Follow for the update.", "cat sunset"),
]

_DEMO_CHAT = [
    ("A", "are you still at the aquarium", "aquarium tunnel"),
    ("B", "yeah why", "aquarium fish"),
    ("A", "the octopus tank. look at it", "octopus tank"),
    ("B", "it's empty??", "empty aquarium"),
    ("A", "it's not empty. look closer", "octopus camouflage"),
    ("B", "omg it's on the glass. it was the same color as the rock", "octopus camouflage glass"),
    ("A", "they can change color in under a second", "octopus color change"),
    ("B", "that's actually terrifying", "octopus closeup"),
    ("A", "wait till you hear it has three hearts", "octopus swimming"),
    ("B", "ok follow for part 2", "octopus"),
]


def offline_script(subject: str, settings: Settings, style: str = "facts") -> Script:
    s = (subject or "").strip()
    if style in MULTI_SPEAKER:
        lines = _DEMO_DIALOGUE if style == "dialogue" else _DEMO_CHAT
        data = dict(_DEMO, topic="octopus superpowers", cast=DEFAULT_CAST[style] if style == "dialogue"
                    else ["Maya", "Me"],
                    scenes=[{"speaker": sp, "narration": n, "search_query": q, "image_prompt": q}
                            for sp, n, q in lines])
        if style == "chat":
            data.update(title="She Was Staring At An Empty Tank", hook_text="IT WAS NEVER EMPTY")
        else:
            data.update(title="Wait, Octopuses Have Three Hearts?", hook_text="3 HEARTS?!")
        return normalize(data, "octopus superpowers", style, "en")
    if style == "reddit":
        data = dict(_DEMO, topic="a cat with three families", title="He Was Eating Three Dinners A Night",
                    hook_text="THREE DINNERS A NIGHT", cast=["r/confessions", "u/quiet_otter22"],
                    description="My neighbor's cat had a secret. Would you have kept feeding him?",
                    hashtags=["#redditstories", "#cats", "#storytime"],
                    tags=["reddit stories", "cat story", "funny cat", "storytime", "confession"],
                    category="entertainment",
                    scenes=[{"narration": n, "search_query": q, "image_prompt": q} for n, q in _DEMO_REDDIT])
        return normalize(data, "a cat with three families", "reddit", "en")
    if not s or "octopus" in s.lower() or s.lower() in {"demo", "random"}:
        d = _DEMO
        data = dict(d, topic="octopus superpowers",
                    scenes=[{"narration": n, "search_query": q, "image_prompt": q} for n, q in d["scenes"]])
        return normalize(data, "octopus superpowers", "facts", "en")
    lines = [
        f"Most people think they know {s}. They don't.",
        f"The story of {s} is stranger than it looks.",
        "The details surprised even the experts who studied it.",
        "And one small fact changes how you see the whole thing.",
        "Would you have guessed that? Follow for more.",
    ]
    # Give every scene a slightly different search so stock footage doesn't repeat: the same subject with a
    # rotating visual modifier. The offline writer can't extract real nouns, but this keeps queries filmable.
    modifiers = ["closeup", "background", "detail", "wide shot", "slow motion"]
    scenes = [{"narration": n,
               "search_query": f"{s} {modifiers[i % len(modifiers)]}",
               "image_prompt": f"{s}, {modifiers[i % len(modifiers)]}"}
              for i, n in enumerate(lines)]
    data = {
        "topic": s, "title": f"The Truth About {s.title()}", "hook_text": "YOU WON'T EXPECT THIS",
        "scenes": scenes,
        "description": f"A quick look at {s}. What would you add?",
        "hashtags": ["#facts"], "tags": [s, "facts"], "category": "education",
    }
    return normalize(data, s, style if style not in MULTI_SPEAKER else "facts", settings.language)
