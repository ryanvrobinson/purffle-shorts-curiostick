"""Everything drawn on top of the footage: word-synced animated captions, the hook title, the
watermark, the end call-to-action, for chat stories an animated text-message screen, and for
Reddit-style stories the post card shown while the title is read.

Two renderers produce the same look:
  * Pillow (default) — captions are pre-rendered PNG states played back by ffmpeg's concat demuxer
    as a single transparent overlay stream (active-word highlight, pop-in, glow, boxes).
  * libass — used for scripts that need complex text shaping (Hindi, Tamil, Arabic, Thai, ...)
    when Pillow was built without raqm, or when CAPTION_RENDERER=ass.
"""

from __future__ import annotations

import functools
import logging
import os
import random
import shutil
from dataclasses import dataclass, fields, replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from . import emphasis as emphasis_rules
from . import ffmpeg
from .config import Settings
from .timing import NO_SPACE_LANGS, Chunk
from .utils import download

log = logging.getLogger("purffle")

# ------------------------------------------------------------------------------------------ fonts
_GF = "https://github.com/google/fonts/raw/main/ofl/"
FONT_SOURCES = {
    "anton": ("Anton-Regular.ttf", _GF + "anton/Anton-Regular.ttf", "Anton"),
    "sans": ("NotoSans-Variable.ttf", _GF + "notosans/NotoSans%5Bwdth,wght%5D.ttf", "Noto Sans"),
    # Inter: the clean, modern sans the premium look is built on. Variable font, weight axis 100-900.
    "inter": ("Inter-Variable.ttf", _GF + "inter/Inter%5Bopsz,wght%5D.ttf", "Inter"),
    "deva": ("NotoSansDevanagari-Variable.ttf", _GF + "notosansdevanagari/NotoSansDevanagari%5Bwdth,wght%5D.ttf",
             "Noto Sans Devanagari"),
    "taml": ("NotoSansTamil-Variable.ttf", _GF + "notosanstamil/NotoSansTamil%5Bwdth,wght%5D.ttf", "Noto Sans Tamil"),
    "telu": ("NotoSansTelugu-Variable.ttf", _GF + "notosanstelugu/NotoSansTelugu%5Bwdth,wght%5D.ttf",
             "Noto Sans Telugu"),
    "beng": ("NotoSansBengali-Variable.ttf", _GF + "notosansbengali/NotoSansBengali%5Bwdth,wght%5D.ttf",
             "Noto Sans Bengali"),
    "arab": ("NotoSansArabic-Variable.ttf", _GF + "notosansarabic/NotoSansArabic%5Bwdth,wght%5D.ttf",
             "Noto Sans Arabic"),
    "thai": ("NotoSansThai-Variable.ttf", _GF + "notosansthai/NotoSansThai%5Bwdth,wght%5D.ttf", "Noto Sans Thai"),
    "jpan": ("NotoSansJP-Variable.ttf", _GF + "notosansjp/NotoSansJP%5Bwght%5D.ttf", "Noto Sans JP"),
    "kore": ("NotoSansKR-Variable.ttf", _GF + "notosanskr/NotoSansKR%5Bwght%5D.ttf", "Noto Sans KR"),
    "hans": ("NotoSansSC-Variable.ttf", _GF + "notosanssc/NotoSansSC%5Bwght%5D.ttf", "Noto Sans SC"),
}
LANG_FONT = {"hi": "deva", "mr": "deva", "ta": "taml", "te": "telu", "bn": "beng", "ar": "arab", "ur": "arab",
             "th": "thai", "ja": "jpan", "ko": "kore", "zh": "hans", "ru": "sans", "uk": "sans"}
COMPLEX_LANGS = {"hi", "mr", "ta", "te", "bn", "ar", "ur", "th"}

# Font fallback chain. Inter is looked for first (installed by the user, or a system package), then a
# neutral UI sans, then whatever bold face the platform certainly ships. Each entry is (regular, bold);
# None means "this family has no separate bold file, use its variable weight axis".
SYSTEM_FALLBACK: list[tuple[str | None, str | None]] = [
    ("C:/Windows/Fonts/Inter-Regular.ttf", "C:/Windows/Fonts/Inter-Bold.ttf"),
    ("C:/Windows/Fonts/InterVariable.ttf", "C:/Windows/Fonts/InterVariable.ttf"),
    ("/usr/share/fonts/truetype/inter/Inter-Regular.ttf", "/usr/share/fonts/truetype/inter/Inter-Bold.ttf"),
    ("/usr/local/share/fonts/Inter-Regular.ttf", "/usr/local/share/fonts/Inter-Bold.ttf"),
    ("/Library/Fonts/Inter-Regular.otf", "/Library/Fonts/Inter-Bold.otf"),
    ("/System/Library/Fonts/Supplemental/Inter-Regular.ttf", "/System/Library/Fonts/Supplemental/Inter-Bold.ttf"),
    ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
]

# Last resort: a heavy display face, kept for styles that want Anton-like impact without a download.
SYSTEM_BOLD = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]


def _first_existing(paths: list[str]) -> Path | None:
    for cand in paths:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def font_key(language: str, style_font: str) -> str:
    lang = language.split("-")[0]
    return LANG_FONT.get(lang, style_font)


def ensure_font(key: str, data_dir: Path) -> Path | None:
    """Download an open-licence Google Font once into data/fonts (falls back to system fonts)."""
    fname, url, _ = FONT_SOURCES[key]
    dest = data_dir / "fonts" / fname
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    try:
        download(url, dest, timeout=60, max_bytes=40 * 1024 * 1024)
        return dest
    except Exception as e:
        log.warning("Could not download font %s (%s); using a system font", fname, e)
        return None


def resolve_font(settings: Settings, style_font: str = "anton", bold: bool = False,
                 offline: bool = False) -> tuple[Path | None, str]:
    """Return (font file, family name), in the order that keeps the look consistent:

    1. ``CAPTION_FONT`` (an explicit path always wins),
    2. ``CAPTION_FONT_FAMILY`` (e.g. ``inter``), downloaded once or found on the system,
    3. the style's own font (``anton`` for the bold styles, ``sans`` for clean text), shaped to the
       script's language for Devanagari, Tamil, Arabic, ... ,
    4. the system fallback chain — Inter, then a neutral UI sans, then DejaVu/Liberation,
    5. Pillow's built-in bitmap font (family ``"Arial"``, path ``None``) so a render never fails.
    """
    family = (settings.caption_font_family or "").strip().lower()
    bold = bold or getattr(settings, "caption_bold", False)
    if settings.caption_font and Path(settings.caption_font).exists():
        return Path(settings.caption_font), Path(settings.caption_font).stem
    if family and family in FONT_SOURCES:
        key = font_key(settings.language, family)
        path = None if (offline and key == family) else ensure_font(key, settings.data_path)
        if path:
            return path, FONT_SOURCES[key][2]
        # A requested family that could not be downloaded still falls through to a system face below.
    key = font_key(settings.language, style_font)
    path = ensure_font(key, settings.data_path)
    if path:
        return path, FONT_SOURCES[key][2]
    if style_font == "anton" and not settings.caption_uppercase:
        # Lowercase display text: Anton has no lowercase-friendly rhythm, so prefer the sans chain.
        path = ensure_font(font_key(settings.language, "sans"), settings.data_path)
        if path:
            return path, FONT_SOURCES[font_key(settings.language, "sans")][2]
    regulars = [reg for reg, _ in SYSTEM_FALLBACK]
    if bold:
        bolds = [b for _, b in SYSTEM_FALLBACK if b]
        path = _first_existing(bolds) or _first_existing(regulars)
    else:
        path = _first_existing(regulars)
    if path:
        return path, path.stem
    for cand in SYSTEM_BOLD:
        if Path(cand).exists():
            return Path(cand), Path(cand).stem
    return None, "Arial"


@functools.lru_cache(maxsize=64)
def load_font(path: str | None, size: int, weight: int = 800) -> ImageFont.FreeTypeFont:
    size = max(8, int(size))
    if not path:
        return ImageFont.load_default(size=size)
    font = ImageFont.truetype(path, size)
    try:
        axes = font.get_variation_axes()
    except Exception:
        return font  # static font
    values = []
    for ax in axes:
        name = ax.get("name", b"")
        name = name.decode(errors="ignore") if isinstance(name, bytes) else str(name)
        if "eight" in name:  # Weight
            values.append(max(ax["minimum"], min(ax["maximum"], weight)))
        else:
            values.append(ax.get("default", ax["maximum"]))
    try:
        font.set_variation_by_axes(values)
    except Exception:
        pass
    return font


def needs_ass(settings: Settings) -> bool:
    mode = os.getenv("CAPTION_RENDERER", "auto").lower()
    if mode == "ass":
        return True
    if mode == "pillow":
        return False
    return settings.language.split("-")[0] in COMPLEX_LANGS and not features.check("raqm")


# ------------------------------------------------------------------------------------------ design
# The premium caption design system. One frozen record holds every visual decision the renderers
# share — colours, weights, size and line metrics, letter spacing, alignment, screen-safe margins,
# the charcoal backing used only to keep text readable, the emphasis treatment and the entrance and
# exit parameters. Settings override it, styles layer small per-look tweaks on top of it, and both
# renderers (Pillow and libass) read the same values so a video looks the same either way.
#
# Vertical 9:16 platform overlays (likes, comments, the app's own caption button) sit along the
# bottom and top edges, so captions stay inside the safe area and away from those bands by default
# (safe_margin, max_lines, and the vertical band in CENTER_Y).


@dataclass(frozen=True)
class CaptionDesign:
    # type
    font: str = "inter"                    # FONT_SOURCES key; inter -> sans -> system fallback
    weight: int = 800                      # variable-font weight for ordinary words
    bold: bool = False                     # prefer a real bold file when the family has one
    size_ratio: float = 0.085              # font size as a fraction of the frame's reference unit
    min_size_ratio: float = 0.045          # never shrink below this while fitting a caption
    line_height: float = 1.18              # line box height as a multiple of the font size
    line_gap: float = 0.10                 # extra space between lines, as a fraction of the font size
    letter_spacing: float = 0.01           # tracking, as a fraction of the font size
    align: str = "center"                  # center|left|right
    max_lines: int = 3                     # balanced wrapping never needs more than three
    safe_margin: float = 0.08              # minimum distance from the left/right frame edges
    box_pad: float = 0.45                  # inner padding of the boxed style, per font size
    box_radius: float = 0.35               # corner radius of the boxed style, per font size
    shadow_offset: float = 0.05            # drop shadow offset, per font size
    stroke_scale: float = 1.0              # multiplies the style's stroke ratio
    # colour
    ink: str = "#FFF7E8"                   # warm white for ordinary words
    signal: str = "#FF7A18"                # controlled signal orange for emphasised words
    emphasis: str = ""                     # blank = the style's active colour (the signal orange)
    emphasis_brightness: float = 1.12      # brightness lift applied to emphasised words
    scrim: str = "#0B0B0F"                 # charcoal backing colour
    scrim_alpha: int = 96                  # backing opacity (kept low: no heavy boxes)
    scrim_pad: float = 0.18                # backing padding around the text, per font size
    contrast: str = "auto"                 # auto|never|safe — draw the backing only when it helps
    # emphasis and motion
    emphasis_mode: str = "auto"            # auto|off|all
    emphasis_max: int = 2                  # at most this many emphasised words per caption
    emphasis_scale: float = 1.12           # scale of an emphasised word
    spoken_scale: float = 1.06             # scale of an already-spoken word
    entrance_scales: tuple[float, ...] = (0.86, 1.04)  # pop-in steps (subtle: no bouncing)
    entrance_seconds: float = 0.10
    exit_seconds: float = 0.18
    transition_stagger: float = 0.35       # 0..1: exit overlaps the next caption by this much
    dust_particles: int = 22
    dust_particles_reduced: int = 10
    reduced: bool = False                  # CAPTION_REDUCED_EFFECTS: dust -> dissolve, no pop-in


CAPTION_DESIGN_BASE = CaptionDesign()

# The design the renderers currently use. build_overlays() sets it from Settings once per render, so
# the renderer internals stay simple while still being fully configurable.
CURRENT_DESIGN: CaptionDesign = CAPTION_DESIGN_BASE


def active_design() -> CaptionDesign:
    """The design in force for the render that is running."""
    return CURRENT_DESIGN


def design_from_settings(settings: Settings, style: CapStyle | None = None) -> CaptionDesign:
    """Fold Settings and an optional per-style override into one :class:`CaptionDesign`.

    Only the fields a style actually changes are copied, so a style can nudge the look (``minimal``
    prefers a lighter weight) without silently resetting everything else to the stock design.
    """
    mode = (settings.caption_contrast or "auto").lower()
    align = (settings.caption_align or "center").lower()
    base = CAPTION_DESIGN_BASE
    d = replace(
        base,
        font=(settings.caption_font_family or base.font).lower(),
        bold=bool(settings.caption_bold),
        size_ratio=base.size_ratio * max(0.4, min(2.5, settings.caption_scale or 1.0)),
        line_height=max(0.9, min(2.0, settings.caption_line_height or base.line_height)),
        letter_spacing=max(0.0, min(0.2, settings.caption_spacing)),
        align=align if align in {"center", "left", "right"} else base.align,
        safe_margin=max(0.0, min(0.3, settings.caption_safe_margin)),
        emphasis=settings.caption_emphasis_color or base.emphasis,
        emphasis_mode=emphasis_rules.normalize_mode(settings.caption_emphasis),
        emphasis_max=max(0, settings.caption_emphasis_max),
        contrast=mode if mode in {"auto", "never", "safe"} else base.contrast,
        exit_seconds=max(0.0, min(0.5, settings.caption_exit_seconds)),
        transition_stagger=max(0.0, min(1.0, settings.caption_transition_stagger)),
        reduced=bool(settings.caption_reduced_effects),
    )
    if style is not None and style.design is not None:
        changed = {f.name: getattr(style.design, f.name) for f in fields(style.design)
                   if getattr(style.design, f.name) != getattr(base, f.name)}
        if changed:
            d = replace(d, **changed)
    if d.reduced:  # the accessibility/resource-saver profile: calm motion, no colour cast
        d = replace(d, entrance_scales=(1.0,), emphasis_scale=min(d.emphasis_scale, 1.06),
                    spoken_scale=1.0, dust_particles=d.dust_particles_reduced,
                    contrast="never" if d.contrast == "auto" else d.contrast)
    return d


# ------------------------------------------------------------------------------------------ styles
@dataclass(frozen=True)
class CapStyle:
    fill: str = "#FFFFFF"
    active: str | None = "#FFE11A"
    spoken: str | None = None
    stroke: str = "#000000"
    stroke_ratio: float = 0.10
    shadow: bool = True
    box: str | None = None
    glow: str | None = None
    active_scale: float = 1.12
    pop: bool = True
    font: str = "anton"
    design: "CaptionDesign | None" = None      # per-style overrides of the shared design system


CAPTION_STYLES = {
    "kinetic": CapStyle(fill="#FFF7E8", active="#FF7A18", stroke="#080808", stroke_ratio=0.09,
                        active_scale=1.14, pop=True),
    "bold": CapStyle(fill="#FFF7E8", active="#FF7A18", stroke="#080808", stroke_ratio=0.10),
    "boxed": CapStyle(active="#FFFFFF", box="#7C3AED", active_scale=1.0, stroke_ratio=0.07),
    "neon": CapStyle(active="#39FF14", glow="#00E5FF", stroke="#001018", stroke_ratio=0.05, shadow=False),
    "clean": CapStyle(active="#FFFFFF", active_scale=1.06, stroke_ratio=0.0, font="sans"),
    "karaoke": CapStyle(active="#FFE11A", spoken="#FFE11A", active_scale=1.06),
    "minimal": CapStyle(active=None, active_scale=1.0, pop=False),
}
# Dialogue: speaker B's highlight and name tag use a second colour so viewers can tell who is talking.
SPEAKER_B = {"active": "#4FC3F7", "box": "#0EA5E9", "tag": "#4FC3F7"}
SPEAKER_A_TAG = "#FFE11A"
POP_SCALES = (0.78, 1.07)
KINETIC_CREAM = "#FFF7E8"
KINETIC_ORANGE = "#FF7A18"
EXIT_MODES = {"auto", "none", "dust", "dissolve"}
CENTER_Y = {"upper": 0.30, "center": 0.56, "lower": 0.68}

# ------------------------------------------------------------------ subscribe end card
# One reusable animated end card rendered into every Short, inside the total
# duration (never appended after it). Roughly 2-3 seconds: a reveal, a subtle
# scale pulse and an optional bell, then a fade. No fake clickable controls.
END_CARD_SECONDS = 2.5
END_CARD_TEXT = "SUBSCRIBE FOR MORE"
END_CARD_BELL = "\U0001f514"  # bell glyph; drawn as a fallback shape when unsupported


def _rgba(hex_color: str | None, alpha: int = 255) -> tuple[int, int, int, int]:
    """A colour as RGBA. ``None`` becomes fully transparent, so a style with no stroke simply has none."""
    if not hex_color:
        return (0, 0, 0, 0)
    h = str(hex_color).lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha
    except ValueError:
        return (0, 0, 0, alpha)


def _brighten(hex_color: str | None, factor: float) -> str:
    """Multiply a colour's brightness (clamped), so an emphasised word can read brighter without
    leaving the palette. A factor of 1.0 returns the colour untouched; ``None`` stays ``None``."""
    if not hex_color or factor == 1.0:
        return hex_color
    h = str(hex_color).lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        rgb = [min(255, int(round(int(h[i:i + 2], 16) * factor))) for i in (0, 2, 4)]
    except ValueError:
        return hex_color
    return "#" + "".join(f"{c:02X}" for c in rgb)


def display_word(text: str, uppercase: bool) -> str:
    t = text.strip().strip(",.;:\"“”()[]")
    t = t or text.strip()
    return t.upper() if uppercase else t


# ------------------------------------------------------------------------------------------ Pillow captions
class PillowCaptions:
    def __init__(self, settings: Settings, style: CapStyle, font_path: Path | None,
                 labels: dict[str, str] | None = None, design: CaptionDesign | None = None):
        self.s = settings
        self.design = design or design_from_settings(settings, style)
        if self.design.reduced:
            style = replace(style, active_scale=min(style.active_scale, 1.06), pop=False, glow=None)
        self.style = style
        self.font_path = str(font_path) if font_path else None
        self.W, self.H = settings.resolution
        self.band_h = int(self.H * 0.30) // 2 * 2
        cy = CENTER_Y.get(settings.caption_position, CENTER_Y["center"])
        self.band_y = int(self.H * cy - self.band_h / 2)
        # Screen-safe margins: never wider than the frame minus a real margin on both sides.
        self.max_w = max(120, int(self.W * (1.0 - 2.0 * self.design.safe_margin)))
        self.max_size = int(settings.unit * self.design.size_ratio)
        self.base = settings.caption_fontsize or self.max_size
        self.min_size = max(12, int(settings.unit * self.design.min_size_ratio))
        self.spacing = self.design.letter_spacing
        self.joiner = "" if settings.language.split("-")[0] in NO_SPACE_LANGS else " "
        self.joiner_w = self._tracked(self.joiner, load_font(self.font_path, 100, self.design.weight),
                                      100) if self.joiner else 0.0
        self.labels = labels or {}   # speaker -> name tag (dialogue)
        self._styles = {"A": style, "B": replace(style, active=SPEAKER_B["active"] if style.active else None,
                                                 box=SPEAKER_B["box"] if style.box else None,
                                                 spoken=SPEAKER_B["active"] if style.spoken else None)}

    # -- typography helpers ---------------------------------------------------------------------
    def _tracked(self, text: str, font: ImageFont.FreeTypeFont, size: int) -> float:
        """Width of a string with the design's letter spacing applied (last glyph gets none)."""
        width = font.getlength(text)
        if not text or self.spacing <= 0:
            return width
        return width + self.spacing * size * max(0, len(text) - 1)

    def word_width(self, word: str, font: ImageFont.FreeTypeFont, size: int) -> float:
        return self._tracked(word, font, size)

    def emphasis_color(self, style: CapStyle) -> str:
        return self.design.emphasis or style.active or self.design.signal

    def _scrim_alpha(self, size: int) -> int:
        """0 when no backing should be drawn, else the alpha of the charcoal scrim."""
        mode = self.design.contrast
        if mode == "never":
            return 0
        return max(0, min(200, self.design.scrim_alpha))

    def _layout(self, words: list[str], size: int, emphasis: set[int] | None = None):
        """Wrap a caption, reserving the enlarged emphasis word's own width so nothing collides.

        Wrapping is greedy per line but considers both the normal word widths and the width the word
        will actually occupy when emphasised, so an emphasised word never overlaps its neighbour and a
        long word is placed on a line it fits on rather than overflowing the frame.
        """
        emphasis = emphasis if emphasis is not None else set()
        font = load_font(self.font_path, size, self.design.weight)
        big = load_font(self.font_path, int(size * self.design.emphasis_scale), self.design.weight)
        sw = int(round(size * self.style.stroke_ratio * self.design.stroke_scale))
        space = self.joiner_w / 100.0 * size if self.joiner else 0.0
        widths = [self.word_width(w, font, size) + sw for w in words]
        # The room an emphasised word reserves (its enlarged width). Lines are measured with this slot
        # so a caption that only fits because its emphasis word was measured small is never produced.
        reserve = [(self.word_width(w, big, int(size * self.design.emphasis_scale)) + sw)
                   if i in emphasis else widths[i] for i, w in enumerate(words)]
        lines, cur, cur_w = [], [], 0.0
        for i, _ in enumerate(words):
            add = reserve[i] + (space if cur else 0)
            if cur and cur_w + add > self.max_w:
                lines.append((cur, cur_w))
                cur, cur_w = [], 0.0
                add = widths[i]
            cur.append(i)
            cur_w += add
        if cur:
            lines.append((cur, cur_w))
        return font, sw, space, widths, reserve, lines

    def _balance(self, lines: list[tuple[list[int], float]], advance: list[float], space: float,
                 tolerance: float = 0.14) -> list[tuple[list[int], float]]:
        """Pull the last word of a line down when it fits on the next line, for even line lengths.

        Balanced wrapping avoids the one long line followed by a single orphaned word, and never
        exceeds the safe width, so multi-line captions stay even and inside the margins.
        """
        if len(lines) < 2:
            return lines
        out = [(list(idxs), w) for idxs, w in lines]
        for i in range(len(out) - 2, -1, -1):
            idxs, w = out[i]
            nxt, nxt_w = out[i + 1]
            if len(idxs) <= 1:
                continue
            moved = idxs[-1]
            new_w = w - advance[moved] - space
            add_w = nxt_w + advance[moved] + space
            if add_w <= self.max_w and new_w >= advance[idxs[0]] and abs(new_w - add_w) < abs(w - nxt_w) - \
                    tolerance * self.max_w:
                out[i] = (idxs[:-1], new_w)
                out[i + 1] = ([moved, *nxt], add_w)
        return out

    def fit_size(self, words: list[str], emphasis: set[int] | None = None) -> int:
        """The biggest size at which the caption fits ``max_lines`` balanced lines inside the margins."""
        size = self.base
        while size > self.min_size:
            _, sw, _, widths, _, lines = self._layout(words, size, emphasis)
            if len(lines) <= self.design.max_lines and max(widths) <= self.max_w:
                break
            size = max(self.min_size, int(size * 0.9))
        return size

    def _tracked_text(self, target: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
                      font: ImageFont.FreeTypeFont, size: int, fill, stroke_width: int = 0,
                      stroke_fill=None) -> None:
        """Draw a word with the design's letter spacing, glyph by glyph (Pillow has no tracking)."""
        x, baseline = xy
        if not text:
            return
        if self.spacing <= 0:
            target.text((x, baseline), text, font=font, anchor="ls", fill=fill,
                        stroke_width=stroke_width, stroke_fill=stroke_fill)
            return
        step = self.spacing * size
        for ch in text:
            target.text((x, baseline), ch, font=font, anchor="ls", fill=fill,
                        stroke_width=stroke_width, stroke_fill=stroke_fill)
            x += font.getlength(ch) + step

    def _line_x(self, line_w: float) -> float:
        if self.design.align == "left":
            return (self.W - self.max_w) / 2
        if self.design.align == "right":
            return (self.W + self.max_w) / 2 - line_w
        return (self.W - line_w) / 2

    def _draw_scrim(self, img: Image.Image, lines, advance, space: float, size: int, y: float,
                    line_h: int, gap: int, alpha: int) -> None:
        """A soft charcoal plate behind the words: only drawn when it is needed for readability."""
        pad_x, pad_y = int(size * self.design.scrim_pad), int(size * self.design.scrim_pad * 0.8)
        top = max(0.0, y - pad_y)
        bottom = min(float(self.band_h), y + len(lines) * line_h + (len(lines) - 1) * gap + pad_y)
        left, right = float(self.W), 0.0
        for idxs, _ in lines:
            line_w = sum(advance[i] for i in idxs) + space * max(0, len(idxs) - 1)
            left = min(left, (self.W - line_w) / 2)
            right = max(right, (self.W + line_w) / 2)
        left, right = max(0.0, left - pad_x), min(float(self.W), right + pad_x)
        plate = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(plate).rounded_rectangle((left, top, right, bottom),
                                               radius=int(size * self.design.box_radius * 1.2),
                                               fill=_rgba(self.design.scrim, alpha))
        img.alpha_composite(plate)

    def render(self, words: list[str], active: int | None, size: int, scale: float = 1.0,
               speaker: str = "A", emphasis: set[int] | None = None) -> Image.Image:
        """One caption state: warm-white words, the emphasised words in the signal colour and a
        little larger and brighter, an optional charcoal scrim behind, all inside the safe area."""
        st = self._styles.get(speaker, self.style)
        emphasis = emphasis or set()
        size = max(10, int(size * scale))
        font, sw, space, widths, reserve, lines = self._layout(words, size, emphasis)
        # An emphasised word owns a wider slot than its plain width, and (when it is also the spoken
        # word) a yet wider one again. The advance, the line width and the centring all have to use the
        # *same* slot, or the enlarged word slides into its neighbour (the classic "THEOCTOPUS" bug).
        advance = [(reserve[i] if (i in emphasis or i == active) else widths[i]) for i in range(len(words))]
        lines = self._balance(lines, advance, space)
        ascent, descent = font.getmetrics()
        line_h = int(size * self.design.line_height)
        gap = int(size * self.design.line_gap)
        total_h = len(lines) * line_h + (len(lines) - 1) * gap
        y = (self.band_h - total_h) / 2

        img = Image.new("RGBA", (self.W, self.band_h), (0, 0, 0, 0))
        label = self.labels.get(speaker)
        if label:
            self._name_tag(img, label, SPEAKER_B["tag"] if speaker == "B" else SPEAKER_A_TAG, y)
        scrim = self._scrim_alpha(size)
        if scrim and lines:
            self._draw_scrim(img, lines, advance, space, size, y, line_h, gap, scrim)

        em_scale = self.design.emphasis_scale * scale
        big = load_font(self.font_path, int(size * em_scale), self.design.weight) \
            if em_scale != 1.0 else font
        big_sw = int(round(big.size * st.stroke_ratio * self.design.stroke_scale))
        em_color = self.emphasis_color(st)
        fx = Image.new("RGBA", img.size, (0, 0, 0, 0)) if (st.shadow or st.glow) else None
        draw, fxd = ImageDraw.Draw(img), ImageDraw.Draw(fx) if fx else None
        for idxs, _line_w in lines:
            line_w = sum(advance[i] for i in idxs) + space * max(0, len(idxs) - 1)
            x = self._line_x(line_w)
            baseline = y + (line_h + ascent - descent) / 2
            for i in idxs:
                word = words[i]
                is_em = i in emphasis
                is_active = active is not None and i == active
                enlarged = is_em or is_active
                f = big if enlarged else font
                s_w = big_sw if enlarged else sw
                if st.box and enlarged:
                    s_w = 0  # a boxed word reads cleaner without an outline
                wx = x + (advance[i] - f.getlength(word)) / 2
                color = st.fill or self.design.ink
                if is_em:
                    color = _brighten(em_color, self.design.emphasis_brightness)
                elif is_active and st.active:
                    color = st.active
                elif st.spoken and active is not None and i < active:
                    color = st.spoken
                if enlarged and st.box:
                    pad_x, pad_y = size * 0.16, size * 0.12
                    x0, y0, x1, y1 = draw.textbbox((wx, baseline), word, font=f, anchor="ls")
                    draw.rounded_rectangle((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y),
                                           radius=int(size * 0.2), fill=_rgba(st.box))
                if fxd is not None and not (enlarged and st.box):
                    if st.glow:
                        fxd.text((wx, baseline), word, font=f, anchor="ls", fill=_rgba(st.glow),
                                 stroke_width=max(2, s_w * 3), stroke_fill=_rgba(st.glow))
                    else:
                        off = max(2, int(size * self.design.shadow_offset))
                        fxd.text((wx + off, baseline + off), word, font=f, anchor="ls",
                                 fill=(0, 0, 0, 170), stroke_width=s_w, stroke_fill=(0, 0, 0, 170))
                self._tracked_text(draw, (wx, baseline), word, f, size, _rgba(color),
                                   stroke_width=s_w, stroke_fill=_rgba(st.stroke))
                x += advance[i] + space
            y += line_h + gap

        if fx is not None:
            fx = fx.filter(ImageFilter.GaussianBlur(radius=size * (0.12 if st.glow else 0.05)))
            if st.glow:
                fx = Image.alpha_composite(fx, fx)
            img = Image.alpha_composite(fx, img)
        return img

    def _name_tag(self, img: Image.Image, text: str, color: str, text_top: float) -> None:
        size = int(self.s.unit * 0.04)
        font = load_font(self.font_path, size)
        pad_x, pad_y = int(size * 0.55), int(size * 0.28)
        tw = int(font.getlength(text))
        ascent, descent = font.getmetrics()
        h = ascent + descent + 2 * pad_y
        y1 = int(max(0, text_top - size * 0.35))
        y0 = max(0, y1 - h)
        x0 = (self.W - tw) // 2 - pad_x
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((x0, y0, x0 + tw + 2 * pad_x, y0 + h), radius=h // 2, fill=_rgba(color, 235))
        d.text((x0 + pad_x, y0 + pad_y + ascent), text, font=font, anchor="ls", fill=(10, 10, 14, 255))

    def _exit_frame(self, img: Image.Image, progress: float, mode: str, seed: int, size: int) -> Image.Image:
        """A deterministic, seek-safe exit frame. Dust is drawn behind the words, then the whole band
        fades so the caption never hard-cuts against the footage."""
        frame = img.copy()
        if mode == "dust":
            # The caption genuinely comes apart: the words are progressively eroded into grain, and
            # the grain itself drifts up and out. Deterministic (seeded), so any seek re-renders the
            # same frame and ffmpeg can freely cut the overlay stream.
            rng = random.Random(seed)
            eroded = frame.copy()
            pa = eroded.getchannel("A")
            step = max(2, int(size * (0.05 + 0.30 * progress)))
            bbox = pa.getbbox()
            if bbox:  # punch dissolving gaps through the glyphs themselves
                holes = ImageDraw.Draw(eroded)
                for _ in range(max(6, self.design.dust_particles * 3)):
                    hx = rng.uniform(bbox[0], bbox[2])
                    hy = rng.uniform(bbox[1], bbox[3])
                    hr = max(1, int(step * rng.uniform(0.35, 1.15)))
                    holes.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=(0, 0, 0, 0))
            frame = eroded
            dust = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(dust)
            for _ in range(self.design.dust_particles):
                x0 = rng.uniform(0.08, 0.92) * self.W
                y0 = rng.uniform(0.30, 0.86) * self.band_h
                drift_x = rng.uniform(-0.14, 0.14) * size * progress
                drift_y = rng.uniform(-0.44, -0.12) * size * progress
                r = max(1, int(size * rng.uniform(0.012, 0.034) * (1.0 - 0.35 * progress)))
                color = self.design.signal if rng.random() < 0.32 else self.design.ink
                alpha = int(rng.uniform(90, 185) * (1.0 - progress) ** 1.3)
                d.ellipse((x0 + drift_x - r, y0 + drift_y - r, x0 + drift_x + r, y0 + drift_y + r),
                          fill=_rgba(color, max(0, alpha)))
            frame = Image.alpha_composite(frame, dust)
        if mode == "dissolve" and progress > 0:
            frame = frame.filter(ImageFilter.GaussianBlur(radius=max(0.6, size * 0.018 * progress)))
        keep = max(0.0, (1.0 - progress) ** (1.5 if mode == "dust" else 1.1))
        alpha = frame.getchannel("A").point(lambda a: int(a * keep))
        frame.putalpha(alpha)
        return frame

    def build(self, chunks: list[Chunk], total: float, out_dir: Path) -> Path:
        """Render every caption state and write an ffconcat playlist spanning ``total`` seconds."""
        out_dir.mkdir(parents=True, exist_ok=True)
        fps = self.s.fps
        frame = 1.0 / fps
        blank = out_dir / "blank.png"
        Image.new("RGBA", (self.W, self.band_h), (0, 0, 0, 0)).save(blank)
        entries: list[tuple[str, float]] = []
        t = 0.0
        n = 0

        def add(name: str, until: float):
            nonlocal t
            if until - t > 0.001:
                entries.append((name, until - t))
                t = until

        mode = (self.s.caption_exit or "auto").lower()
        if mode not in EXIT_MODES:
            mode = "auto"
        if mode == "auto":
            mode = "dust" if self.s.caption_style in {"kinetic", "bold"} else "none"
        if self.design.reduced and mode == "dust":
            mode = "dissolve"
        exit_sec = self.design.exit_seconds if mode != "none" else 0.0
        stagger = self.design.transition_stagger if mode != "none" else 0.0

        for ci, chunk in enumerate(chunks):
            words = [display_word(w.text, self.s.caption_uppercase) for w in chunk.words]
            if not any(words):
                continue
            em = set(emphasis_rules.emphasised(words, self.design.emphasis_mode, self.design.emphasis_max))
            size = self.fit_size(words, em)
            add(blank.name, chunk.start)
            states = list(range(len(words))) if self.style.active or self.style.spoken else [None]
            last_k = states[-1] if states else None
            # Keep the exit inside the last word's time slice, so a short final word stays readable.
            last_word_start = chunk.words[-1].start if chunk.words else chunk.start
            readable = max(0.0, chunk.end - last_word_start)
            chunk_exit = min(exit_sec, readable * 0.5)
            # In dust and dissolve modes the exit deliberately runs a little past the chunk's end so it
            # overlaps the *start* of the next caption: the old text is already breaking up as the new
            # one arrives, so the two never sit on top of each other at full opacity.
            next_start = chunks[ci + 1].start if ci + 1 < len(chunks) else total
            overlap = min(stagger * chunk_exit, max(0.0, next_start - chunk.end))
            frames = max(2, min(5, int(chunk_exit * fps))) if chunk_exit >= 2 / fps else 0
            exit_start = chunk.end - chunk_exit if frames else chunk.end
            if frames and exit_start - last_word_start < 0.05:
                frames, exit_start = 0, chunk.end
            for k in states:
                if k is None:
                    end = exit_start
                else:
                    end = chunk.words[k + 1].start if k + 1 < len(words) else exit_start
                    end = min(max(end, t), exit_start)
                if k in (0, None) and self.style.pop:
                    for sc in self.design.entrance_scales:
                        n += 1
                        name = f"c{ci:04d}_pop{n:05d}.png"
                        self.render(words, k, size, sc, chunk.speaker, em).save(out_dir / name, compress_level=1)
                        add(name, min(t + frame, end))
                n += 1
                name = f"c{ci:04d}_w{n:05d}.png"
                self.render(words, k, size, speaker=chunk.speaker, emphasis=em).save(out_dir / name, compress_level=1)
                add(name, end)
            if frames:
                seed = ci * 1009 + len(words) * 17 + int(chunk.start * 1000)
                for e in range(1, frames + 1):
                    progress = e / frames
                    n += 1
                    name = f"c{ci:04d}_out{n:05d}.png"
                    base = self.render(words, last_k, size, speaker=chunk.speaker, emphasis=em)
                    self._exit_frame(base, progress, mode, seed, size).save(out_dir / name, compress_level=1)
                    add(name, min(exit_start + (chunk_exit + overlap) * progress, total))
        add(blank.name, total)
        if not entries:
            entries.append((blank.name, total))
        lines = ["ffconcat version 1.0"]
        for name, dur in entries:
            lines += [f"file '{name}'", f"duration {dur:.4f}"]
        lines.append(f"file '{entries[-1][0]}'")  # concat demuxer quirk: repeat the last file
        playlist = out_dir / "captions.ffconcat"
        playlist.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Captions: %d states rendered (%s style)", n, self.s.caption_style)
        return playlist


# ------------------------------------------------------------------------------------------ static PNG overlays
def _text_box(settings: Settings, font_path: Path | None, text: str, *, size_ratio: float, fill: str,
              box: str | None, box_alpha: int, max_lines: int = 3, uppercase: bool = True) -> Image.Image:
    text = text.upper() if uppercase else text
    size = int(settings.unit * size_ratio)
    max_w = int(min(settings.width * 0.80, settings.unit * 1.2))
    for _ in range(10):
        font = load_font(str(font_path) if font_path else None, size)
        words, lines, cur = text.split(), [], ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if font.getlength(cand) <= max_w or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        if len(lines) <= max_lines and all(font.getlength(line) <= max_w for line in lines):
            break
        size = int(size * 0.9)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    pad = int(size * 0.45)
    sw = max(1, int(size * 0.06)) if not box else 0
    tw = int(max(font.getlength(line) for line in lines)) + 2 * pad + 2 * sw
    th = line_h * len(lines) + 2 * pad
    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if box:
        d.rounded_rectangle((0, 0, tw - 1, th - 1), radius=int(size * 0.35), fill=_rgba(box, box_alpha))
    y = pad
    for line in lines:
        d.text((tw / 2, y + ascent), line, font=font, anchor="ms", fill=_rgba(fill),
               stroke_width=sw, stroke_fill=(0, 0, 0, 255))
        y += line_h
    return img


def hook_png(settings: Settings, font_path: Path | None, text: str, dest: Path,
             y_ratio: float = 0.15, max_lines: int = 3) -> tuple[Path, int]:
    img = _text_box(settings, font_path, text, size_ratio=0.075, fill="#FFFFFF", box="#000000", box_alpha=175,
                    max_lines=max_lines)
    img.save(dest)
    return dest, int(settings.height * y_ratio)


def cta_png(settings: Settings, font_path: Path | None, text: str, dest: Path,
            y_ratio: float = 0.30) -> tuple[Path, int]:
    img = _text_box(settings, font_path, text, size_ratio=0.062, fill="#FFFFFF", box="#FF0033", box_alpha=235,
                    max_lines=2)
    img.save(dest)
    return dest, int(settings.height * y_ratio)


def watermark_png(settings: Settings, font_path: Path | None, text: str, dest: Path) -> tuple[Path, int]:
    font = load_font(str(font_path) if font_path else None, int(settings.unit * 0.034))
    sw = 2
    w = int(font.getlength(text)) + 8 + 2 * sw
    ascent, descent = font.getmetrics()
    img = Image.new("RGBA", (w, ascent + descent + 8), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((4 + sw, 4 + ascent), text, font=font, anchor="ls", fill=(255, 255, 255, 150),
                             stroke_width=sw, stroke_fill=(0, 0, 0, 110))
    img.save(dest)
    return dest, int(settings.height * 0.045)


# ------------------------------------------------------------------------------------------ libass
def _ass_color(hex_color: str, alpha: int = 0) -> str:
    r, g, b, _ = _rgba(hex_color)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(t: str) -> str:
    return t.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def write_ass(settings: Settings, style: CapStyle, family: str, chunks: list[Chunk], total: float, dest: Path, *,
              hook: str = "", hook_until: float = 0.0, cta: str = "", cta_from: float = 0.0,
              watermark: str = "", hook_y: float = 0.15, cta_y: float = 0.30,
              labels: dict[str, str] | None = None, design: CaptionDesign | None = None) -> Path:
    """The libass twin of :class:`PillowCaptions`, for scripts Pillow cannot shape (raqm missing).

    It reads the same :class:`CaptionDesign`, so colours, safe margins, letter spacing and the exit
    behaviour match the Pillow renderer. libass cannot shatter glyphs, so ``dust`` becomes a fade of
    the last state — the same visual beat, minus the particles.
    """
    W, H = settings.resolution
    U = settings.unit
    d = design or design_from_settings(settings, style)
    size = settings.caption_fontsize or int(U * d.size_ratio)
    outline = max(2, int(size * (style.stroke_ratio or 0.06) * d.stroke_scale))
    margin = int(W * d.safe_margin)
    spacing = int(round(d.letter_spacing * size))
    align = {"left": 4, "center": 5, "right": 6}.get(d.align, 5)  # ASS numpad alignment (bottom row)
    cy = int(H * CENTER_Y.get(settings.caption_position, CENTER_Y["center"]))
    joiner = "" if settings.language.split("-")[0] in NO_SPACE_LANGS else " "
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}",
        "ScaledBorderAndShadow: yes", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Cap,{family},{size},{_ass_color(style.fill or d.ink)},{_ass_color(style.active or style.fill)},"
        f"{_ass_color(style.glow or style.stroke)},&H{max(0, 255 - d.scrim_alpha):02X}{d.scrim.lstrip('#')[4:6]}"
        f"{d.scrim.lstrip('#')[2:4]}{d.scrim.lstrip('#')[0:2]},-1,0,0,0,100,100,{spacing},0,1,{outline},"
        f"{2 if style.shadow else 0},{align},{margin},{margin},0,1",
        f"Style: Hook,{family},{int(U * 0.07)},&H00FFFFFF,&H00FFFFFF,&H50000000,&H50000000,-1,0,0,0,100,100,0,0,3,"
        f"{int(U * 0.02)},0,8,60,60,{int(H * hook_y)},1",
        f"Style: Cta,{family},{int(U * 0.058)},&H00FFFFFF,&H00FFFFFF,&H003300FF,&H003300FF,-1,0,0,0,100,100,0,0,3,"
        f"{int(U * 0.022)},0,8,60,60,{int(H * cta_y)},1",
        f"Style: Mark,{family},{int(U * 0.032)},&H66FFFFFF,&H66FFFFFF,&H90000000,&H00000000,-1,0,0,0,100,100,0,0,1,"
        f"2,0,8,20,20,{int(H * 0.045)},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    big = int(style.active_scale * 100)
    em_big = int(d.emphasis_scale * 100)
    labels = labels or {}
    exit_mode = (settings.caption_exit or "auto").lower()
    if exit_mode not in EXIT_MODES:
        exit_mode = "auto"
    if exit_mode == "auto":
        exit_mode = "dissolve" if settings.caption_style in {"kinetic", "bold"} else "none"
    if d.reduced and exit_mode == "dust":
        exit_mode = "dissolve"                       # libass cannot shatter glyphs; fade instead
    exit_ms = int(d.exit_seconds * 1000) if exit_mode != "none" else 0
    em_ms = int(exit_ms * (1.0 + d.transition_stagger))  # the steeper fade covers the next caption in
    for chunk in chunks:
        b = chunk.speaker == "B"
        active_c = _ass_color(SPEAKER_B["active"] if b else style.active) if style.active else None
        spoken_c = _ass_color(SPEAKER_B["active"] if b else style.spoken) if style.spoken else None
        em_c = _ass_color(d.emphasis or style.active or d.signal)
        tag = labels.get(chunk.speaker)
        if tag:
            tag_c = _ass_color(SPEAKER_B["tag"] if b else SPEAKER_A_TAG)
            lines.append(f"Dialogue: 1,{_ass_time(chunk.start)},{_ass_time(chunk.end)},Cap,,0,0,0,,"
                         f"{{\\pos({W // 2},{int(cy - size * 1.3)})\\fs{int(U * 0.036)}\\c{tag_c}}}{_ass_escape(tag)}")
        words = [_ass_escape(display_word(w.text, settings.caption_uppercase)) for w in chunk.words]
        em = set(emphasis_rules.emphasised(words, d.emphasis_mode, d.emphasis_max))
        states = list(range(len(words))) if (style.active or style.spoken) else [None]
        last_k = states[-1] if states else None
        for k in states:
            start = chunk.start if k in (0, None) else chunk.words[k].start
            end = chunk.end if k is None or k + 1 >= len(words) else chunk.words[k + 1].start
            if end - start < 0.01:
                continue
            parts = []
            for i, w in enumerate(words):
                if i in em:
                    parts.append(f"{{\\c{em_c}\\fscx{em_big}\\fscy{em_big}}}{w}{{\\r}}")
                elif k is not None and i == k and active_c:
                    parts.append(f"{{\\c{active_c}\\fscx{big}\\fscy{big}}}{w}{{\\r}}")
                elif spoken_c and k is not None and i < k:
                    parts.append(f"{{\\c{spoken_c}}}{w}{{\\r}}")
                else:
                    parts.append(w)
            pop = ("{\\fscx78\\fscy78\\t(0,70,\\fscx107\\fscy107)\\t(70,130,\\fscx100\\fscy100)}"
                   if style.pop and not d.reduced and k in (0, None) else "")
            blur = "{\\blur6}" if style.glow else ""
            fade = f"{{\\fad(0,{em_ms})}}" if exit_ms and k == last_k else ""
            lines.append(f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Cap,,0,0,0,,"
                         f"{{\\pos({W // 2},{cy})}}{blur}{pop}{fade}{joiner.join(parts)}")
    if hook:
        lines.append(f"Dialogue: 2,{_ass_time(0)},{_ass_time(hook_until)},Hook,,0,0,0,,{{\\fad(150,300)}}"
                     f"{_ass_escape(hook.upper())}")
    if cta:
        lines.append(f"Dialogue: 2,{_ass_time(cta_from)},{_ass_time(total)},Cta,,0,0,0,,{{\\fad(250,0)}}"
                     f"{_ass_escape(cta)}")
    if watermark:
        lines.append(f"Dialogue: 0,{_ass_time(0)},{_ass_time(total)},Mark,,0,0,0,,{_ass_escape(watermark)}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


# ------------------------------------------------------------------------------------------ chat stories
@dataclass
class ChatMessage:
    speaker: str        # A = the contact (left), B = "Me" (right)
    text: str
    start: float


class ChatScreen:
    """A phone-style text conversation where each message pops in as it is read aloud, with a typing
    indicator before the contact's replies. Played back like the captions: PNG states + ffconcat."""

    def __init__(self, settings: Settings, font_path: Path | None, contact: str):
        self.s = settings
        W, H, U = settings.width, settings.height, settings.unit
        self.pw = int(min(W * 0.88, U * 0.92)) // 2 * 2
        self.x = (W - self.pw) // 2
        self.y = int(H * 0.25)
        self.ph = int(H * 0.66) // 2 * 2
        self.font_path = str(font_path) if font_path else None
        self.size = int(U * 0.043)
        self.font = load_font(self.font_path, self.size, weight=500)
        self.pad = int(U * 0.03)
        self.header_h = int(U * 0.13)
        self.contact = contact or "Unknown"
        self.max_bubble = int(self.pw * 0.74)
        self.line_h = sum(self.font.getmetrics()) + int(self.size * 0.12)
        self.gap = int(self.size * 0.45)

    def _wrap(self, text: str) -> list[str]:
        lines, cur = [], ""
        inner = self.max_bubble - 2 * int(self.size * 0.7)
        for w in text.split():
            cand = f"{cur} {w}".strip()
            if not cur or self.font.getlength(cand) <= inner:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    def _bubble_h(self, msg: ChatMessage | None) -> int:
        n = len(self._wrap(msg.text)) if msg else 1
        return n * self.line_h + int(self.size * 0.9)

    def _base(self) -> Image.Image:
        img = Image.new("RGBA", (self.pw, self.ph), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        r = int(self.s.unit * 0.05)
        d.rounded_rectangle((0, 0, self.pw - 1, self.ph - 1), radius=r, fill=(14, 15, 20, 238))
        d.rounded_rectangle((0, 0, self.pw - 1, self.header_h), radius=r, fill=(28, 29, 36, 255))
        d.rectangle((0, self.header_h - r, self.pw - 1, self.header_h), fill=(28, 29, 36, 255))
        # avatar with the contact's initial, then the name
        av = int(self.header_h * 0.58)
        ax, ay = self.pad, (self.header_h - av) // 2
        d.ellipse((ax, ay, ax + av, ay + av), fill=(99, 102, 241, 255))
        big = load_font(self.font_path, int(av * 0.5), weight=700)
        initial = (self.contact.strip()[:1] or "?").upper()
        d.text((ax + av / 2, ay + av / 2), initial, font=big, anchor="mm", fill=(255, 255, 255, 255))
        name_font = load_font(self.font_path, int(self.size * 1.05), weight=700)
        d.text((ax + av + self.pad * 0.8, self.header_h / 2), self.contact, font=name_font, anchor="lm",
               fill=(245, 246, 250, 255))
        d.line((0, self.header_h, self.pw, self.header_h), fill=(255, 255, 255, 28), width=2)
        return img

    def _draw_bubble(self, d: ImageDraw.ImageDraw, msg: ChatMessage | None, y: int, alpha: int = 255) -> None:
        pad_x, pad_y = int(self.size * 0.7), int(self.size * 0.45)
        mine = msg is not None and msg.speaker == "B"
        if msg is None:  # typing indicator
            w = int(self.size * 2.6)
            h = self._bubble_h(None)
            x0 = self.pad
            d.rounded_rectangle((x0, y, x0 + w, y + h), radius=int(self.size * 0.8), fill=(44, 44, 50, alpha))
            r = max(3, int(self.size * 0.14))
            for k in range(3):
                cx = x0 + w * (0.3 + 0.2 * k)
                d.ellipse((cx - r, y + h / 2 - r, cx + r, y + h / 2 + r), fill=(170, 170, 180, alpha))
            return
        lines = self._wrap(msg.text)
        tw = int(max(self.font.getlength(ln) for ln in lines))
        w, h = tw + 2 * pad_x, self._bubble_h(msg)
        x0 = self.pw - self.pad - w if mine else self.pad
        fill = (10, 132, 255, alpha) if mine else (44, 44, 50, alpha)
        d.rounded_rectangle((x0, y, x0 + w, y + h), radius=int(self.size * 0.8), fill=fill)
        ascent = self.font.getmetrics()[0]
        ty = y + pad_y
        for ln in lines:
            d.text((x0 + pad_x, ty + ascent), ln, font=self.font, anchor="ls", fill=(255, 255, 255, alpha))
            ty += self.line_h

    def render(self, shown: list[ChatMessage], typing: bool = False, enter: float = 0.0) -> Image.Image:
        """``enter`` (0..1) slides the newest bubble up into place."""
        img = self._base()
        d = ImageDraw.Draw(img)
        items: list[ChatMessage | None] = list(shown) + ([None] if typing else [])
        top, bottom = self.header_h + self.pad, self.ph - self.pad
        heights = [self._bubble_h(m) for m in items]
        # Newest messages stay visible: drop the oldest until the rest fit (the chat "scrolls").
        first = 0
        while first < len(items) - 1 and sum(heights[first:]) + self.gap * (len(items) - first - 1) > bottom - top:
            first += 1
        y = top
        for k in range(first, len(items)):
            newest = k == len(items) - 1
            dy = int(self.size * 0.8 * enter) if newest else 0
            alpha = int(255 * (1 - 0.5 * enter)) if newest else 255
            self._draw_bubble(d, items[k], y + dy, alpha)
            y += heights[k] + self.gap
        return img

    def build(self, messages: list[ChatMessage], total: float, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = 1.0 / self.s.fps
        entries: list[tuple[str, float]] = []
        t, n = 0.0, 0

        def add(img: Image.Image, until: float):
            nonlocal t, n
            if until - t > 0.001:
                n += 1
                name = f"chat{n:04d}.png"
                img.save(out_dir / name, compress_level=1)
                entries.append((name, until - t))
                t = until

        for k, msg in enumerate(messages):
            shown = messages[:k]
            if k and msg.speaker == "A":  # the contact is typing...
                typing_from = max(t, msg.start - 0.55)
                add(self.render(shown), typing_from)
                add(self.render(shown, typing=True), msg.start)
            else:
                add(self.render(shown), msg.start)
            end = messages[k + 1].start if k + 1 < len(messages) else total
            for e in (0.6, 0.25):  # pop-in
                add(self.render(messages[:k + 1], enter=e), min(t + frame, end))
        add(self.render(messages), total)
        if not entries:
            add(self.render(messages), max(total, frame))
        lines = ["ffconcat version 1.0"]
        for name, dur in entries:
            lines += [f"file '{name}'", f"duration {dur:.4f}"]
        lines.append(f"file '{entries[-1][0]}'")
        playlist = out_dir / "chat.ffconcat"
        playlist.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Chat screen: %d messages, %d states", len(messages), n)
        return playlist


# ------------------------------------------------------------------------------------------ story post card
CARD_ACCENTS = ["#FF6B35", "#7C3AED", "#0EA5E9", "#10B981", "#F59E0B", "#EC4899"]


def _compact(n: int) -> str:
    return (f"{n / 1000:.1f}".rstrip("0").rstrip(".") + "k") if n >= 1000 else str(n)


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: int, by_char: bool = False) -> list[str]:
    tokens = list(text) if by_char else text.split()
    sep = "" if by_char else " "
    lines, cur = [], ""
    for tok in tokens:
        cand = f"{cur}{sep}{tok}" if cur else tok
        if not cur or font.getlength(cand) <= width:
            cur = cand
        else:
            lines.append(cur)
            cur = tok.lstrip() if by_char else tok
    if cur:
        lines.append(cur)
    return lines or [""]


def post_card_png(settings: Settings, font_path: Path | None, community: str, user: str, title: str,
                  dest: Path) -> tuple[Path, int, int]:
    """A forum-post card (community, poster, title, votes, comments) shown while the title is read aloud,
    the way Reddit story videos open. The numbers are decoration, derived from the title so a re-render
    draws the same card. Returns (png, x, y)."""
    W, H, U = settings.width, settings.height, settings.unit
    cw = int(min(W * 0.88, U * 0.95)) // 2 * 2
    pad = int(U * 0.045)
    fp = str(font_path) if font_path else None
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(title))
    accent = _rgba(CARD_ACCENTS[seed % len(CARD_ACCENTS)])
    votes, comments = 1200 + seed * 37 % 46000, 90 + seed * 13 % 3800
    ink, grey, pill = (26, 26, 27, 255), (112, 117, 122, 255), (234, 237, 239, 255)
    by_char = settings.language.split("-")[0] in NO_SPACE_LANGS

    size = int(U * 0.058)
    for _ in range(12):
        title_f = load_font(fp, size, weight=700)
        lines = _wrap(title, title_f, cw - 2 * pad, by_char)
        if len(lines) <= 6:
            break
        size = int(size * 0.9)
    ascent, descent = title_f.getmetrics()
    line_h = ascent + descent + int(size * 0.08)
    name_f = load_font(fp, int(U * 0.036), weight=700)
    meta_f = load_font(fp, int(U * 0.03), weight=500)
    small_f = load_font(fp, int(U * 0.031), weight=700)

    av = int(U * 0.075)
    title_top = pad + av + int(pad * 0.55)
    foot_top = title_top + line_h * len(lines) + int(pad * 0.45)
    foot_h = int(U * 0.062)
    ch = foot_top + foot_h + pad
    img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, cw - 1, ch - 1), radius=int(U * 0.035), fill=(255, 255, 255, 250),
                        outline=(0, 0, 0, 40), width=2)
    # header: avatar with the community's initial, community name, poster and age
    d.ellipse((pad, pad, pad + av, pad + av), fill=accent)
    core = community.split("/", 1)[-1]
    d.text((pad + av / 2, pad + av / 2), (core[:1] or "?").upper(),
           font=load_font(fp, int(av * 0.52), weight=800), anchor="mm", fill=(255, 255, 255, 255))
    tx = pad + av + int(pad * 0.45)
    d.text((tx, pad + av * 0.30), community, font=name_f, anchor="lm", fill=ink)
    d.text((tx, pad + av * 0.76), f"{user} · {3 + seed % 20}h", font=meta_f, anchor="lm", fill=grey)
    y = title_top
    for ln in lines:
        d.text((pad, y + ascent), ln, font=title_f, anchor="ls", fill=ink)
        y += line_h

    # footer pills: votes, comments, share
    x, r = pad, foot_h // 2
    mid = foot_top + foot_h / 2
    a = foot_h * 0.36                                              # arrow size

    def up(cx: float, cy: float, s: float, down: bool = False) -> None:
        k = -1 if down else 1
        d.polygon([(cx - s / 2, cy + k * s * 0.05), (cx, cy - k * s / 2), (cx + s / 2, cy + k * s * 0.05)], fill=ink)
        d.rectangle((cx - s * 0.17, min(cy, cy + k * s * 0.5), cx + s * 0.17, max(cy, cy + k * s * 0.5)), fill=ink)

    votes_txt = _compact(votes)
    w1 = int(a * 2 + small_f.getlength(votes_txt) + foot_h * 1.1)
    d.rounded_rectangle((x, foot_top, x + w1, foot_top + foot_h), radius=r, fill=pill)
    up(x + foot_h * 0.45, mid, a)
    d.text((x + foot_h * 0.45 + a * 0.9, mid), votes_txt, font=small_f, anchor="lm", fill=ink)
    up(x + w1 - foot_h * 0.45, mid, a, down=True)
    x += w1 + int(pad * 0.4)
    com_txt = _compact(comments)
    w2 = int(a * 1.4 + small_f.getlength(com_txt) + foot_h * 0.9)
    d.rounded_rectangle((x, foot_top, x + w2, foot_top + foot_h), radius=r, fill=pill)
    bx, bw, bh = x + foot_h * 0.35, a * 1.15, a * 0.85
    d.rounded_rectangle((bx, mid - bh / 2, bx + bw, mid + bh / 2), radius=int(bh * 0.35), outline=ink,
                        width=max(2, int(U * 0.004)))
    d.polygon([(bx + bw * 0.2, mid + bh / 2 - 1), (bx + bw * 0.2, mid + bh * 0.85), (bx + bw * 0.5, mid + bh / 2 - 1)],
              fill=ink)
    d.text((bx + bw + a * 0.45, mid), com_txt, font=small_f, anchor="lm", fill=ink)
    x += w2 + int(pad * 0.4)
    share = "Share"
    w3 = int(small_f.getlength(share) + foot_h * 0.9)
    if x + w3 <= cw - pad:
        d.rounded_rectangle((x, foot_top, x + w3, foot_top + foot_h), radius=r, fill=pill)
        d.text((x + w3 / 2, mid), share, font=small_f, anchor="mm", fill=ink)

    img.save(dest)
    y0 = int(min(max(H * 0.5 - ch / 2, H * 0.12), max(H * 0.12, H - ch - H * 0.12)))
    return dest, (W - cw) // 2, y0


# ------------------------------------------------------------------------------------------ subscribe end card
def _draw_bell(d: ImageDraw.ImageDraw, cx: int, cy: int, size: int, fill) -> None:
    """A small, restrained bell icon centered at (cx, cy). Drawn with primitives so
    it never depends on an emoji-capable font."""
    r = size * 0.28
    # bell bowl
    d.ellipse((cx - r, cy - r * 0.5, cx + r, cy + r * 0.5), outline=fill, width=max(2, int(size * 0.05)))
    # flared rim
    rim = size * 0.10
    d.arc((cx - r - rim, cy - r * 0.55, cx + r + rim, cy - r * 0.45), start=180, end=360, fill=fill,
          width=max(2, int(size * 0.05)))
    # top loop
    loop = size * 0.16
    d.arc((cx - loop, cy - r - loop, cx + loop, cy - r + loop * 0.4), start=0, end=180, fill=fill,
          width=max(2, int(size * 0.05)))
    # clapper
    d.line((cx, cy + r * 0.45, cx, cy + r * 0.9), fill=fill, width=max(2, int(size * 0.05)))


def _end_card_frame(
    settings: Settings,
    font_path: Path | None,
    work: Path,
    name: str,
    scale: float,
    alpha: int,
    glow: bool,
    bell: bool,
) -> Path:
    """One end-card state: the subscribe button at ``scale`` with ``alpha``, an
    optional restrained glow and an optional bell. Full frame so it composites
    cleanly over the footage."""
    W, H = settings.width, settings.height
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    bw = int(W * 0.70 * scale)
    bh = int(H * 0.14 * scale)
    bh = max(bh, 40)
    bx = (W - bw) // 2
    by = int(H * 0.58) - bh // 2
    radius = max(8, int(bh * 0.5))

    if glow:
        glow_pad = int(bh * 0.6)
        gx = bx - glow_pad
        gy = by - glow_pad
        gw = bw + 2 * glow_pad
        gh = bh + 2 * glow_pad
        glow_img = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow_img)
        gd.rounded_rectangle((0, 0, gw - 1, gh - 1), radius=radius + glow_pad,
                             fill=(255, 122, 24, 34))
        glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=max(4, int(bh * 0.25))))
        img.alpha_composite(glow_img, (gx, gy))

    # button body
    body = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body)
    bd.rounded_rectangle((0, 0, bw - 1, bh - 1), radius=radius,
                         fill=(11, 11, 15, alpha),
                         outline=(255, 255, 255, max(40, int(90 * alpha / 255))), width=max(2, int(bh * 0.03)))
    # inner restrained moving outline (a thinner accent stroke inside the button)
    bd.rounded_rectangle((3, 3, bw - 4, bh - 4), radius=max(2, radius - 3),
                         outline=(255, 122, 24, max(60, int(140 * alpha / 255))),
                         width=max(1, int(bh * 0.02)))

    # text
    font = load_font(font_path, max(20, int(bh * 0.55)), weight=800)
    tw = font.getlength(END_CARD_TEXT)
    ascent, descent = font.getmetrics()
    tx = (bw - tw) // 2
    ty = (bh - (ascent + descent)) // 2 + ascent
    bd.text((tx, ty), END_CARD_TEXT, font=font, anchor="ls",
            fill=(255, 255, 255, alpha), stroke_width=max(1, int(bh * 0.015)),
            stroke_fill=(0, 0, 0, int(120 * alpha / 255)))

    # bell
    if bell:
        bell_size = int(bh * 0.55)
        _draw_bell(bd, bw - int(bh * 0.35), bh // 2, bell_size,
                   (255, 255, 255, alpha))

    img.alpha_composite(body, (bx, by))
    dest = work / name
    img.save(dest, compress_level=1)
    return dest


def build_end_card(settings: Settings, font_path: Path | None, total: float, work: Path
                   ) -> tuple[Path, int, int, float, float] | None:
    """Build the reusable animated subscribe end card.

    Returns ``(playlist, x, y, start, duration)`` or ``None`` when it cannot be
    built. The card plays inside ``[start, start + duration]`` which lies within
    ``[0, total]`` so it never extends the video past its total duration.
    """
    try:
        from PIL import ImageFilter  # noqa: F401 (used in _end_card_frame)
    except Exception:
        return None
    work.mkdir(parents=True, exist_ok=True)
    dur = END_CARD_SECONDS
    start = max(0.0, total - dur)
    if start <= 0:
        return None

    # Key states: reveal (scale up), hold with pulse + bell, then fade out.
    states = [
        (0.00, 0.18, 0.55, 110, False, False),
        (0.18, 0.42, 1.00, 235, True, False),
        (0.42, 0.78, 1.00, 255, True, False),
        (0.78, 1.30, 1.06, 255, True, True),
        (1.30, 1.72, 1.00, 255, True, True),
        (1.72, 2.20, 1.00, 210, True, False),
        (2.20, 2.50, 1.00, 90, False, False),
    ]

    fps = settings.fps
    entries: list[tuple[str, float]] = []
    t = start
    for i, (s0, s1, scale, alpha, glow, bell) in enumerate(states):
        name = f"endcard{i:02d}.png"
        _end_card_frame(settings, font_path, work, name, scale, alpha, glow, bell)
        until = min(start + dur, start + s1)
        if until - t > 0.005:
            entries.append((name, until - t))
            t = until
    # Blank tail so the card fully leaves before the video ends.
    blank = work / "endcard_blank.png"
    Image.new("RGBA", (settings.width, settings.height), (0, 0, 0, 0)).save(blank, compress_level=1)
    if start + dur - t > 0.005:
        entries.append(("endcard_blank.png", start + dur - t))
    if not entries:
        return None

    lines = ["ffconcat version 1.0"]
    for name, length in entries:
        lines += [f"file '{name}'", f"duration {length:.4f}"]
    lines.append(f"file '{entries[-1][0]}'")
    playlist = work / "endcard.ffconcat"
    playlist.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return playlist, 0, 0, start, dur


# ------------------------------------------------------------------------------------------ plan
@dataclass
class OverlayPlan:
    mode: str                                  # pillow | ass
    captions: Path | None = None               # ffconcat playlist (pillow)
    captions_y: int = 0
    hook: tuple[Path, int] | None = None       # (png, y)
    hook_until: float = 0.0
    watermark: tuple[Path, int] | None = None
    cta: tuple[Path, int] | None = None
    cta_from: float = 0.0
    ass: Path | None = None                    # subtitles file (ass)
    fonts_dir: Path | None = None
    chat: tuple[Path, int, int] | None = None  # (ffconcat playlist, x, y) for chat stories
    card: tuple[Path, int, int] | None = None  # (png, x, y) post card for Reddit-style stories
    card_until: float = 0.0
    end_card: tuple[Path, int, int, float, float] | None = None
    # (playlist, x, y, start, duration) animated subscribe end card inside the total duration


def build_overlays(settings: Settings, chunks: list[Chunk], total: float, hook_text: str, work: Path, *,
                   labels: dict[str, str] | None = None, chat: list[ChatMessage] | None = None,
                   contact: str = "", card: tuple[str, str, str] | None = None,
                   card_until: float = 0.0) -> OverlayPlan:
    """labels: speaker name tags over the captions (dialogue). chat: render a text-message screen
    instead of captions (the messages are the captions). card: (community, poster, title) of a post card
    shown until ``card_until`` in place of the hook title (pass captions that start after it)."""
    global CURRENT_DESIGN
    style = CAPTION_STYLES.get(settings.caption_style, CAPTION_STYLES["bold"])
    if not settings.caption_uppercase and style.font == "anton":
        style = replace(style, font="sans")
    # One design for the whole render: every caption of a video, and of every video using this .env,
    # is drawn with the same colours, weights, metrics and margins.
    design = design_from_settings(settings, style)
    CURRENT_DESIGN = design
    style_font = design.font if design.font in FONT_SOURCES else style.font
    font_path, family = resolve_font(settings, style_font, bold=design.bold or style_font == "anton")
    card = card if card and card_until > 0 else None
    hook_until = min(2.8, total * 0.4) if settings.hook_overlay and hook_text and not card else 0.0
    cta_text = settings.end_cta.strip()
    cta_from = max(total - 2.6, total * 0.6) if cta_text else 0.0
    hook_y = 0.09 if chat else 0.15
    chat_plan = None
    if chat:
        if settings.language.split("-")[0] in COMPLEX_LANGS and not features.check("raqm"):
            log.warning("This Pillow build can't shape %s text; chat bubbles may look wrong", settings.language)
        sans, _ = resolve_font(settings, "sans")
        screen = ChatScreen(settings, sans, contact)
        chat_plan = (screen.build(chat, total, work / "chat"), screen.x, screen.y)
        chunks = []
    card_plan = None
    if card:
        if settings.language.split("-")[0] in COMPLEX_LANGS and not features.check("raqm"):
            log.warning("This Pillow build can't shape %s text; the post card may look wrong", settings.language)
        sans, _ = resolve_font(settings, "sans")
        card_plan = post_card_png(settings, sans, *card, work / "card.png")

    if needs_ass(settings) and ffmpeg.has_filter("ass"):
        fonts = work / "fonts"
        fonts.mkdir(parents=True, exist_ok=True)
        if font_path:
            shutil.copy(font_path, fonts / font_path.name)
        ass = write_ass(settings, style, family, chunks, total, work / "captions.ass",
                        hook=hook_text if hook_until else "", hook_until=hook_until,
                        cta=cta_text, cta_from=cta_from, watermark=settings.watermark_text,
                        hook_y=hook_y, cta_y=0.10 if chat else 0.30, labels=labels, design=design)
        log.info("Captions: libass renderer (%s, font %s)", settings.language, family)
        return OverlayPlan("ass", ass=ass, fonts_dir=fonts, chat=chat_plan, card=card_plan,
                           card_until=card_until if card_plan else 0.0)

    plan = OverlayPlan("pillow", chat=chat_plan, card=card_plan, card_until=card_until if card_plan else 0.0)
    if chunks:
        caps = PillowCaptions(settings, style, font_path, labels, design=design)
        plan.captions, plan.captions_y = caps.build(chunks, total, work / "captions"), caps.band_y
    ui_font, _ = resolve_font(settings, "anton" if settings.caption_uppercase else "sans")
    if hook_until:
        plan.hook = hook_png(settings, ui_font, hook_text, work / "hook.png", y_ratio=hook_y,
                             max_lines=2 if chat else 3)
        plan.hook_until = hook_until
    if settings.watermark_text:
        plan.watermark = watermark_png(settings, ui_font, settings.watermark_text, work / "watermark.png")
    if cta_text:
        # Chat stories keep the conversation readable: the call-to-action takes the hook's place above it.
        plan.cta = cta_png(settings, ui_font, cta_text, work / "cta.png", y_ratio=0.10 if chat else 0.30)
        plan.cta_from = cta_from
    # Reusable animated subscribe end card, inside the total duration (never after it).
    ec = build_end_card(settings, font_path, total, work)
    if ec is not None:
        plan.end_card = ec
    return plan
