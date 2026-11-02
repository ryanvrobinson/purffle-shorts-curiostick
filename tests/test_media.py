import os
from pathlib import Path
from io import BytesIO

import pytest
from PIL import Image

from purffle_shorts import media as M
from purffle_shorts.config import Settings


class R:
    def __init__(self, data):
        self.status_code, self._d, self.text = 200, data, ""

    def json(self):
        return self._d


class S:
    def __init__(self, data):
        self.data, self.params = data, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.params.append(params)
        return R(self.data)


def test_pexels_file_choice_prefers_1080_not_4k():
    files = [
        {"link": "u4k", "width": 2160, "height": 3840, "file_type": "video/mp4"},
        {"link": "u1080", "width": 1080, "height": 1920, "file_type": "video/mp4"},
        {"link": "u720", "width": 720, "height": 1280, "file_type": "video/mp4"},
        {"link": "", "width": 1080, "height": 1920},
        {"link": "hls", "width": None, "height": None},
    ]
    assert M._pexels_pick_file(files, prefer_4k=False)["link"] == "u1080"
    assert M._pexels_pick_file(files, prefer_4k=True)["link"] == "u4k"
    assert M._pexels_pick_file(files[2:3], False)["link"] == "u720"


def test_pexels_videos_parse(monkeypatch):
    data = {"videos": [{"id": 7, "duration": 12, "user": {"name": "Ann"},
                        "video_files": [{"link": "a", "width": 1080, "height": 1920, "file_type": "video/mp4"}]}]}
    sess = S(data)
    monkeypatch.setattr(M, "http", lambda: sess)
    items = M.pexels_videos("octopus", "key")
    assert items[0].key == "pexels:7" and items[0].portrait and items[0].duration == 12
    assert sess.params[0]["orientation"] == "portrait"


def test_pixabay_prefers_medium_and_exposes_tags(monkeypatch):
    data = {"hits": [{"id": 3, "duration": 9, "user": "bob", "tags": "octopus, underwater, coral",
                      "pageURL": "https://pixabay.com/videos/x-3/", "videos": {
        "large": {"url": "L", "width": 3840, "height": 2160},
        "medium": {"url": "Md", "width": 1920, "height": 1080},
        "small": {"url": "", "width": 0, "height": 0}}}]}
    monkeypatch.setattr(M, "http", lambda: S(data))
    item = M.pixabay_videos("x", "k")[0]
    assert item.url == "Md"
    assert item.extra["tags"] == ["octopus", "underwater", "coral"]
    assert M.pixabay_videos("x", "k", prefer_4k=True)[0].url == "L"


def test_claim_skips_used_media():
    v = M.Visuals(Settings(), used_keys={"pexels:1"})
    a = M.MediaItem("pexels", "1", "video", width=1080, height=1920, duration=10)
    b = M.MediaItem("pexels", "2", "video", width=1920, height=1080, duration=10)
    assert v._claim([a, b], 5).key == "pexels:2"
    assert v._claim([a, b], 5) is None


def test_simplify_query():
    assert M.simplify_query("The octopus in the deep ocean") == "octopus deep"


def test_no_footage_falls_back_to_generated(tmp_path, caplog):
    import logging
    v = M.Visuals(Settings(visual_sources=["local"], media_dir=str(tmp_path / "none")))
    with caplog.at_level(logging.WARNING):
        item = v.for_scene(0, "octopus", "octopus", "sea", 3.0, tmp_path)
    assert item.kind == "generated"
    assert "no real visual" in caplog.text and "animated gradient" in caplog.text
    assert v.reports[0]["chosen"] is None and "no usable stock" in v.reports[0]["reason"]


class FetchResp:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.text = data, status, ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]


class FetchSession:
    def __init__(self, data, status=200):
        self.data, self.status, self.calls = data, status, []

    def get(self, url, headers=None, timeout=None, stream=False):
        self.calls.append({"url": url, "headers": headers or {}, "stream": stream})
        return FetchResp(self.data, self.status)


def _noise_png() -> bytes:
    image = Image.frombytes("RGB", (200, 200), os.urandom(200 * 200 * 3))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_pixabay_download_sends_referer_and_accepts_a_real_image(tmp_path, monkeypatch):
    sess = FetchSession(_noise_png())
    monkeypatch.setattr(M, "http", lambda: sess)
    v = M.Visuals(Settings(visual_sources=["pixabay"], pixabay_api_key="k"))
    item = M.MediaItem("pixabay-photo", "42", "image", url="https://cdn.pixabay.com/photo/x.jpg")
    out = v._fetch(item, tmp_path, 0)
    assert out.path.exists() and out.path.stat().st_size > 10000
    assert out.extra["validated"] is True
    assert sess.calls[0]["stream"] is True
    assert "Referer" in sess.calls[0]["headers"] and "User-Agent" in sess.calls[0]["headers"]


def test_pixabay_download_rejects_tiny_file(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "http", lambda: FetchSession(b"nope"))
    v = M.Visuals(Settings(visual_sources=["pixabay"], pixabay_api_key="k"))
    item = M.MediaItem("pixabay-photo", "43", "image", url="https://cdn.pixabay.com/photo/x.jpg")
    with pytest.raises(RuntimeError, match="invalid file"):
        v._fetch(item, tmp_path, 0)


def test_download_validation_rejects_an_html_error_page(tmp_path, monkeypatch):
    path = tmp_path / "fake.mp4"
    path.write_bytes(b"<html>" * 4000)
    monkeypatch.setattr(M.ffmpeg, "probe", lambda _: (_ for _ in ()).throw(RuntimeError("no stream")))
    with pytest.raises(RuntimeError, match="unreadable"):
        M.validate_download("video", path)


def test_relevance_uses_tags_not_photographer_credit():
    v = M.Visuals(Settings())
    good = M.MediaItem("pixabay", "1", "video", credit="Ann", extra={"tags": ["octopus", "coral"]})
    bad = M.MediaItem("pixabay", "2", "video", credit="Octopus Photographer", extra={"tags": ["beach", "sunset"]})
    kept = v._filter_relevant([good, bad], "octopus swimming", "octopus")
    assert [i.id for i in kept] == ["1"]


def test_stock_video_is_selected_before_agnes_even_when_agnes_is_listed_first(tmp_path, monkeypatch):
    candidate = M.MediaItem("pixabay", "17", "video", url="https://example.invalid/clip.mp4",
                            width=1080, height=1920, duration=8, extra={"tags": ["octopus"]})
    monkeypatch.setattr(M, "pixabay_videos", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(M, "pixabay_photos", lambda *args, **kwargs: [])
    agnes_calls = []
    monkeypatch.setattr(M, "agnes_image", lambda *args, **kwargs: agnes_calls.append(args))
    v = M.Visuals(Settings(visual_sources=["agnes", "pixabay"], pixabay_api_key="k", agnes_api_key="a"))
    monkeypatch.setattr(v, "_fetch", lambda item, *args: item)

    selected = v.for_scene(0, "octopus", "octopus", "ocean", 4.0, tmp_path)

    assert selected.source == "pixabay" and selected.kind == "video"
    assert agnes_calls == []
    assert v.reports[0]["chosen"]["kind"] == "video"
    assert "video" in v.reports[0]["reason"]


def test_stock_photo_used_when_no_videos_and_reason_is_reported(tmp_path, monkeypatch, caplog):
    photo = M.MediaItem("pixabay-photo", "22", "image", url="https://example.invalid/p.jpg",
                        width=1080, height=1920, extra={"tags": ["octopus"]})
    monkeypatch.setattr(M, "pixabay_videos", lambda *args, **kwargs: [])
    monkeypatch.setattr(M, "pixabay_photos", lambda *args, **kwargs: [photo])
    v = M.Visuals(Settings(visual_sources=["pixabay"], pixabay_api_key="k"))
    monkeypatch.setattr(v, "_fetch", lambda item, *args: item)

    with caplog.at_level("WARNING"):
        selected = v.for_scene(0, "octopus", "octopus", "ocean", 4.0, tmp_path)

    assert selected.kind == "image"
    assert "no suitable stock video" in v.reports[0]["reason"]
    assert "no suitable stock video" in caplog.text


def test_missing_stock_key_is_reported_and_ai_fallback_is_explicit(tmp_path, monkeypatch, caplog):
    def fake_agnes(prompt, dest, key, model, orientation, **kwargs):
        # Noise texture image passes blank/blur QC.
        import os
        Image.frombytes("RGB", (512, 768), os.urandom(512 * 768 * 3)).save(dest, "PNG")
        return dest

    monkeypatch.setattr(M, "agnes_image", fake_agnes)
    v = M.Visuals(Settings(visual_sources=["pixabay", "agnes"], agnes_api_key="a"))

    with caplog.at_level("WARNING"):
        selected = v.for_scene(0, "octopus", "octopus", "ocean", 4.0, tmp_path)

    assert selected.source == "agnes"
    assert "PIXABAY_API_KEY is not set" in caplog.text
    assert "no usable stock video or photo" in v.reports[0]["reason"]
    assert "generated an AI image" in v.reports[0]["reason"]


def test_pollinations_payment_required_stops_asking(monkeypatch, tmp_path, caplog):
    calls = []

    def refuse(prompt, dest, width, height, key=""):
        calls.append(key)
        raise M.PermanentError("download https://image.pollinations.ai/...: HTTP 402 — {}")
    monkeypatch.setattr(M, "pollinations_image", refuse)
    v = M.Visuals(Settings(visual_sources=["pollinations"], pollinations_api_key="pk_x"))
    items = [v.for_scene(i, "cat", "a cat", "cats", 3.0, tmp_path) for i in range(4)]
    assert calls == ["pk_x"] and all(i.kind == "generated" for i in items)
    assert "POLLINATIONS_API_KEY" in caplog.text and caplog.text.count("refused") == 1


def _png_bytes(size: tuple[int, int] = (512, 512)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, "#FF7A18").save(buf, "PNG")
    return buf.getvalue()


def test_agnes_image_posts_bearer_key_and_writes_base64(monkeypatch, tmp_path):
    import base64

    payload = _png_bytes()

    class PostResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"data": [{"b64_json": base64.b64encode(payload).decode()}]}

    class PostSession:
        def __init__(self):
            self.call = None

        def post(self, url, json=None, headers=None, timeout=None):
            self.call = {"url": url, "json": json, "headers": headers, "timeout": timeout}
            return PostResponse()

    session = PostSession()
    monkeypatch.setattr(M, "http", lambda: session)
    dest = tmp_path / "scene.png"
    result = M.agnes_image("cinematic octopus", dest, "test-key", "agnes-image-2.5-flash", "portrait")

    assert result == dest
    assert dest.read_bytes() == payload
    assert session.call["url"] == "https://apihub.agnes-ai.com/v1/images/generations"
    assert session.call["headers"]["Authorization"] == "Bearer test-key"
    assert session.call["json"]["model"] == "agnes-image-2.5-flash"
    assert session.call["json"]["size"] == "1024x1536"


def test_generated_image_validation_rejects_non_images(tmp_path):
    path = tmp_path / "not-an-image.png"
    path.write_bytes(b"<html>payment required</html>" * 100)
    with pytest.raises(RuntimeError, match="unreadable"):
        M.validate_generated_image(path)


def test_generated_image_validation_rejects_tiny_images(tmp_path):
    path = tmp_path / "tiny.png"
    Image.frombytes("RGB", (128, 128), os.urandom(128 * 128 * 3)).save(path, "PNG")
    with pytest.raises(RuntimeError, match="below.*minimum"):
        M.validate_generated_image(path)


def test_generated_image_validation_accepts_a_real_image(tmp_path):
    path = tmp_path / "scene.png"
    path.write_bytes(_png_bytes((512, 512)))
    assert M.validate_generated_image(path) == (512, 512)


def test_agnes_is_selected_as_a_generated_visual(monkeypatch, tmp_path):
    calls = []

    def fake_agnes(prompt, dest, key, model, orientation, **kwargs):
        calls.append((prompt, key, model, orientation))
        import os
        Image.frombytes("RGB", (512, 768), os.urandom(512 * 768 * 3)).save(dest, "PNG")
        return dest

    monkeypatch.setattr(M, "agnes_image", fake_agnes)
    settings = Settings(visual_sources=["agnes"], agnes_api_key="test-key")
    visual = M.Visuals(settings)
    item = visual.for_scene(0, "octopus", "detailed octopus", "ocean life", 3.0, tmp_path)

    assert item.source == "agnes"
    assert item.kind == "image"
    assert item.path.exists()
    assert calls and calls[0][1] == "test-key"


def test_ai_is_used_only_after_stock_sources_are_exhausted(monkeypatch, tmp_path):
    candidate = M.MediaItem("pixabay", "17", "video", url="https://example.invalid/clip.mp4",
                            width=1080, height=1920, duration=5, extra={"tags": ["neutron"]})
    monkeypatch.setattr(M, "pixabay_videos", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(M, "pixabay_photos", lambda *args, **kwargs: [])
    visual = M.Visuals(Settings(visual_sources=["agnes", "pixabay"], pixabay_api_key="fake-pixabay-key"))
    monkeypatch.setattr(visual, "_fetch", lambda item, *args: item)

    selected = visual.for_scene(0, "neutron star", "neutron star", "space", 3.0, tmp_path)

    assert selected.source == "pixabay"


def test_hybrid_mode_alternates_pixabay_clips_and_agnes_images(monkeypatch, tmp_path):
    def videos(query, *args, **kwargs):
        return [M.MediaItem("pixabay", f"v-{query}", "video", url=f"https://example.invalid/{query}.mp4",
                            width=1080, height=1920, duration=5, extra={"tags": query.split()})]

    prompts = []

    def agnes(prompt, dest, *args, **kwargs):
        prompts.append(prompt)
        import os
        Image.frombytes("RGB", (512, 768), os.urandom(512 * 768 * 3)).save(dest, "PNG")
        return dest

    monkeypatch.setattr(M, "pixabay_videos", videos)
    monkeypatch.setattr(M, "pixabay_photos", lambda *args, **kwargs: [])
    monkeypatch.setattr(M, "agnes_image", agnes)
    visual = M.Visuals(Settings(visual_sources=["pixabay", "agnes"], pixabay_api_key="fake-pixabay-key",
                                agnes_api_key="fake-agnes-key", visual_mode="hybrid", hybrid_image_every=2))
    monkeypatch.setattr(visual, "_fetch", lambda item, *args: item)
    jobs = [(f"ocean {i}", f"prompt {i}", 3.0, f"narration {i}") for i in range(5)]

    selected = visual.for_scenes(jobs, "ocean", tmp_path)

    assert [(m.source, m.kind) for m in selected] == [
        ("pixabay", "video"), ("agnes", "image"), ("pixabay", "video"),
        ("agnes", "image"), ("pixabay", "video")]
    assert len(prompts) == 2
    prompt_texts = set(prompts)
    assert any("prompt 1" in p and "narration 1" in p for p in prompt_texts)
    assert any("prompt 3" in p and "narration 3" in p for p in prompt_texts)
    assert visual.reports[1]["reason"].startswith("hybrid mode")
    assert visual.reports[1]["chosen"]["source"] == "agnes"


def test_hybrid_ai_failure_falls_back_to_stock(monkeypatch, tmp_path, caplog):
    candidate = M.MediaItem("pixabay", "18", "video", url="https://example.invalid/clip.mp4",
                            width=1080, height=1920, duration=5, extra={"tags": ["ocean"]})
    monkeypatch.setattr(M, "pixabay_videos", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(M, "pixabay_photos", lambda *args, **kwargs: [])

    def fail(*args, **kwargs):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(M, "agnes_image", fail)
    visual = M.Visuals(Settings(visual_sources=["pixabay", "agnes"], pixabay_api_key="fake-pixabay-key",
                                agnes_api_key="fake-agnes-key", visual_mode="hybrid"))
    monkeypatch.setattr(visual, "_fetch", lambda item, *args: item)

    selected = visual.for_scene(0, "ocean", "ocean", "ocean", 3.0, tmp_path, prefer_image=True,
                                narration="An octopus has three hearts.")

    assert selected.source == "pixabay"
    assert visual.reports[0]["chosen"]["source"] == "pixabay"
    assert "hybrid AI image unavailable" in visual.reports[0]["reason"]
    assert "falling back to stock footage" in caplog.text


def test_hybrid_missing_ai_key_reports_and_uses_stock(monkeypatch, tmp_path, caplog):
    candidate = M.MediaItem("pixabay", "19", "video", url="https://example.invalid/clip.mp4",
                            width=1080, height=1920, duration=5, extra={"tags": ["ocean"]})
    monkeypatch.setattr(M, "pixabay_videos", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(M, "pixabay_photos", lambda *args, **kwargs: [])
    monkeypatch.setattr(M, "agnes_image", lambda *args, **kwargs: pytest.fail("must not call Agnes without a key"))
    visual = M.Visuals(Settings(visual_sources=["pixabay", "agnes"], pixabay_api_key="fake-pixabay-key",
                                agnes_api_key="", visual_mode="hybrid"))
    monkeypatch.setattr(visual, "_fetch", lambda item, *args: item)

    selected = visual.for_scene(0, "ocean", "ocean", "ocean", 3.0, tmp_path, prefer_image=True)

    assert selected.source == "pixabay"
    assert "AGNES_API_KEY is not set" in caplog.text
    assert "hybrid AI image unavailable" in visual.reports[0]["reason"]


def test_stock_download_failure_tries_another_candidate_before_ai(monkeypatch, tmp_path):
    candidates = [
        M.MediaItem("pixabay", "20", "video", url="https://example.invalid/a.mp4",
                    width=1080, height=1920, duration=5, extra={"tags": ["ocean"]}),
        M.MediaItem("pixabay", "21", "video", url="https://example.invalid/b.mp4",
                    width=1080, height=1920, duration=5, extra={"tags": ["ocean"]}),
    ]
    downloads = []

    def fetch(item, *args):
        downloads.append(item.id)
        if len(downloads) == 1:
            raise RuntimeError("403 from CDN")
        return item

    monkeypatch.setattr(M, "pixabay_videos", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(M, "pixabay_photos", lambda *args, **kwargs: [])
    monkeypatch.setattr(M, "agnes_image", lambda *args, **kwargs: pytest.fail("stock retry should succeed"))
    visual = M.Visuals(Settings(visual_sources=["pixabay", "agnes"], pixabay_api_key="fake-pixabay-key",
                                agnes_api_key="fake-agnes-key"))
    monkeypatch.setattr(visual, "_fetch", fetch)

    selected = visual.for_scene(0, "ocean", "ocean", "ocean", 3.0, tmp_path)

    assert len(downloads) == 2
    assert selected.id == downloads[1] and selected.kind == "video"
    assert any(a["result"] == "download-failed" for a in visual.reports[0]["attempts"])


# ------------------------------------------------------------------ Agnes prompt builder tests
def test_build_agnes_prompt_includes_narration_and_visuals():
    prompt = M.build_agnes_prompt(
        narration="The octopus has three hearts and blue blood.",
        visual_description="octopus swimming in deep ocean",
        scene_index=0,
        topic="octopus",
    )
    assert "octopus swimming in deep ocean" in prompt
    assert "Scene narrative: The octopus has three hearts" in prompt
    assert "documentary cinematography" in prompt
    assert "eye-level medium shot" in prompt
    assert "golden hour lighting" in prompt
    assert "vertical 9:16" in prompt
    assert "No text" in prompt


def test_build_agnes_prompt_rotates_camera_and_lighting():
    p0 = M.build_agnes_prompt("narration", "subject", scene_index=0)
    p1 = M.build_agnes_prompt("narration", "subject", scene_index=1)
    p2 = M.build_agnes_prompt("narration", "subject", scene_index=2)
    assert p0 != p1 != p2  # different scenes get different camera/lighting
    assert "eye-level medium shot" in p0
    assert "low-angle wide shot" in p1
    assert "close-up detail shot" in p2


# ------------------------------------------------------------------ Image QC tests
def _write_test_png(dest, size=(512, 768), noise=True):
    """Write a PNG that passes QC (portrait, textured, not blank)."""
    import os
    if noise:
        img = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3))
    else:
        img = Image.new("RGB", size, "#FF7A18")
    img.save(dest, "PNG")


def test_detect_blank_image_rejects_solid_color(tmp_path):
    img = Image.new("RGB", (100, 100), "#FF7A18")
    blank, msg = M._detect_blank_image(img)
    assert blank is True
    assert "blank" in msg.lower()


def test_detect_blank_image_passes_textured(tmp_path):
    img = Image.frombytes("RGB", (100, 100), os.urandom(100 * 100 * 3))
    blank, msg = M._detect_blank_image(img)
    assert blank is False


def test_detect_blurry_image_passes_textured():
    img = Image.frombytes("RGB", (200, 200), os.urandom(200 * 200 * 3))
    blurry, msg = M._detect_blurry_image(img)
    assert blurry is False


def test_check_vertical_composition_passes_portrait():
    img = Image.new("RGB", (512, 768), "#FF7A18")
    ok, msg = M._check_vertical_composition(img)
    assert ok is True
    assert "portrait" in msg.lower()


def test_check_vertical_composition_fails_landscape():
    img = Image.new("RGB", (768, 512), "#FF7A18")
    ok, msg = M._check_vertical_composition(img)
    assert ok is False
    assert "landscape" in msg.lower()


def test_validate_image_quality_passes_good_image(tmp_path):
    path = tmp_path / "good.png"
    _write_test_png(path)
    result = M.validate_image_quality(path, min_dimension=256)
    assert result.passed is True
    assert result.score > 50
    assert result.dimensions == (512, 768)


def test_validate_image_quality_rejects_blank(tmp_path):
    path = tmp_path / "blank.png"
    Image.new("RGB", (512, 768), "#FF7A18").save(path, "PNG")
    result = M.validate_image_quality(path, min_dimension=256)
    assert result.passed is False
    assert "blank" in " ".join(result.reasons).lower()


def test_validate_image_quality_rejects_landscape(tmp_path):
    path = tmp_path / "landscape.png"
    Image.new("RGB", (768, 512), "#FF7A18").save(path, "PNG")
    result = M.validate_image_quality(path, min_dimension=256)
    assert result.passed is False
    assert any("composition" in r.lower() or "landscape" in r.lower() for r in result.reasons)


def test_validate_image_quality_rejects_missing_file(tmp_path):
    path = tmp_path / "nonexistent.png"
    result = M.validate_image_quality(path, min_dimension=256)
    assert result.passed is False
    assert result.score == 0.0


# ------------------------------------------------------------------ Agnes QA with retries
def test_agnes_image_with_qa_passes_on_first_try(tmp_path, monkeypatch):
    calls = []

    def fake_agnes(prompt, dest, key, model, orientation, **kwargs):
        calls.append(prompt)
        import os
        Image.frombytes("RGB", (512, 768), os.urandom(512 * 768 * 3)).save(dest, "PNG")
        return dest

    monkeypatch.setattr(M, "agnes_image", fake_agnes)
    dest = tmp_path / "scene.png"
    result = M.agnes_image_with_qa(
        "octopus", dest, "test-key",
        scene_index=0, narration="The octopus has three hearts.",
        visual_description="octopus swimming",
    )
    assert result is not None
    path, qr = result
    assert path == dest
    assert qr.passed is True
    assert qr.score > 50
    assert len(calls) == 1  # no retries needed


def test_agnes_image_with_qa_retries_on_qc_failure(tmp_path, monkeypatch):
    """First attempt returns blank image, second returns valid image."""
    attempts = []

    def fake_agnes(prompt, dest, key, model, orientation, **kwargs):
        attempts.append(len(attempts))
        import os
        if len(attempts) == 1:
            # First attempt: blank solid color (fails QC)
            Image.new("RGB", (512, 768), "#FF7A18").save(dest, "PNG")
        else:
            # Second attempt: textured (passes QC)
            Image.frombytes("RGB", (512, 768), os.urandom(512 * 768 * 3)).save(dest, "PNG")
        return dest

    monkeypatch.setattr(M, "agnes_image", fake_agnes)
    dest = tmp_path / "scene.png"
    result = M.agnes_image_with_qa(
        "octopus", dest, "test-key",
        scene_index=0, narration="The octopus has three hearts.",
        visual_description="octopus swimming",
        max_retries=2,
    )
    assert result is not None
    path, qr = result
    assert qr.passed is True
    assert len(attempts) == 2  # first failed QC, second passed


def test_agnes_image_with_qa_fails_after_all_retries(tmp_path, monkeypatch):
    """All attempts return blank images, so QA returns None."""
    def fake_agnes(prompt, dest, key, model, orientation, **kwargs):
        import os
        Image.new("RGB", (512, 768), "#FF7A18").save(dest, "PNG")
        return dest

    monkeypatch.setattr(M, "agnes_image", fake_agnes)
    dest = tmp_path / "scene.png"
    result = M.agnes_image_with_qa(
        "octopus", dest, "test-key",
        scene_index=0, narration="The octopus has three hearts.",
        visual_description="octopus swimming",
        max_retries=1,
    )
    assert result is None  # all retries exhausted


# ------------------------------------------------------------------ visual mixing regression tests
def _fake_pixabay_video(query, *args, **kwargs):
    return [M.MediaItem("pixabay", f"v-{query}", "video",
                        url=f"https://example.invalid/{query}.mp4",
                        width=1080, height=1920, duration=5,
                        extra={"tags": query.split()})]


def _fake_agnes_image(prompt, dest, *args, **kwargs):
    import os
    Image.frombytes("RGB", (512, 768), os.urandom(512 * 768 * 3)).save(dest, "PNG")
    return dest


def test_hybrid_suitability_image_slots_prefer_ai_for_filmable_scenes(monkeypatch, tmp_path):
    """Hybrid + suitability: an image slot requests AI first even for filmable content,
    so the same video mixes Pixabay clips and Agnes images."""
    monkeypatch.setattr(M, "pixabay_videos", _fake_pixabay_video)
    monkeypatch.setattr(M, "pixabay_photos", lambda *args, **kwargs: [])
    monkeypatch.setattr(M, "agnes_image", _fake_agnes_image)
    visual = M.Visuals(Settings(visual_sources=["pixabay", "agnes"],
                                pixabay_api_key="fake-pixabay-key",
                                agnes_api_key="fake-agnes-key",
                                visual_mode="hybrid", hybrid_image_every=2,
                                visual_choice="suitability"))
    monkeypatch.setattr(visual, "_fetch", lambda item, *args: item)
    # Even-numbered scenes (index 1, 3) are hybrid image slots.
    jobs = [(f"ocean {i}", f"prompt {i}", 3.0, f"narration {i}") for i in range(4)]
    selected = visual.for_scenes(jobs, "ocean", tmp_path)

    sources = [(m.source, m.kind) for m in selected]
    # Scene 0 and 2 are stock slots (filmable -> pixabay); scene 1 and 3 are image slots
    # that request AI first even though the content is filmable.
    assert sources[0] == ("pixabay", "video")
    assert sources[1] == ("agnes", "image")
    assert sources[2] == ("pixabay", "video")
    assert sources[3] == ("agnes", "image")
    assert len(visual.reports) == 4
    for rep in visual.reports.values():
        assert rep["chosen"] is not None, rep  # every scene records its selected source


def test_missing_agnes_key_clear_diagnostic_and_stock_fallback(monkeypatch, tmp_path, caplog):
    """No Agnes key: a clear diagnostic is logged, the source is dropped, and the
    image slot falls back to Pixabay stock rather than silently using AI."""
    monkeypatch.setattr(M, "pixabay_videos", _fake_pixabay_video)
    monkeypatch.setattr(M, "pixabay_photos", lambda *args, **kwargs: [])
    called = []
    monkeypatch.setattr(M, "agnes_image", lambda *a, **k: called.append(True) or (_ for _ in ()).throw(
        AssertionError("Agnes must not be called without a key")))
    visual = M.Visuals(Settings(visual_sources=["pixabay", "agnes"],
                                pixabay_api_key="fake-pixabay-key",
                                agnes_api_key="", visual_mode="hybrid"))
    monkeypatch.setattr(visual, "_fetch", lambda item, *args: item)

    selected = visual.for_scene(0, "ocean", "ocean", "ocean", 3.0, tmp_path,
                                prefer_image=True, narration="An octopus has three hearts.")

    assert selected.source == "pixabay"
    assert "AGNES_API_KEY is not set" in caplog.text
    assert "hybrid AI image unavailable" in visual.reports[0]["reason"]
    assert not called  # Agnes was never invoked


def test_ai_failure_for_one_scene_does_not_abort_others(monkeypatch, tmp_path):
    """One scene's AI generation fails: that scene falls back to stock gracefully
    (recorded in the report) instead of aborting the whole video, and later image
    slots still receive AI images — so the same video mixes sources."""
    def videos(query, *args, **kwargs):
        return [M.MediaItem("pixabay", f"v-{query}", "video",
                            url=f"https://example.invalid/{query}.mp4",
                            width=1080, height=1920, duration=5,
                            extra={"tags": query.split()})]

    def agnes(prompt, dest, *args, **kwargs):
        # Scene 1's image slot (prompt 1) fails; scene 3's succeeds. Deterministic per scene
        # so the result is stable regardless of parallel scheduling order.
        if "prompt 1" in prompt:
            raise RuntimeError("Agnes transient failure")
        Image.frombytes("RGB", (512, 768), os.urandom(512 * 768 * 3)).save(dest, "PNG")
        return dest

    monkeypatch.setattr(M, "pixabay_videos", videos)
    monkeypatch.setattr(M, "pixabay_photos", lambda *args, **kwargs: [])
    monkeypatch.setattr(M, "agnes_image", agnes)
    visual = M.Visuals(Settings(visual_sources=["pixabay", "agnes"],
                                pixabay_api_key="fake-pixabay-key",
                                agnes_api_key="fake-agnes-key",
                                visual_mode="hybrid", hybrid_image_every=2))
    monkeypatch.setattr(visual, "_fetch", lambda item, *args: item)

    # Scenes 0 (stock), 1 (image slot -> AI fails -> stock fallback), 2 (stock),
    # 3 (image slot -> AI succeeds). No scene aborts the whole video.
    jobs = [(f"ocean {i}", f"prompt {i}", 3.0, f"narration {i}") for i in range(4)]
    selected = visual.for_scenes(jobs, "ocean", tmp_path)

    assert [m.source for m in selected] == ["pixabay", "pixabay", "pixabay", "agnes"]
    assert len(visual.reports) == 4
    for rep in visual.reports.values():
        assert rep["chosen"] is not None, rep
    # Scene 1's transient AI failure was handled safely (stock fallback, logged+reported).
    assert visual.reports[1]["chosen"]["source"] == "pixabay"
    assert visual.reports[3]["chosen"]["source"] == "agnes"


def test_all_slot_workflows_request_mixed_visual_sources():
    """All three scheduled workflows enable hybrid mixing of Pixabay stock and Agnes AI images."""
    import yaml
    from pathlib import Path as P
    root = P(__file__).resolve().parent.parent
    for name in ("curiostick-slot-1.yml", "curiostick-slot-2.yml", "curiostick-slot-3.yml"):
        data = yaml.safe_load((root / ".github/workflows" / name).read_text(encoding="utf-8"))
        env = data["jobs"]["publish-short"]["env"]
        assert env["VISUAL_SOURCES"] == "pixabay,agnes", name
        assert env["VISUAL_CHOICE"] == "suitability", name
        assert env["VISUAL_MODE"] == "hybrid", name
        assert "AGNES_API_KEY" in env and "PIXABAY_API_KEY" in env, name
