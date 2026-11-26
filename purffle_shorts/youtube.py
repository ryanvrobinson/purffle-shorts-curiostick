"""YouTube Data API v3: OAuth (token.json), resumable uploads with retries, scheduled publishing,
AI-content disclosure, playlists and quota awareness."""

from __future__ import annotations

import logging
import os
import pickle
import random
import shutil
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path

from .config import Settings
from .utils import truncate_title

log = logging.getLogger("purffle")

SCOPE_UPLOAD = "https://www.googleapis.com/auth/youtube.upload"
SCOPE_MANAGE = "https://www.googleapis.com/auth/youtube"
SCOPE_READONLY = "https://www.googleapis.com/auth/youtube.readonly"
RETRY_STATUS = {500, 502, 503, 504}
PROCESSING_TIMEOUT_SECONDS = 900
PROCESSING_POLL_INTERVAL_SECONDS = 10

# ``invalid_grant`` from Google's OAuth2 token endpoint means the refresh token
# is revoked, expired, or belongs to a different client. It is permanent: no
# amount of retrying will obtain a new access token. Only a fresh interactive
# authorization flow (``purffle-shorts auth``) can repair it.
_INVALID_GRANT = "invalid_grant"

_lock = threading.Lock()


class DuplicateCheckError(RuntimeError):
    """YouTube could not verify whether a matching video already exists."""


class VideoUploadIncomplete(RuntimeError):
    """A private YouTube upload exists but was not safely finalized."""

    def __init__(self, video_id: str | None, message: str):
        super().__init__(message)
        self.video_id = video_id


class ThumbnailUploadError(VideoUploadIncomplete):
    """The uploaded video remains private because its thumbnail was not verified."""


class UploadFinalizationError(VideoUploadIncomplete):
    """The thumbnail was attached but the requested final status was not verified."""


def find_existing_video(yt, marker: str) -> str | None:
    """Return a matching video ID or fail closed when YouTube cannot be queried.

    Used as a cross-runner idempotency net: if a slot's video is already on YouTube
    (and the history database did not carry over to this runner), we do not re-upload.
    """
    try:
        resp = yt.search().list(part="id", forMine=True, type="video",
                                q=marker, maxResults=5).execute()
        items = resp.get("items") or []
        if items:
            video_id = (items[0].get("id") or {}).get("videoId")
            if not video_id:
                raise ValueError("duplicate search returned a malformed video resource")
            return str(video_id)
    except Exception as e:
        log.warning("YouTube duplicate search failed (%s); upload blocked", type(e).__name__)
        raise DuplicateCheckError(
            f"YouTube duplicate search failed ({type(e).__name__}); upload blocked."
        ) from e
    return None


class QuotaExceeded(RuntimeError):
    pass


class NotAuthorized(RuntimeError):
    pass


def scopes_for(settings: Settings) -> list[str]:
    return [SCOPE_UPLOAD] + ([SCOPE_MANAGE] if settings.playlist_id else [])


def auth_scopes(settings: Settings) -> list[str]:
    """Everything the current settings will need; asked for once by the `auth` command."""
    extra = [SCOPE_READONLY] if settings.learn_from_stats and not settings.youtube_api_key else []
    return scopes_for(settings) + extra


def _load_legacy_pickle():
    """PurffleShorts 1.x stored credentials in token.pickle; migrate them once to token.json."""
    p = Path("token.pickle")
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)  # file written by this app on this machine
    except Exception as e:
        log.warning("Could not read legacy token.pickle: %s", e)
        return None


def _backup_existing_token(token: Path) -> None:
    """Back up an existing token file before replacement, without exposing contents.

    The backup is a plain copy (original is preserved). Contents are never logged
    or printed.
    """
    if token.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = token.with_name(f"{token.name}.bak.{stamp}")
        try:
            shutil.copy2(token, backup)
            log.info("Existing token backed up to %s", backup.name)
        except OSError as e:
            log.warning("Could not back up existing token file: %s", e)


def _run_interactive_flow(secrets: Path, scopes: list[str]):
    """Run an interactive OAuth authorization-code flow for remote environments.

    Designed for GitHub Codespaces where a localhost callback server cannot be
    reached from the user's browser. Instead of ``run_local_server()`` this
    displays the Google authorization URL, asks the user to open it in their
    normal browser, and then prompts them to paste the authorization response
    (or just the code) back into the terminal.

    Uses PKCE with state validation. Never prints or logs authorization codes,
    access tokens, refresh tokens, client secrets, or complete credential JSON.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), scopes)

    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
    )

    # Safe diagnostic: verify the authorization URL carries the required scopes
    # without leaking the full URL, state, or credentials.
    parsed_auth_url = urllib.parse.urlparse(auth_url)
    auth_params = urllib.parse.parse_qs(parsed_auth_url.query)
    auth_scope = auth_params.get("scope", [""])[0]
    has_scope = bool(auth_scope)
    has_youtube_upload = "https://www.googleapis.com/auth/youtube.upload" in auth_scope
    redirect_uri = flow.redirect_uri or "http://localhost"
    is_localhost = redirect_uri.startswith("http://localhost") or redirect_uri.startswith(
        "http://127.0.0.1"
    )

    print("\n" + "=" * 66)
    print("YouTube OAuth Authorization")
    print("=" * 66)
    print("1. Open this URL in your browser:")
    print()
    print(f"   {auth_url}")
    print()
    print("OAuth authorization URL diagnostics:")
    print(f"  Scope present: {has_scope}")
    print(f"  YouTube upload scope: {has_youtube_upload}")
    print(f"  Redirect URI localhost: {is_localhost}")
    print()
    if not has_scope or not has_youtube_upload:
        print(
            "WARNING: Authorization URL may be missing required scopes. "
            "Authorization may fail with Error 400: invalid_request."
        )
    print()
    print("2. Sign in to your Google account and grant permission.")
    print("3. After authorization, copy the FULL redirect URL from your")
    print("   browser's address bar (it may start with http://localhost).")
    print()
    print("4. Paste the full URL (or just the authorization code) below:")
    print("=" * 66)

    response = input("Authorization response: ").strip()
    if not response:
        raise NotAuthorized("No authorization response provided.")

    if response.startswith("http://") or response.startswith("https://"):
        flow.fetch_token(authorization_response=response)
    else:
        # User pasted just the authorization code; wrap it in a synthetic
        # authorization response URL so PKCE and state validation still run.
        redirect_uri = flow.redirect_uri or "http://localhost"
        fake_response = f"{redirect_uri}?{urllib.parse.urlencode({'code': response, 'state': state})}"
        flow.fetch_token(authorization_response=fake_response)

    creds = flow.credentials

    if not creds.refresh_token:
        raise NotAuthorized(
            "Google did not issue a refresh token. This can happen if you already "
            "authorized this OAuth client before (Google issues refresh tokens only "
            "on first grant) or if the client is not configured as a Desktop app in "
            "Google Cloud Console. Revoke the client in your Google Account "
            "permissions (myaccount.google.com/permissions), then run "
            "'purffle-shorts auth' again."
        )

    return creds


def get_credentials(settings: Settings, interactive: bool = True, extra_scopes: list[str] | None = None):
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    scopes = list(dict.fromkeys(scopes_for(settings) + list(extra_scopes or [])))
    token = Path(settings.token_file)
    creds = None
    if token.exists():
        creds = Credentials.from_authorized_user_file(str(token))
    else:
        creds = _load_legacy_pickle()
    if creds is not None and not creds.has_scopes(scopes):
        log.info("Saved YouTube authorization lacks required scopes; asking again")
        creds = None
    if creds is not None and not creds.valid and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as e:
            err_text = str(e).lower()
            if _INVALID_GRANT in err_text:
                log.error(
                    "YouTube refresh token is revoked or invalid (invalid_grant). "
                    "Re-authorization is required: run 'purffle-shorts auth' with credentials.json, "
                    "then paste the new token.json into the YOUTUBE_TOKEN_JSON GitHub Actions secret. "
                    "Code cannot repair a revoked refresh token."
                )
            else:
                log.warning("Refreshing YouTube token failed (%s); re-authorization needed", e)
            creds = None
        except Exception as e:
            log.warning("Refreshing YouTube token failed (%s); re-authorization needed", e)
            creds = None
    if creds is None or not creds.valid:
        if not interactive:
            raise NotAuthorized("YouTube is not authorized (or needs a new permission). Run:  purffle-shorts auth")
        secrets = Path(settings.client_secrets)
        if not secrets.exists():
            raise NotAuthorized(
                f"{secrets} not found. Create an OAuth client (Desktop app) in Google Cloud Console with the "
                "YouTube Data API v3 enabled, download its JSON as credentials.json, then run the auth command.")
        creds = _run_interactive_flow(secrets, scopes)
    _backup_existing_token(token)
    token.write_text(creds.to_json(), encoding="utf-8")
    try:
        os.chmod(token, 0o600)
    except OSError:
        pass
    return creds


def service(settings: Settings, interactive: bool = True, extra_scopes: list[str] | None = None):
    from googleapiclient.discovery import build
    creds = get_credentials(settings, interactive, extra_scopes)
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _tz(settings: Settings):
    if settings.timezone:
        from zoneinfo import ZoneInfo
        return ZoneInfo(settings.timezone)
    return datetime.now().astimezone().tzinfo


def next_publish_slot(settings: Settings, taken: list[str], now: datetime | None = None) -> datetime | None:
    """Next free PUBLISH_TIMES slot (local time) at least 15 minutes from now, skipping taken ones."""
    if not settings.publish_times:
        return None
    tz = _tz(settings)
    now = (now or datetime.now(timezone.utc)).astimezone(tz)
    slots = []
    for t in settings.publish_times:
        try:
            hh, mm = (int(x) for x in t.split(":"))
            slots.append(dtime(hh, mm))
        except ValueError:
            log.warning("Ignoring bad PUBLISH_TIMES entry %r (use HH:MM)", t)
    taken_set = {x for x in taken if x}
    for day in range(0, 60):
        date = (now + timedelta(days=day)).date()
        for slot in sorted(slots):
            cand = datetime.combine(date, slot, tzinfo=tz)
            if cand < now + timedelta(minutes=15):
                continue
            iso = to_rfc3339(cand)
            if iso not in taken_set:
                return cand
    return None


def to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_body(settings: Settings, title: str, description: str, tags: list[str], category_id: str,
               language: str, publish_at: datetime | None = None) -> dict:
    status = {
        "privacyStatus": "private" if publish_at else settings.privacy,
        "selfDeclaredMadeForKids": settings.made_for_kids,
        "embeddable": True,
    }
    if settings.synthetic_media:
        status["containsSyntheticMedia"] = True
    if publish_at:
        status["publishAt"] = to_rfc3339(publish_at)
    return {
        "snippet": {
            "title": truncate_title(title, 100),
            "description": description[:4900],
            "tags": tags,
            "categoryId": category_id,
            "defaultLanguage": language,
            "defaultAudioLanguage": language,
        },
        "status": status,
    }


def upload(settings: Settings, video: Path, body: dict) -> dict:
    """Upload privately, attach a verified thumbnail, then apply the requested visibility."""
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    thumbnail = video.parent / "cover.jpg"
    if not thumbnail.is_file() or thumbnail.stat().st_size == 0:
        raise ThumbnailUploadError(None, "cover.jpg is missing or empty; upload was not started")

    with _lock:  # one upload at a time; the API client is not thread-safe
        yt = service(settings, interactive=False)
        media = MediaFileUpload(str(video), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
        requested_status = dict(body.get("status") or {})
        requested_status.setdefault("privacyStatus", "private")
        private_status = dict(requested_status)
        private_status["privacyStatus"] = "private"
        private_status.pop("publishAt", None)
        private_body = dict(body)
        private_body["status"] = private_status
        request = yt.videos().insert(part="snippet,status", body=private_body, media_body=media)
        response, retry, last_pct = None, 0, -1
        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    pct = int(status.progress() * 100)
                    if pct >= last_pct + 25:
                        log.info("Upload %d%%", pct)
                        last_pct = pct
            except HttpError as e:
                text = str(e)
                # quotaExceeded comes as a 403 and uploadLimitExceeded as a 400; both clear at midnight Pacific
                if "quotaExceeded" in text or "uploadLimitExceeded" in text:
                    raise QuotaExceeded(text) from e
                if e.resp.status not in RETRY_STATUS:
                    raise
                retry += 1
            except (ConnectionError, TimeoutError, OSError) as e:
                log.warning("Upload connection error: %s", e)
                retry += 1
            else:
                continue
            if retry > 8:
                raise RuntimeError("Upload failed after 8 retries")
            wait = min(64, 2 ** retry) + random.random()
            log.warning("Upload retry %d in %.0fs", retry, wait)
            time.sleep(wait)

        video_id = response.get("id") if isinstance(response, dict) else None
        if not video_id:
            raise RuntimeError("YouTube upload completed without returning a video ID")

        try:
            _wait_until_processed(yt, str(video_id))
            thumb_response = yt.thumbnails().set(
                videoId=str(video_id),
                media_body=MediaFileUpload(str(thumbnail), mimetype="image/jpeg", resumable=False),
            ).execute()
            _verify_thumbnail_response(thumb_response)
        except Exception as e:
            log.error("Thumbnail attachment failed for private video %s (%s)",
                      video_id, type(e).__name__)
            raise ThumbnailUploadError(
                str(video_id),
                f"Thumbnail attachment failed for video {video_id}; it remains private.",
            ) from e

        if settings.playlist_id:
            try:
                playlist_body = {"snippet": {
                    "playlistId": settings.playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": str(video_id)},
                }}
                yt.playlistItems().insert(part="snippet", body=playlist_body).execute()
            except Exception as e:
                log.warning("Could not add video to playlist %s: %s", settings.playlist_id, e)

        try:
            finalized = yt.videos().update(
                part="status",
                body={"id": str(video_id), "status": requested_status},
            ).execute()
            finalized_status = (finalized or {}).get("status") or {}
            expected_privacy = requested_status["privacyStatus"]
            if (not isinstance(finalized, dict) or finalized.get("id") != str(video_id)
                    or finalized_status.get("privacyStatus") != expected_privacy):
                raise ValueError("YouTube did not confirm the requested privacy status")
            return finalized
        except Exception as e:
            log.error("Final status update failed for video %s (%s); video was not confirmed",
                      video_id, type(e).__name__)
            raise UploadFinalizationError(
                str(video_id),
                f"Final YouTube status was not verified for video {video_id}.",
            ) from e


def _wait_until_processed(yt, video_id: str) -> None:
    deadline = time.monotonic() + PROCESSING_TIMEOUT_SECONDS
    while True:
        response = yt.videos().list(part="processingDetails", id=video_id).execute()
        items = response.get("items") if isinstance(response, dict) else None
        if not items:
            raise RuntimeError("YouTube returned no processing status for the uploaded video")
        status = (items[0].get("processingDetails") or {}).get("processingStatus")
        if status == "succeeded":
            return
        if status in {"failed", "terminated"}:
            raise RuntimeError(f"YouTube video processing ended with status {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError("YouTube video processing did not finish before the thumbnail timeout")
        time.sleep(PROCESSING_POLL_INTERVAL_SECONDS)


def _verify_thumbnail_response(response: dict) -> None:
    items = response.get("items") if isinstance(response, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("YouTube returned no thumbnail resource")
    for item in items:
        if not isinstance(item, dict):
            continue
        if any(isinstance(item.get(size), dict) and item[size].get("url")
               for size in ("default", "medium", "high", "standard", "maxres", "shortsAspectRatio")):
            return
    raise ValueError("YouTube response did not confirm a thumbnail URL")
