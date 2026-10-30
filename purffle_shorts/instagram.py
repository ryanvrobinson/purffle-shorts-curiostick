"""Instagram Reels publishing via Meta Graph API.

Requires:
- Instagram Professional Account (Business or Creator)
- Meta App with Instagram Graph API access
- Long-lived access token with permissions:
  - instagram_graph_user_profile
  - instagram_graph_user_media
  - instagram_content_publish

Environment variables:
- INSTAGRAM_APP_ID: Meta App ID
- INSTAGRAM_APP_SECRET: Meta App Secret
- INSTAGRAM_USER_ID: Instagram Business Account ID (not the username)
- INSTAGRAM_ACCESS_TOKEN: Long-lived access token
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Settings
from .pipeline import Result
from .utils import http, raise_for_status

log = logging.getLogger("purffle")

API_BASE = "https://graph.facebook.com/v20.0"


@dataclass
class InstagramConfig:
    """Instagram publishing configuration."""
    app_id: str
    app_secret: str
    user_id: str
    access_token: str

    @classmethod
    def from_env(cls) -> Optional["InstagramConfig"]:
        """Load config from environment variables."""
        app_id = os.getenv("INSTAGRAM_APP_ID")
        app_secret = os.getenv("INSTAGRAM_APP_SECRET")
        user_id = os.getenv("INSTAGRAM_USER_ID")
        access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")

        if not all([app_id, app_secret, user_id, access_token]):
            return None

        return cls(app_id, app_secret, user_id, access_token)


@dataclass
class PublishResult:
    """Result of an Instagram publish attempt."""
    ok: bool
    media_id: Optional[str] = None
    permalink: Optional[str] = None
    error: str = ""


def _create_media_container(config: InstagramConfig, video_path: Path, caption: str) -> str:
    """Create a media container for the Reel. Returns the container ID."""
    import requests

    url = f"{API_BASE}/{config.user_id}/media"
    params = {
        "media_type": "REELS",
        "video_url": f"file://{video_path.absolute()}",  # Will need to be accessible URL
        "caption": caption,
        "access_token": config.access_token,
    }

    # Note: The Meta Graph API requires the video to be accessible via URL.
    # For local files, we need to upload to a temporary location first.
    # This is a simplified version - real implementation needs a file hosting solution.
    raise NotImplementedError(
        "Instagram Reels publishing requires video to be hosted at an accessible URL. "
        "Implement file upload to a CDN or use Meta's resumable upload endpoint."
    )


def _wait_for_processing(config: InstagramConfig, container_id: str, timeout: int = 300) -> bool:
    """Wait for media container to finish processing."""
    import requests

    url = f"{API_BASE}/{container_id}"
    params = {"fields": "status_code", "access_token": config.access_token}

    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(url, params=params, timeout=30)
        raise_for_status(response, "check Instagram media status")
        data = response.json()
        status = data.get("status_code")

        if status == "FINISHED":
            return True
        if status in ("ERROR", "EXPIRED"):
            return False

        time.sleep(5)

    return False


def _publish_media(config: InstagramConfig, container_id: str) -> str:
    """Publish the processed media container. Returns the media ID."""
    import requests

    url = f"{API_BASE}/{config.user_id}/media_publish"
    params = {
        "creation_id": container_id,
        "access_token": config.access_token,
    }

    response = requests.post(url, params=params, timeout=60)
    raise_for_status(response, "publish Instagram media")
    return response.json().get("id", "")


def _get_media_permalink(config: InstagramConfig, media_id: str) -> str:
    """Get the permalink for published media."""
    import requests

    url = f"{API_BASE}/{media_id}"
    params = {"fields": "permalink", "access_token": config.access_token}

    response = requests.get(url, params=params, timeout=30)
    raise_for_status(response, "get Instagram media permalink")
    return response.json().get("permalink", "")


def publish_reel(settings: Settings, result: Result) -> bool:
    """Publish a rendered video to Instagram Reels.

    This is a placeholder implementation. The Meta Graph API requires videos
    to be hosted at a publicly accessible URL. For production use, implement
    file upload to a CDN or use Meta's resumable upload endpoint.

    Returns True if published successfully, False otherwise.
    """
    config = InstagramConfig.from_env()
    if not config:
        log.warning("Instagram not configured (missing INSTAGRAM_* environment variables)")
        return False

    if not result.video or not result.video.exists():
        log.error("No video file to publish")
        return False

    # Build caption from metadata
    meta_path = result.folder / "metadata.json" if result.folder else None
    caption = ""
    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            title = meta.get("title", "")
            description = meta.get("description", "")
            hashtags = meta.get("tags", [])
            caption = f"{title}\n\n{description}\n\n{' '.join(hashtags)}"
        except Exception:
            caption = result.title

    # Truncate caption to Instagram limit (2200 chars)
    caption = caption[:2200]

    log.info("Instagram Reels publishing not fully implemented - requires video hosting")
    log.info(f"Would publish: {result.video.name} with caption: {caption[:100]}...")
    return False


def is_instagram_configured() -> bool:
    """Check if Instagram publishing is configured."""
    return InstagramConfig.from_env() is not None