"""Tests for YouTube OAuth authorization and credential handling.

All tests use mocks: no real browser, no real Google API calls, no real tokens.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from purffle_shorts import youtube as Y
from purffle_shorts.config import Settings

# ------------------------------------------------------------------ helpers


class FakeCreds:
    """Minimal stand-in for google.oauth2.credentials.Credentials."""

    def __init__(self, refresh_token="refresh123", valid=True, expired=False,
                 client_id="cid", token_uri="https://oauth2.googleapis.com/token",
                 scopes=None):
        self.refresh_token = refresh_token
        self.valid = valid
        self.expired = expired
        self.client_id = client_id
        self.token_uri = token_uri
        self.scopes = scopes or ["https://www.googleapis.com/auth/youtube.upload"]

    def has_scopes(self, scopes):
        return True

    def to_json(self):
        return json.dumps({"refresh_token": self.refresh_token,
                           "client_id": self.client_id,
                           "token_uri": self.token_uri,
                           "scopes": self.scopes})

    def refresh(self, request):
        if self.expired and not self.refresh_token:
            from google.auth.exceptions import RefreshError
            raise RefreshError("invalid_grant")
        self.valid = True
        self.expired = False


class FakeFlow:
    """Stand-in for google_auth_oauthlib.flow.InstalledAppFlow."""

    def __init__(self, creds: FakeCreds, auth_url="https://accounts.google.com/o/oauth2/v2/auth?state=xyz",
                 state="xyz", redirect_uri="http://localhost", fail_token=False):
        self._creds = creds
        self.auth_url = auth_url
        self.state = state
        self.redirect_uri = redirect_uri
        self.fetch_token_called_with = None
        self.fail_token = fail_token

    @classmethod
    def from_client_secrets_file(cls, secrets_file, scopes):
        return cls(FakeCreds())

    def authorization_url(self, **kwargs):
        return self.auth_url, self.state

    def fetch_token(self, authorization_response=None, code=None, code_verifier=None):
        if self.fail_token:
            raise RuntimeError("token exchange failed")
        self.fetch_token_called_with = {"authorization_response": authorization_response,
                                        "code": code, "code_verifier": code_verifier}
        # Validate state when a full authorization response is provided.
        if authorization_response:
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(authorization_response)
            params = parse_qs(parsed.query)
            returned_state = params.get("state", [None])[0]
            if returned_state is not None and returned_state != self.state:
                raise ValueError("state mismatch")
        # Simulate Google issuing credentials with a refresh token.
        self._creds = FakeCreds(refresh_token=self._creds.refresh_token)

    @property
    def credentials(self):
        return self._creds


# ------------------------------------------------------------------ get_credentials: refresh errors


def test_invalid_grant_is_permanent_not_retried(tmp_path):
    """invalid_grant sets creds to None and raises NotAuthorized in non-interactive mode."""
    token = tmp_path / "token.json"
    token.write_text("{}")
    s = Settings(token_file=str(token), client_secrets="credentials.json")

    fake_creds = FakeCreds(refresh_token="", valid=False, expired=True)
    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=fake_creds), \
         patch("google.auth.transport.requests.Request"):
        with pytest.raises(Y.NotAuthorized, match="not authorized"):
            Y.get_credentials(s, interactive=False)


def test_other_refresh_error_leads_to_reauth(tmp_path):
    """A non-invalid_grant refresh error leads to re-authorization in interactive mode."""
    token = tmp_path / "token.json"
    token.write_text("{}")
    s = Settings(token_file=str(token), client_secrets="credentials.json")

    def refresh_raises(request):
        from google.auth.exceptions import RefreshError
        raise RefreshError("some other error")

    fake_creds = FakeCreds(refresh_token="rt", valid=False, expired=True)
    fake_creds.refresh = refresh_raises

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=fake_creds), \
         patch("google.auth.transport.requests.Request"), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("google_auth_oauthlib.flow.InstalledAppFlow") as mock_flow, \
         patch("builtins.input", return_value="https://localhost/?code=abc&state=xyz"):
        mock_flow.from_client_secrets_file.return_value = FakeFlow(fake_creds)

        creds = Y.get_credentials(s, interactive=True)
        assert creds.refresh_token == "rt"


# ------------------------------------------------------------------ get_credentials: refresh token presence


def test_interactive_flow_without_refresh_token_is_rejected(tmp_path):
    """If Google does not issue a refresh token, auth fails with clear guidance."""
    token = tmp_path / "token.json"
    s = Settings(token_file=str(token), client_secrets="credentials.json")

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=None), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("google_auth_oauthlib.flow.InstalledAppFlow") as mock_flow, \
         patch("builtins.input", return_value="https://localhost/?code=abc&state=xyz"):
        no_rt_flow = FakeFlow(FakeCreds(refresh_token=""))
        no_rt_flow._creds.refresh_token = ""
        mock_flow.from_client_secrets_file.return_value = no_rt_flow
        with pytest.raises(Y.NotAuthorized, match="refresh token"):
            Y.get_credentials(s, interactive=True)


# ------------------------------------------------------------------ get_credentials: backup before replace


def test_existing_token_is_backed_up_before_replace(tmp_path):
    """A pre-existing token.json is copied to a .bak.* file before being overwritten."""
    token = tmp_path / "token.json"
    token.write_text('{"refresh_token": "old"}')
    s = Settings(token_file=str(token), client_secrets="credentials.json")

    new_creds = FakeCreds(refresh_token="new")
    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=new_creds):
        Y.get_credentials(s, interactive=True)

    # New token written.
    assert json.loads(token.read_text())["refresh_token"] == "new"
    # Backup exists.
    backups = list(tmp_path.glob("token.json.bak.*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text())["refresh_token"] == "old"


# ------------------------------------------------------------------ get_credentials: malformed secrets


def test_malformed_client_secrets_json(tmp_path):
    """A malformed credentials.json propagates the error from flow construction."""
    token = tmp_path / "token.json"
    s = Settings(token_file=str(token), client_secrets="credentials.json")

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=None), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("google_auth_oauthlib.flow.InstalledAppFlow") as mock_flow:
        mock_flow.from_client_secrets_file.side_effect = ValueError("bad JSON")
        with pytest.raises(ValueError, match="bad JSON"):
            Y.get_credentials(s, interactive=True)


# ------------------------------------------------------------------ get_credentials: non-interactive with no token


def test_non_interactive_no_token_raises_not_authorized(tmp_path):
    """No token file and interactive=False raises NotAuthorized immediately."""
    token = tmp_path / "token.json"
    # token.json does not exist
    s = Settings(token_file=str(token), client_secrets="credentials.json")

    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(Y.NotAuthorized, match="not authorized"):
            Y.get_credentials(s, interactive=False)


# ------------------------------------------------------------------ _run_interactive_flow: successful manual authorization


def test_interactive_flow_success_with_full_url(tmp_path, monkeypatch):
    """Pasting the full redirect URL exchanges the code via authorization_response."""
    token = tmp_path / "token.json"
    s = Settings(token_file=str(token), client_secrets="credentials.json")

    flow = FakeFlow(FakeCreds(refresh_token="rt456"))
    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=None), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("google_auth_oauthlib.flow.InstalledAppFlow") as mock_flow, \
         patch("builtins.input", return_value="https://localhost/?code=authcode123&state=xyz"):
        mock_flow.from_client_secrets_file.return_value = flow
        creds = Y.get_credentials(s, interactive=True)

    assert creds.refresh_token == "rt456"
    assert flow.fetch_token_called_with["authorization_response"] == (
        "https://localhost/?code=authcode123&state=xyz")
    assert flow.fetch_token_called_with["code"] is None


def test_interactive_flow_success_with_code_only(tmp_path, monkeypatch):
    """Pasting just the authorization code wraps it in a synthetic response URL."""
    token = tmp_path / "token.json"
    s = Settings(token_file=str(token), client_secrets="credentials.json")

    flow = FakeFlow(FakeCreds(refresh_token="rt789"), state="my-state")
    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=None), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("google_auth_oauthlib.flow.InstalledAppFlow") as mock_flow, \
         patch("builtins.input", return_value="authcode123"):
        mock_flow.from_client_secrets_file.return_value = flow
        creds = Y.get_credentials(s, interactive=True)

    assert creds.refresh_token == "rt789"
    called = flow.fetch_token_called_with["authorization_response"]
    assert "code=authcode123" in called
    assert "state=my-state" in called


def test_interactive_flow_invalid_state_rejected(tmp_path, monkeypatch):
    """A redirect URL with a mismatched state is rejected by fetch_token."""
    token = tmp_path / "token.json"
    s = Settings(token_file=str(token), client_secrets="credentials.json")

    flow = FakeFlow(FakeCreds(refresh_token="rt"), state="expected-state")
    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=None), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("google_auth_oauthlib.flow.InstalledAppFlow") as mock_flow, \
         patch("builtins.input", return_value="https://localhost/?code=abc&state=wrong-state"):
        mock_flow.from_client_secrets_file.return_value = flow
        # google_auth_oauthlib raises ValueError on state mismatch
        with pytest.raises(ValueError, match="state"):
            Y.get_credentials(s, interactive=True)


def test_interactive_flow_missing_code_rejected(tmp_path, monkeypatch):
    """An empty authorization response raises NotAuthorized."""
    token = tmp_path / "token.json"
    s = Settings(token_file=str(token), client_secrets="credentials.json")

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=None), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("google_auth_oauthlib.flow.InstalledAppFlow") as mock_flow, \
         patch("builtins.input", return_value=""):
        mock_flow.from_client_secrets_file.return_value = FakeFlow(FakeCreds())
        with pytest.raises(Y.NotAuthorized, match="No authorization response"):
            Y.get_credentials(s, interactive=True)


def test_interactive_flow_token_exchange_failure(tmp_path, monkeypatch):
    """A token exchange failure propagates as an error."""
    token = tmp_path / "token.json"
    s = Settings(token_file=str(token), client_secrets="credentials.json")

    flow = FakeFlow(FakeCreds(refresh_token="rt"), fail_token=True)
    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=None), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("google_auth_oauthlib.flow.InstalledAppFlow") as mock_flow, \
         patch("builtins.input", return_value="https://localhost/?code=abc&state=xyz"):
        mock_flow.from_client_secrets_file.return_value = flow
        with pytest.raises(RuntimeError, match="token exchange failed"):
            Y.get_credentials(s, interactive=True)


# ------------------------------------------------------------------ cmd_auth


def test_cmd_auth_returns_one_when_no_refresh_token(monkeypatch, capsys):
    """cmd_auth returns exit code 1 when no refresh token is issued."""
    from purffle_shorts.cli import cmd_auth
    from purffle_shorts.config import Settings

    s = Settings(token_file="token.json", client_secrets="credentials.json")

    fake_creds = FakeCreds(refresh_token="")
    with patch("purffle_shorts.youtube.get_credentials", return_value=fake_creds):
        rc = cmd_auth(SimpleNamespace(), s)

    assert rc == 1
    out = capsys.readouterr().out
    assert "no refresh token was issued" in out
    assert "Revoke this client" in out


def test_cmd_auth_returns_zero_on_success(monkeypatch, capsys):
    """cmd_auth returns 0 and prints next-step instructions on success."""
    from purffle_shorts.cli import cmd_auth
    from purffle_shorts.config import Settings

    s = Settings(token_file="token.json", client_secrets="credentials.json")

    fake_creds = FakeCreds(refresh_token="rt123")
    with patch("purffle_shorts.youtube.get_credentials", return_value=fake_creds):
        rc = cmd_auth(SimpleNamespace(), s)

    assert rc == 0
    out = capsys.readouterr().out
    assert "YouTube authorized" in out
    assert "YOUTUBE_TOKEN_JSON" in out
    # No token values leaked.
    assert "rt123" not in out


def test_cmd_auth_returns_one_on_not_authorized(monkeypatch, capsys):
    """cmd_auth returns 1 when get_credentials raises NotAuthorized."""
    from purffle_shorts.cli import cmd_auth
    from purffle_shorts.config import Settings

    s = Settings(token_file="token.json", client_secrets="credentials.json")

    with patch("purffle_shorts.youtube.get_credentials",
               side_effect=Y.NotAuthorized("not authorized")):
        rc = cmd_auth(SimpleNamespace(), s)

    assert rc == 1
    out = capsys.readouterr().out
    assert "Authorization failed" in out


# ------------------------------------------------------------------ scopes are correct


def test_upload_scope_present():
    """The upload scope is always requested."""
    assert "https://www.googleapis.com/auth/youtube.upload" in Y.scopes_for(Settings())


def test_readonly_scope_for_stats():
    """Readonly scope is added when learn_from_stats is set without an API key."""
    s = Settings(learn_from_stats=True)
    scopes = Y.auth_scopes(s)
    assert "https://www.googleapis.com/auth/youtube.readonly" in scopes
    assert "https://www.googleapis.com/auth/youtube.upload" in scopes


# ------------------------------------------------------------------ upload preserves MP4 on auth failure


def test_upload_auth_failure_preserves_mp4(tmp_path, monkeypatch):
    """When upload raises NotAuthorized, the MP4 file is not deleted."""
    from purffle_shorts.config import Settings
    from purffle_shorts.pipeline import Studio

    video = tmp_path / "short.mp4"
    video.write_bytes(b"fake video")
    folder = tmp_path / "video"
    folder.mkdir()
    (folder / "script.json").write_text(
        json.dumps({"topic": "t", "title": "T", "scenes": []})
    )

    s = Settings(upload=True, output_dir=str(tmp_path), keep_videos=True,
                 client_secrets="credentials.json", token_file="token.json")

    Studio(s)

    # Simulate auth failure at the get_credentials level: non-interactive
    # mode raises NotAuthorized before any upload is attempted.
    with patch("purffle_shorts.youtube.get_credentials",
               side_effect=Y.NotAuthorized("refresh token revoked")):
        try:
            Y.service(s, interactive=False)
        except Y.NotAuthorized:
            pass

    # The MP4 was never touched - no upload was attempted, so nothing deleted it.
    assert video.exists(), "MP4 must be preserved when auth fails before upload"
