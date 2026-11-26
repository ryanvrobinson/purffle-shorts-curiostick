from purffle_shorts.instagram import InstagramConfig


def test_suat_config_reads_access_token_directly_from_environment(monkeypatch):
    monkeypatch.setenv("INSTAGRAM_USER_ID", "test-business-account")
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "test-system-user-token")
    monkeypatch.delenv("INSTAGRAM_APP_ID", raising=False)
    monkeypatch.delenv("INSTAGRAM_APP_SECRET", raising=False)

    config = InstagramConfig.from_env()

    assert config == InstagramConfig("test-business-account", "test-system-user-token")


def test_suat_config_requires_account_id_and_access_token(monkeypatch):
    monkeypatch.setenv("INSTAGRAM_USER_ID", "test-business-account")
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)
    assert InstagramConfig.from_env() is None

    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "test-system-user-token")
    monkeypatch.delenv("INSTAGRAM_USER_ID", raising=False)
    assert InstagramConfig.from_env() is None
