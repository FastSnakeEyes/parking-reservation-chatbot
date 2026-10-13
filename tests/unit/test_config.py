"""app/config.py: in production the app refuses to start with the shared-secret
defaults published in this repository, or with a short key."""

import pytest
from app.config import Settings
from app.constants import DEV_ONLY_SECRET
from pydantic import ValidationError

STRONG = "x" * 32


def _settings(**overrides) -> Settings:
    # _env_file=None: only these values and the test env, never a local .env file
    return Settings(_env_file=None, **overrides)


@pytest.mark.parametrize(
    ("admin_key", "mcp_key", "weak"),
    [
        (DEV_ONLY_SECRET, STRONG, "ADMIN_API_KEY"),
        (STRONG, DEV_ONLY_SECRET, "MCP_API_KEY"),
        ("short-key", STRONG, "ADMIN_API_KEY"),
    ],
    ids=["default-admin-key", "default-mcp-key", "short-admin-key"],
)
def test_production_refuses_weak_secrets(admin_key, mcp_key, weak):
    with pytest.raises(ValidationError, match=f"needs real secrets: set {weak}"):
        _settings(app_env="production", admin_api_key=admin_key, mcp_api_key=mcp_key)


def test_production_accepts_real_secrets():
    settings = _settings(app_env="production", admin_api_key=STRONG, mcp_api_key=STRONG)

    assert settings.admin_api_key.get_secret_value() == STRONG


def test_development_keeps_the_convenient_defaults(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)  # set by tests/conftest.py

    settings = _settings(app_env="development")

    assert settings.admin_api_key.get_secret_value() == DEV_ONLY_SECRET
