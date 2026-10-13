"""app/config.py's production secret check: at API startup in production, the
app refuses the shared-secret defaults published in this repository, keys that
are too short or too repetitive, and one key reused for admin and MCP."""

import pytest
from app.config import Settings, check_production_secrets, is_weak_secret
from app.constants import DEV_ONLY_SECRET

ADMIN_KEY = "4f1c9e2a7b3d8065c1e4"
MCP_KEY = "9b7d2e4c6a1f3085d2b6"


def _settings(**overrides) -> Settings:
    # _env_file=None: only these values and the test env, never a local .env file
    return Settings(_env_file=None, **overrides)


@pytest.mark.parametrize(
    ("value", "weak"),
    [
        (DEV_ONLY_SECRET, True),
        ("short-key", True),  # fewer than 16 characters
        ("aaaaaaaaaaaaaaaaaaaa", True),  # long, but one character repeated
        ("1234123412341234", True),  # only 4 different characters
        ("   " + "x" * 2 + " " * 20, True),  # whitespace doesn't count
        (ADMIN_KEY, False),
        ("  " + ADMIN_KEY + "\n", False),  # surrounding whitespace is ignored
    ],
    ids=["dev-default", "short", "repeated", "low-variety", "whitespace", "random", "random-padded"],
)
def test_is_weak_secret(value, weak):
    assert is_weak_secret(value) is weak


@pytest.mark.parametrize(
    ("admin_key", "mcp_key", "problem"),
    [
        (DEV_ONLY_SECRET, MCP_KEY, "ADMIN_API_KEY is missing, the dev default, or too weak"),
        (ADMIN_KEY, "aaaaaaaaaaaaaaaaaaaa", "MCP_API_KEY is missing, the dev default, or too weak"),
        (ADMIN_KEY, ADMIN_KEY, "ADMIN_API_KEY and MCP_API_KEY must be different"),
    ],
    ids=["weak-admin-key", "weak-mcp-key", "same-key-for-both"],
)
def test_production_refuses_weak_or_shared_secrets(admin_key, mcp_key, problem):
    config = _settings(app_env="production", admin_api_key=admin_key, mcp_api_key=mcp_key)

    with pytest.raises(RuntimeError, match=problem):
        check_production_secrets(config)


def test_production_accepts_real_distinct_secrets():
    check_production_secrets(_settings(app_env="production", admin_api_key=ADMIN_KEY, mcp_api_key=MCP_KEY))


def test_settings_load_even_with_weak_production_secrets():
    """Only API startup checks the keys, so migrations and ingestion scripts -
    which load the same settings but never use these keys - still run."""
    config = _settings(app_env="production", admin_api_key=DEV_ONLY_SECRET, mcp_api_key=DEV_ONLY_SECRET)

    assert config.app_env == "production"


def test_development_keeps_the_convenient_defaults(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)  # set by tests/conftest.py
    monkeypatch.delenv("MCP_API_KEY", raising=False)

    config = _settings(app_env="development")

    check_production_secrets(config)  # not production: nothing to enforce
    assert config.admin_api_key.get_secret_value() == DEV_ONLY_SECRET
