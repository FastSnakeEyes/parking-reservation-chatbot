import os

from app.config import settings
from app.lifespan import _configure_langsmith
from pydantic import SecretStr


def test_langsmith_disabled_by_default_sets_no_env_vars(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    _configure_langsmith()

    assert "LANGSMITH_TRACING" not in os.environ


def test_langsmith_enabled_sets_expected_env_vars(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", True)
    monkeypatch.setattr(settings, "langsmith_project", "test-project")
    monkeypatch.setattr(settings, "langsmith_api_key", SecretStr("secret-key"))

    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    try:
        _configure_langsmith()

        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_PROJECT"] == "test-project"
        assert os.environ["LANGSMITH_API_KEY"] == "secret-key"
    finally:
        # _configure_langsmith sets these directly on os.environ, not via
        # monkeypatch, so they need explicit cleanup to avoid leaking into
        # later tests.
        for key in ("LANGSMITH_TRACING", "LANGSMITH_PROJECT", "LANGSMITH_API_KEY"):
            os.environ.pop(key, None)
