"""app/lifespan.py's _load_injection_classifier: an enabled classifier that
can't load stops startup with a clear error, instead of failing every chat."""

import pytest
from app import lifespan
from app.config import settings


def test_disabled_classifier_is_not_loaded(monkeypatch):
    monkeypatch.setattr(settings, "use_injection_model", False)
    monkeypatch.setattr(lifespan, "get_injection_classifier", lambda: pytest.fail("must not load"))

    lifespan._load_injection_classifier()


def test_enabled_classifier_without_its_backend_fails_startup_clearly(monkeypatch):
    def missing_backend():
        raise ImportError("PyTorch is not installed")

    monkeypatch.setattr(settings, "use_injection_model", True)
    monkeypatch.setattr(lifespan, "get_injection_classifier", missing_backend)

    with pytest.raises(RuntimeError, match=r"USE_INJECTION_MODEL=true.*PyTorch"):
        lifespan._load_injection_classifier()


def test_enabled_classifier_is_loaded_at_startup(monkeypatch):
    loaded = []
    monkeypatch.setattr(settings, "use_injection_model", True)
    monkeypatch.setattr(lifespan, "get_injection_classifier", lambda: loaded.append(True))

    lifespan._load_injection_classifier()

    assert loaded == [True]
