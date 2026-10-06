import logging

from app.logger import get_logger, setup_logging


def test_get_logger_returns_named_logger():
    logger = get_logger("app.something")

    assert logger.name == "app.something"


def test_setup_logging_is_idempotent():
    # Should not raise or duplicate handlers when called more than once.
    setup_logging()
    handlers_after_first = list(logging.getLogger().handlers)
    setup_logging()

    assert logging.getLogger().handlers == handlers_after_first
