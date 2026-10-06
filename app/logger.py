"""App-wide logging setup.

setup_logging() is called once, at import time of app.main (before the
app object or the lifespan run), so every module can just do
`logger = get_logger(__name__)` at import time without caring about
init order.
"""

import logging
import sys

from app.config import settings

_CONFIGURED = False


def setup_logging() -> None:
    global _CONFIGURED

    if _CONFIGURED:
        return

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Quiet noisy third-party loggers; app.* loggers still follow log_level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
