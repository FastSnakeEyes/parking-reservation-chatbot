from enum import StrEnum


class AppEnvironment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TokenCounterKind(StrEnum):
    """How app/ai/token_counting.py counts tokens."""

    TIKTOKEN = "tiktoken"  # exact counts for OpenAI models, computed locally
    APPROXIMATE = "approximate"  # LangChain's fast heuristic
