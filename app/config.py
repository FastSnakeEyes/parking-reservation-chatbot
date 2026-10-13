import os
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

from app.constants import (
    BASE_DIR,
    DEV_ONLY_SECRET,
    MIN_PRODUCTION_SECRET_DISTINCT_CHARS,
    MIN_PRODUCTION_SECRET_LENGTH,
)
from app.enums import AppEnvironment, LogLevel, TokenCounterKind


def selected_env_file() -> Path:
    # APP_ENV must be set before starting Python to select another environment.
    environment = AppEnvironment(os.getenv("APP_ENV", AppEnvironment.DEVELOPMENT.value))
    explicit_file = os.getenv("ENV_FILE")

    if explicit_file:
        path = Path(explicit_file)
        return path if path.is_absolute() else BASE_DIR / path

    return BASE_DIR / f".env.{environment.value}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=selected_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    timezone: str = "Europe/Vilnius"

    # Optional production URL; overrides individual PostgreSQL fields.
    database_url_override: SecretStr | None = Field(
        default=None,
        validation_alias="DATABASE_URL",
    )

    postgres_db: str
    postgres_user: str
    postgres_password: SecretStr
    postgres_host: str
    postgres_port: int = Field(default=5432, ge=1, le=65535)

    redis_url: SecretStr | None = None

    openai_api_key: SecretStr
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    milvus_uri: str = Field(
        default="./milvus_parking.db",
        validation_alias="VECTOR_DB_URI",
    )
    milvus_token: SecretStr = SecretStr("")
    milvus_collection: str = "parking_static"
    retriever_k: int = Field(default=4, ge=1)

    cookie_name: str = "parking_chat"
    cookie_ttl: int = Field(default=1800, gt=0)
    chat_cleanup_interval: int = Field(default=600, gt=0)  # seconds between idle chat thread sweeps
    max_active_users: int = Field(default=100, gt=0)
    active_user_ttl: int = Field(default=60, gt=0)
    rate_limit_times: int = Field(default=2, gt=0)
    rate_limit_seconds: int = Field(default=10, gt=0)

    history_max_tokens: int = Field(default=3000, gt=0)
    history_keep_tokens: int = Field(default=1500, gt=0)
    token_counter: TokenCounterKind = TokenCounterKind.TIKTOKEN

    public_contacts: list[str] = Field(default_factory=list)
    use_injection_model: bool = False
    injection_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    spacy_model: str = "en_core_web_sm"

    admin_api_key: SecretStr = SecretStr(DEV_ONLY_SECRET)

    mcp_server_url: str = "http://localhost:9000/mcp"
    mcp_api_key: SecretStr = SecretStr(DEV_ONLY_SECRET)

    log_level: LogLevel = LogLevel.INFO

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "vilniuspark-chatbot"

    @property
    def database_url(self) -> str:
        if self.database_url_override is not None:
            return self.database_url_override.get_secret_value()

        return URL.create(
            drivername="postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        ).render_as_string(hide_password=False)

    @property
    def database_url_psycopg(self) -> str:
        """Plain postgresql:// DSN for psycopg (LangGraph's AsyncPostgresSaver),
        as opposed to database_url's SQLAlchemy-style postgresql+asyncpg://."""
        if self.database_url_override is not None:
            return self.database_url_override.get_secret_value().replace("postgresql+asyncpg://", "postgresql://")

        return URL.create(
            drivername="postgresql",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        ).render_as_string(hide_password=False)

        weak = [
            name
            for name, secret in (("ADMIN_API_KEY", self.admin_api_key), ("MCP_API_KEY", self.mcp_api_key))
            if secret.get_secret_value() == DEV_ONLY_SECRET
            or len(secret.get_secret_value()) < MIN_PRODUCTION_SECRET_LENGTH
        ]
        if weak:
            raise ValueError(
                f"APP_ENV=production needs real secrets: set {', '.join(weak)} to a unique random value "
                f"of at least {MIN_PRODUCTION_SECRET_LENGTH} characters."
            )
        return self


def is_weak_secret(value: str) -> bool:
    """A shared secret anyone could guess: the repo's published dev default,
    or - ignoring surrounding whitespace - too short or too repetitive.
    mcp_server/server.py mirrors this rule (a test keeps the two in step)."""
    secret = value.strip()
    return (
        secret == DEV_ONLY_SECRET
        or len(secret) < MIN_PRODUCTION_SECRET_LENGTH
        or len(set(secret)) < MIN_PRODUCTION_SECRET_DISTINCT_CHARS
    )


def check_production_secrets(config: Settings) -> None:
    """Called at API startup (app/lifespan.py), not when settings load, so
    migrations and ingestion scripts - which never use these keys - still run.
    In production, refuse weak keys, and one key reused for both the admin API
    and the MCP server (a leak of one would expose both)."""
    if config.app_env != AppEnvironment.PRODUCTION:
        return

    keys = {"ADMIN_API_KEY": config.admin_api_key, "MCP_API_KEY": config.mcp_api_key}
    problems = [
        f"{name} is missing, the dev default, or too weak"
        for name, key in keys.items()
        if is_weak_secret(key.get_secret_value())
    ]
    if config.admin_api_key.get_secret_value().strip() == config.mcp_api_key.get_secret_value().strip():
        problems.append("ADMIN_API_KEY and MCP_API_KEY must be different")

    if problems:
        raise RuntimeError(
            "APP_ENV=production needs real secrets: " + "; ".join(problems) + ". Use unique random values "
            f"(at least {MIN_PRODUCTION_SECRET_LENGTH} characters), e.g. `openssl rand -hex 32`."
        )


settings = Settings()
