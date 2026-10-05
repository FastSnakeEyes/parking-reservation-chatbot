import os
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class AppEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


BASE_DIR = Path(__file__).resolve().parent.parent


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
    max_active_users: int = Field(default=100, gt=0)
    active_user_ttl: int = Field(default=60, gt=0)
    rate_limit_times: int = Field(default=2, gt=0)
    rate_limit_seconds: int = Field(default=10, gt=0)

    history_max_tokens: int = Field(default=3000, gt=0)
    history_keep_tokens: int = Field(default=1500, gt=0)
    history_max_stored: int = Field(default=40, gt=0)
    max_tool_rounds: int = Field(default=8, gt=0)
    token_counter: Literal["tiktoken", "approximate"] = "tiktoken"

    public_contacts: list[str] = Field(default_factory=list)
    use_injection_model: bool = False
    injection_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    spacy_model: str = "en_core_web_sm"

    admin_api_key: SecretStr = SecretStr("local-dev-only-change-me")

    mcp_server_url: str = "http://localhost:9000/mcp"
    mcp_api_key: SecretStr = SecretStr("local-dev-only-change-me")

    graph_checkpoint_path: str = "./data/graph_checkpoints.sqlite"

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


settings = Settings()
