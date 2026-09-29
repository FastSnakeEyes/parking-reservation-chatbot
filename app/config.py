import os
from enum import Enum
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class AppEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


BASE_DIR = Path(__file__).resolve().parent.parent


def selected_env_file() -> Path:
    raw_environment = os.getenv("APP_ENV", AppEnvironment.DEVELOPMENT.value)
    environment = AppEnvironment(raw_environment)
    return BASE_DIR / f".env.{environment.value}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=selected_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    postgres_db: str = "fastapi_db"
    postgres_user: str = "fastapi_user"
    postgres_password: SecretStr = SecretStr("local-dev-only-change-me")
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    redis_host: str = "localhost"
    redis_port: int = 6379

    @property
    def database_url(self) -> str:
        return URL.create(
            "postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        ).render_as_string(hide_password=False)

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"


settings = Settings()