from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from environment variables prefixed with HRE_ (or a .env file)."""

    model_config = SettingsConfigDict(env_prefix="HRE_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    embedding_model: str = "intfloat/multilingual-e5-small"


def get_settings() -> Settings:
    return Settings()
