from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from environment variables prefixed with HRE_ (or a .env file)."""

    model_config = SettingsConfigDict(env_prefix="HRE_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = ""
    llm_api_key: str | None = None  # set HRE_LLM_API_KEY; never commit it


def get_settings() -> Settings:
    return Settings()
