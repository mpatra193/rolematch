from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rolematch:rolematch@localhost:5432/rolematch"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    llm_provider: str = "mock"  # "openai" | "mock"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Optional: point at any OpenAI-compatible endpoint (Groq, OpenRouter, etc.).
    # Empty string -> OpenAI's default base URL.
    openai_base_url: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
