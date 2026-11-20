from functools import lru_cache

from app.config import get_settings
from app.llm.base import LLM
from app.llm.mock_client import MockLLM
from app.llm.openai_client import OpenAILLM


@lru_cache
def get_llm() -> LLM:
    settings = get_settings()
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAILLM(settings.openai_api_key, settings.openai_model)
    return MockLLM()
