"""Local embedding model (Sentence Transformers). Loaded lazily and cached."""
from functools import lru_cache

from app.config import get_settings


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed(text: str) -> list[float]:
    return _model().encode(text, normalize_embeddings=True).tolist()


def embed_many(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _model().encode(texts, normalize_embeddings=True).tolist()


def job_text(title: str, company: str, description: str, skills: list[str]) -> str:
    """Canonical text used to embed a job posting."""
    return f"{title} at {company}. {description} Skills: {', '.join(skills)}"
