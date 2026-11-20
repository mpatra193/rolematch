"""Local embedding model via fastembed (ONNX runtime).

ONNX keeps the memory footprint small (no PyTorch), so the app fits comfortably
on free hosting tiers. The model is loaded lazily and cached.
"""
from functools import lru_cache

from app.config import get_settings


@lru_cache
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=get_settings().embedding_model)


def embed(text: str) -> list[float]:
    return next(iter(_model().embed([text]))).tolist()


def embed_many(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [e.tolist() for e in _model().embed(list(texts))]


def job_text(title: str, company: str, description: str, skills: list[str]) -> str:
    """Canonical text used to embed a job posting."""
    return f"{title} at {company}. {description} Skills: {', '.join(skills)}"
