from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

settings = get_settings()


def _normalize(url: str) -> str:
    """Accept a plain postgres URL (e.g. from Neon/Supabase) and ensure the
    psycopg v3 driver is used, so DATABASE_URL can be pasted as given."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


engine = create_engine(_normalize(settings.database_url), pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create the pgvector extension and all tables. Idempotent."""
    from app.models import Base  # noqa: WPS433 (avoid circular import at module load)

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)


def seed_jobs_if_empty() -> int:
    """Load sample jobs (with embeddings) if the jobs table is empty.

    Lets a fresh deployment come up with data to demo. Returns rows inserted.
    """
    import json
    from pathlib import Path

    from app.models import Job
    from app.services.embeddings import embed, job_text

    data_file = Path(__file__).resolve().parents[1] / "data" / "sample_jobs.json"
    if not data_file.exists():
        return 0
    db = SessionLocal()
    try:
        if db.query(Job).count() > 0:
            return 0
        inserted = 0
        for j in json.loads(data_file.read_text(encoding="utf-8")):
            job = Job(**j)
            job.embedding = embed(job_text(job.title, job.company, job.description, job.skills))
            db.add(job)
            inserted += 1
        db.commit()
        return inserted
    finally:
        db.close()
