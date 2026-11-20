"""Semantic search over jobs using pgvector cosine distance, with SQL-level filters."""
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Job
from app.services.embeddings import embed


@dataclass
class JobFilters:
    min_salary: int | None = None
    locations: list[str] = field(default_factory=list)
    remote_only: bool = False
    max_years: float | None = None  # candidate's experience; drops jobs needing more
    seniority: str | None = None


def _apply_filters(stmt, f: JobFilters):
    if f.min_salary is not None:
        # Keep jobs whose top of band meets the floor (unknown salary is kept).
        stmt = stmt.where((Job.salary_max.is_(None)) | (Job.salary_max >= f.min_salary))
    if f.remote_only:
        stmt = stmt.where(Job.remote.is_(True))
    if f.locations:
        loc = [l.lower() for l in f.locations]
        conds = [Job.location.ilike(f"%{l}%") for l in loc]
        from sqlalchemy import or_

        stmt = stmt.where(or_(Job.remote.is_(True), *conds))
    if f.max_years is not None:
        stmt = stmt.where((Job.min_years.is_(None)) | (Job.min_years <= f.max_years))
    if f.seniority:
        stmt = stmt.where((Job.seniority.is_(None)) | (Job.seniority.ilike(f.seniority)))
    return stmt


def semantic_search(
    db: Session, query: str, filters: JobFilters | None = None, k: int = 10
) -> list[tuple[Job, float]]:
    """Return up to k (job, semantic_score) pairs, best first.

    semantic_score is cosine similarity in [0, 1] derived from pgvector distance.
    """
    qvec = embed(query)
    dist = Job.embedding.cosine_distance(qvec).label("dist")
    stmt = select(Job, dist).where(Job.embedding.is_not(None))
    if filters:
        stmt = _apply_filters(stmt, filters)
    stmt = stmt.order_by(dist).limit(k)
    rows = db.execute(stmt).all()
    # cosine_distance = 1 - cosine_similarity
    return [(job, max(0.0, 1.0 - float(d))) for job, d in rows]
