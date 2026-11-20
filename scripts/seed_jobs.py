"""Load data/sample_jobs.json into the DB, computing embeddings.

Run:  python -m scripts.seed_jobs
"""
import json
from pathlib import Path

from app.db import SessionLocal, init_db
from app.models import Job
from app.services.embeddings import embed, job_text

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_jobs.json"


def main() -> None:
    init_db()
    jobs = json.loads(DATA.read_text(encoding="utf-8"))
    db = SessionLocal()
    try:
        inserted = 0
        for j in jobs:
            exists = db.query(Job).filter_by(title=j["title"], company=j["company"]).first()
            if exists:
                continue
            job = Job(**j)
            job.embedding = embed(job_text(job.title, job.company, job.description, job.skills))
            db.add(job)
            inserted += 1
        db.commit()
        print(f"Seeded {inserted} new jobs (total {db.query(Job).count()}).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
