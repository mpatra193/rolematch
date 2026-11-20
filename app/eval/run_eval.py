"""Offline evaluation of the recommendation pipeline against a labeled query set.

Reads data/eval_set.json: a list of {candidate, preferences, relevant_titles}.
For each case it seeds an in-request candidate, runs the pipeline over the jobs
already in the DB, and compares the ranked job titles to the relevant set.

Run (jobs must be seeded first):  python -m app.eval.run_eval
"""
import json
from pathlib import Path
from statistics import mean

from app.db import SessionLocal, init_db
from app.eval.metrics import evaluate
from app.models import Candidate
from app.schemas import Preferences
from app.services import tools

KS = [1, 3, 5]
EVAL_FILE = Path(__file__).resolve().parents[2] / "data" / "eval_set.json"


def _transient_candidate(spec: dict) -> Candidate:
    c = spec.get("candidate", {})
    return Candidate(
        id=-1,
        name=c.get("name", "Eval"),
        raw_resume="",
        title=c.get("title"),
        years_experience=c.get("years_experience"),
        skills=c.get("skills", []),
        summary=c.get("summary"),
        preferences={},
    )


def main() -> None:
    init_db()
    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    db = SessionLocal()
    per_case, aggregate = [], []
    try:
        for spec in cases:
            cand = _transient_candidate(spec)
            prefs = Preferences(**spec.get("preferences", {}))
            recs = tools.recommend(db, cand, prefs, k=max(KS), pool=50)
            ranked = [r.job.title for r in recs]
            relevant = set(spec["relevant_titles"])
            scores = evaluate(ranked, relevant, KS)
            per_case.append({"name": cand.name, "ranked": ranked, **scores})
            aggregate.append(scores)
    finally:
        db.close()

    print("=== Per-case ===")
    for row in per_case:
        print(f"\n{row['name']}")
        print("  ranked:", row["ranked"])
        for k in KS:
            print(
                f"  @{k}: P={row[f'precision@{k}']}  "
                f"R={row[f'recall@{k}']}  NDCG={row[f'ndcg@{k}']}"
            )

    print("\n=== Aggregate (mean over cases) ===")
    if aggregate:
        keys = aggregate[0].keys()
        for key in keys:
            print(f"  {key}: {round(mean(a[key] for a in aggregate), 4)}")


if __name__ == "__main__":
    main()
