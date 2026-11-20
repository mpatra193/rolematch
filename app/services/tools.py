"""Tool/function-calling definitions and the recommendation pipeline they drive."""
from sqlalchemy.orm import Session

from app.models import Candidate
from app.schemas import Preferences, Recommendation, JobOut
from app.services import explanations, ranking
from app.services.embeddings import job_text
from app.services.search import JobFilters, semantic_search

# OpenAI-style tool schema. The mock client emits the same tool name.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "update_preferences_and_search",
            "description": (
                "Update the candidate's remembered preferences, filter jobs by salary, "
                "location, remote and experience, then return ranked recommendations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_salary": {"type": "integer", "description": "Minimum base salary in USD"},
                    "locations": {"type": "array", "items": {"type": "string"}},
                    "remote_only": {"type": "boolean"},
                    "roles": {"type": "array", "items": {"type": "string"}},
                    "seniority": {"type": "string"},
                    "dealbreakers": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }
]


def merge_preferences(prefs: Preferences, args: dict) -> Preferences:
    data = prefs.model_dump()
    if "min_salary" in args and args["min_salary"]:
        data["min_salary"] = args["min_salary"]
    if "remote_only" in args:
        data["remote_ok"] = bool(args["remote_only"])
    if args.get("locations"):
        data["locations"] = sorted(set(data["locations"]) | set(args["locations"]))
    if args.get("roles"):
        data["roles"] = sorted(set(data["roles"]) | set(args["roles"]))
    if args.get("seniority"):
        data["seniority"] = args["seniority"]
    if args.get("dealbreakers"):
        data["dealbreakers"] = sorted(set(data["dealbreakers"]) | set(args["dealbreakers"]))
    return Preferences(**data)


def recommend(
    db: Session, candidate: Candidate, prefs: Preferences, k: int = 5, pool: int = 25
) -> list[Recommendation]:
    """Full pipeline: build query -> semantic retrieve (filtered) -> rank -> explain."""
    query = _query_from(candidate, prefs)
    filters = JobFilters(
        min_salary=prefs.min_salary,
        locations=prefs.locations,
        remote_only=bool(prefs.remote_ok) and not prefs.locations,
        max_years=candidate.years_experience,
        seniority=prefs.seniority,
    )
    retrieved = semantic_search(db, query, filters, k=pool)

    scored: list[Recommendation] = []
    for job, sem in retrieved:
        if ranking.dealbreaker_violated(job, prefs):
            continue
        pref = ranking.preference_score(job, prefs, candidate)
        total = ranking.combined_score(sem, pref)
        scored.append(
            Recommendation(
                job=JobOut.model_validate(job),
                score=round(total, 4),
                semantic_score=round(sem, 4),
                preference_score=round(pref, 4),
                explanation=explanations.explain(job, prefs, candidate),
            )
        )
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:k]


def _query_from(candidate: Candidate, prefs: Preferences) -> str:
    parts = [candidate.title or "", " ".join(candidate.skills or [])]
    parts += prefs.roles + prefs.locations
    if candidate.summary:
        parts.append(candidate.summary)
    return " ".join(p for p in parts if p).strip() or "software engineer"
