"""Re-rank semantically retrieved jobs against candidate preferences."""
from app.models import Candidate, Job
from app.schemas import Preferences


def preference_score(job: Job, prefs: Preferences, candidate: Candidate) -> float:
    """Heuristic 0..1 score for how well a job matches stated preferences."""
    score, weight = 0.0, 0.0

    def add(hit: float, w: float):
        nonlocal score, weight
        score += hit * w
        weight += w

    # Location / remote.
    if prefs.remote_ok and job.remote:
        add(1.0, 1.0)
    elif prefs.locations:
        match = any(loc.lower() in (job.location or "").lower() for loc in prefs.locations)
        add(1.0 if (match or job.remote) else 0.0, 1.0)

    # Salary.
    if prefs.min_salary is not None:
        top = job.salary_max or job.salary_min
        if top is not None:
            add(1.0 if top >= prefs.min_salary else 0.0, 1.5)

    # Role keyword overlap.
    if prefs.roles:
        text = f"{job.title} {job.description}".lower()
        hits = sum(1 for r in prefs.roles if r.lower() in text)
        add(min(1.0, hits / len(prefs.roles)), 1.0)

    # Skill overlap with candidate.
    cand_skills = set(s.lower() for s in (candidate.skills or []))
    job_skills = set(s.lower() for s in (job.skills or []))
    if job_skills:
        overlap = len(cand_skills & job_skills) / len(job_skills)
        add(overlap, 1.0)

    return (score / weight) if weight else 0.5


def dealbreaker_violated(job: Job, prefs: Preferences) -> bool:
    text = f"{job.title} {job.company} {job.location} {job.description}".lower()
    return any(db.lower() in text for db in prefs.dealbreakers if db.strip())


def combined_score(semantic: float, preference: float) -> float:
    """Blend semantic relevance with preference fit."""
    return 0.6 * semantic + 0.4 * preference
