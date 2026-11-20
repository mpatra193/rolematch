"""LLM-generated, grounded explanation for why a job is recommended."""
from app.llm import get_llm
from app.models import Candidate, Job
from app.schemas import Preferences

_PROMPT = """You are a career agent. In 2-3 sentences, explain why this job is
recommended for the candidate. Be specific and honest: cite matching skills,
experience, and preferences, and name one thing the candidate should still check.
Do not invent facts.

Candidate: {title}, {years} yrs. Skills: {cand_skills}.
Preferences: {prefs}

Job: {job_title} at {company} ({location}{remote}). Needs: {job_skills}.
Salary: {salary}. Description: {desc}

Explanation:"""


def _fallback(job: Job, prefs: Preferences, candidate: Candidate) -> str:
    cand = set(s.lower() for s in (candidate.skills or []))
    js = set(s.lower() for s in (job.skills or []))
    shared = sorted(cand & js)
    bits = []
    if shared:
        bits.append(f"matches your {', '.join(shared[:4])}")
    if job.remote and prefs.remote_ok:
        bits.append("it's remote, which you wanted")
    if prefs.min_salary and (job.salary_max or 0) >= prefs.min_salary:
        bits.append("salary clears your floor")
    reason = "; ".join(bits) if bits else "it overlaps with your background"
    return (
        f"{job.title} at {job.company} is recommended because {reason}. "
        f"Check the team's seniority expectations and the exact comp before applying."
    )


def explain(job: Job, prefs: Preferences, candidate: Candidate) -> str:
    llm = get_llm()
    if not llm.available:
        return _fallback(job, prefs, candidate)
    salary = (
        f"${job.salary_min or '?'}-${job.salary_max or '?'}"
        if (job.salary_min or job.salary_max)
        else "not listed"
    )
    try:
        return llm.complete(
            _PROMPT.format(
                title=candidate.title or "candidate",
                years=candidate.years_experience or "?",
                cand_skills=", ".join(candidate.skills or []) or "n/a",
                prefs=prefs.model_dump(),
                job_title=job.title,
                company=job.company,
                location=job.location or "unspecified",
                remote=", remote" if job.remote else "",
                job_skills=", ".join(job.skills or []) or "n/a",
                salary=salary,
                desc=(job.description or "")[:500],
            )
        ).strip()
    except Exception:
        return _fallback(job, prefs, candidate)
