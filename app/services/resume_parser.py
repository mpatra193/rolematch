"""Resume -> structured candidate profile.

Uses the LLM when available (provider=openai) for high-quality extraction, and
falls back to a deterministic heuristic parser so the pipeline works offline.
"""
import io
import json
import re

from app.llm import get_llm

# A compact, common skill vocabulary for the heuristic fallback.
SKILL_VOCAB = [
    "python", "fastapi", "django", "flask", "java", "go", "rust", "c++", "typescript",
    "javascript", "react", "node", "sql", "postgresql", "mongodb", "redis", "docker",
    "kubernetes", "aws", "gcp", "azure", "terraform", "pytorch", "tensorflow",
    "machine learning", "nlp", "llm", "embeddings", "pgvector", "pandas", "numpy",
    "spark", "kafka", "graphql", "rest", "ci/cd", "git", "linux",
]

_EXTRACT_PROMPT = """You extract a structured profile from a resume.
Return STRICT JSON with keys: name (string), email (string|null),
title (string, current/most recent role), years_experience (number),
skills (array of lowercase strings), summary (2-sentence string).
Resume:
---
{resume}
---
JSON:"""


def extract_text_from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _heuristic_parse(resume: str) -> dict:
    lower = resume.lower()
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", resume)
    # First non-empty line is usually the name.
    name = next((ln.strip() for ln in resume.splitlines() if ln.strip()), "Unknown")
    years = 0.0
    ymatch = re.search(r"(\d+)\+?\s*years?", lower)
    if ymatch:
        years = float(ymatch.group(1))
    skills = sorted({s for s in SKILL_VOCAB if s in lower})
    # Guess a title from a line containing a common role keyword.
    title = None
    for ln in resume.splitlines():
        if re.search(r"engineer|developer|scientist|manager|designer|analyst", ln, re.I):
            title = ln.strip()[:120]
            break
    return {
        "name": name[:120],
        "email": email_match.group(0) if email_match else None,
        "title": title,
        "years_experience": years,
        "skills": skills,
        "summary": resume.strip()[:280],
    }


def parse_resume(resume: str) -> dict:
    llm = get_llm()
    if llm.available:
        try:
            raw = llm.complete(_EXTRACT_PROMPT.format(resume=resume[:6000]))
            data = json.loads(_strip_json(raw))
            data.setdefault("skills", [])
            data["skills"] = [s.lower() for s in data.get("skills", [])]
            return data
        except Exception:
            pass  # fall through to heuristic on any LLM/JSON failure
    return _heuristic_parse(resume)


def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 else text
