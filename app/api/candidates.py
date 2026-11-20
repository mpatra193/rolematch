from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Candidate
from app.schemas import CandidateOut
from app.services import resume_parser

router = APIRouter(prefix="/candidates", tags=["candidates"])


@router.post("", response_model=CandidateOut)
async def create_candidate(
    resume: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
):
    """Create a candidate from pasted resume text or an uploaded PDF/txt file."""
    if file is not None:
        data = await file.read()
        if file.filename and file.filename.lower().endswith(".pdf"):
            resume = resume_parser.extract_text_from_pdf(data)
        else:
            resume = data.decode("utf-8", errors="ignore")
    if not resume:
        raise HTTPException(400, "Provide resume text or a file.")

    profile = resume_parser.parse_resume(resume)
    cand = Candidate(
        name=profile.get("name") or "Unknown",
        email=profile.get("email"),
        raw_resume=resume,
        title=profile.get("title"),
        years_experience=profile.get("years_experience"),
        skills=profile.get("skills", []),
        summary=profile.get("summary"),
        preferences={},
    )
    db.add(cand)
    db.commit()
    db.refresh(cand)
    return cand


@router.get("/{candidate_id}", response_model=CandidateOut)
def get_candidate(candidate_id: int, db: Session = Depends(get_db)):
    cand = db.get(Candidate, candidate_id)
    if not cand:
        raise HTTPException(404, "Candidate not found")
    return cand
