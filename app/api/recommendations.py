from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Candidate
from app.schemas import Recommendation
from app.services import memory, tools

router = APIRouter(tags=["recommendations"])


@router.get("/candidates/{candidate_id}/recommendations", response_model=list[Recommendation])
def recommendations(candidate_id: int, k: int = 5, db: Session = Depends(get_db)):
    cand = db.get(Candidate, candidate_id)
    if not cand:
        raise HTTPException(404, "Candidate not found")
    prefs = memory.load_preferences(cand)
    return tools.recommend(db, cand, prefs, k=k)
