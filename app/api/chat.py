from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Candidate
from app.schemas import ChatIn, ChatOut
from app.services import agent

router = APIRouter(tags=["chat"])


@router.post("/candidates/{candidate_id}/chat", response_model=ChatOut)
def chat(candidate_id: int, payload: ChatIn, db: Session = Depends(get_db)):
    cand = db.get(Candidate, candidate_id)
    if not cand:
        raise HTTPException(404, "Candidate not found")
    return agent.handle_message(db, cand, payload.message)
