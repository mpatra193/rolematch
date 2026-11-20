from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import candidates, chat, jobs, recommendations
from app.config import get_settings
from app.db import init_db

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="RoleMatch", version="0.1.0", lifespan=lifespan)

app.include_router(candidates.router)
app.include_router(jobs.router)
app.include_router(chat.router)
app.include_router(recommendations.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["meta"])
def health():
    s = get_settings()
    return {
        "status": "ok",
        "llm_provider": s.llm_provider if s.openai_api_key or s.llm_provider == "mock" else "mock",
        "embedding_model": s.embedding_model,
    }
