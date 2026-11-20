from pydantic import BaseModel, Field


class Preferences(BaseModel):
    locations: list[str] = Field(default_factory=list)
    remote_ok: bool | None = None
    min_salary: int | None = None
    roles: list[str] = Field(default_factory=list)
    seniority: str | None = None
    dealbreakers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CandidateOut(BaseModel):
    id: int
    name: str
    email: str | None = None
    title: str | None = None
    years_experience: float | None = None
    skills: list[str] = []
    summary: str | None = None
    preferences: dict = {}

    class Config:
        from_attributes = True


class JobIn(BaseModel):
    title: str
    company: str
    location: str = ""
    remote: bool = False
    description: str = ""
    skills: list[str] = []
    salary_min: int | None = None
    salary_max: int | None = None
    min_years: float | None = None
    seniority: str | None = None


class JobOut(JobIn):
    id: int

    class Config:
        from_attributes = True


class Recommendation(BaseModel):
    job: JobOut
    score: float
    semantic_score: float
    preference_score: float
    explanation: str


class ChatIn(BaseModel):
    message: str


class ChatOut(BaseModel):
    reply: str
    preferences: dict
    tool_calls: list[dict] = []
    recommendations: list[Recommendation] = []
