"""Conversational agent: remembers preferences, calls tools, returns recommendations."""
from sqlalchemy.orm import Session

from app.llm import get_llm
from app.llm.mock_client import MockLLM
from app.models import Candidate
from app.schemas import ChatOut
from app.services import memory, tools

SYSTEM = (
    "You are RoleMatch, a career agent. You remember the candidate's preferences across "
    "conversations and update them when plans change. Use the "
    "update_preferences_and_search tool whenever the user states a role, location, "
    "remote preference, salary floor, or dealbreaker. Be concise and honest."
)


def handle_message(db: Session, candidate: Candidate, user_message: str) -> ChatOut:
    memory.add_message(db, candidate.id, "user", user_message)
    prefs = memory.load_preferences(candidate)

    llm = get_llm()
    convo = [{"role": "system", "content": SYSTEM}]
    convo += memory.history(db, candidate.id, limit=10)

    degraded = False
    try:
        result = llm.chat_with_tools(convo, tools.TOOL_SCHEMAS)
    except Exception:
        # Provider failed (e.g. no credits / network). Degrade to the offline
        # agent so the app stays usable instead of returning a 500.
        degraded = True
        result = MockLLM().chat_with_tools(convo, tools.TOOL_SCHEMAS)
    reply = result.get("content") or ""
    tool_calls = result.get("tool_calls") or []

    recommendations = []
    executed = []
    for call in tool_calls:
        if call["name"] == "update_preferences_and_search":
            args = call["arguments"]
            prefs = tools.merge_preferences(prefs, args)
            memory.save_preferences(db, candidate, prefs)
            recommendations = tools.recommend(db, candidate, prefs, k=5)
            executed.append({"name": call["name"], "arguments": args})

    if not reply:
        reply = (
            f"Found {len(recommendations)} roles matching your preferences."
            if recommendations
            else "Tell me what you're looking for next."
        )
    if degraded:
        reply += "  (Note: the LLM provider was unavailable, so this reply used the offline agent.)"
    memory.add_message(db, candidate.id, "assistant", reply)

    return ChatOut(
        reply=reply,
        preferences=prefs.model_dump(),
        tool_calls=executed,
        recommendations=recommendations,
    )
