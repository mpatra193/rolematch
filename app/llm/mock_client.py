"""Deterministic, offline LLM stand-in.

Parses preferences out of free text with simple rules and emits tool calls so the
agent + tool-calling path is exercised end-to-end without an API key.
"""
import re

from app.llm.base import LLM


class MockLLM(LLM):
    available = False  # signals callers to prefer heuristic extraction paths

    def complete(self, prompt: str, system: str | None = None) -> str:
        # Used for job-match explanations in offline mode.
        return (
            "This role aligns with your stated preferences and experience. "
            "Worth checking the salary band and team before applying."
        )

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        text = messages[-1]["content"].lower() if messages else ""
        args: dict = {}

        sal = re.search(r"\$?\s*(\d{2,3})\s*k", text) or re.search(r"(\d{5,6})", text)
        if sal:
            n = int(sal.group(1))
            args["min_salary"] = n * 1000 if n < 1000 else n
        if "remote" in text:
            args["remote_only"] = True
        for city in ["san francisco", "sf", "new york", "nyc", "london", "berlin", "austin", "seattle"]:
            if city in text:
                args.setdefault("locations", []).append(city)
        roles = [r for r in ["backend", "frontend", "ml", "machine learning", "data", "platform", "senior", "staff"] if r in text]
        if roles:
            args["roles"] = roles

        if args:
            return {
                "content": "Got it — updating your preferences and finding matches.",
                "tool_calls": [{"name": "update_preferences_and_search", "arguments": args}],
            }
        return {
            "content": "Tell me what you want next — role, location, remote, and target salary.",
            "tool_calls": [],
        }
