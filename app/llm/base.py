from abc import ABC, abstractmethod


class LLM(ABC):
    """Minimal interface the app depends on. Swap providers without touching services."""

    available: bool = True

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str:
        """Return a plain-text completion."""

    @abstractmethod
    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """Return {"content": str|None, "tool_calls": [{"name","arguments"}]}."""
