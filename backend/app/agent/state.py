from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    category: str | None
    tool_results: list[dict[str, Any]]
    answer: str
