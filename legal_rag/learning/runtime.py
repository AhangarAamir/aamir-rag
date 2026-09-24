"""Shared agentic learned-knowledge store.

The query agent may search this store. It cannot save into it.
A reviewer approval is the only write path.
"""

from __future__ import annotations

from typing import Any

from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode

from legal_rag.knowledge.store import get_learnings_knowledge

_machine: LearningMachine | None = None


def learning_config() -> LearnedKnowledgeConfig:
    return LearnedKnowledgeConfig(
        mode=LearningMode.AGENTIC,
        namespace="global",
        agent_can_save=False,
        agent_can_search=True,
    )


def configure_learning_machine(machine: LearningMachine | None) -> None:
    global _machine
    _machine = machine


def get_learning_machine() -> LearningMachine:
    global _machine
    if _machine is None:
        _machine = LearningMachine(
            knowledge=get_learnings_knowledge(),
            learned_knowledge=learning_config(),
        )
    return _machine


def publish_lesson(lesson: dict[str, Any]) -> bool:
    store = get_learning_machine().learned_knowledge_store
    if store is None:
        return False
    tags = [str(tag) for tag in (lesson.get("tags") or [])]
    return bool(
        store.save(
            title=lesson.get("title") or "Approved lesson",
            learning=lesson.get("lesson") or "",
            context=lesson.get("context") or lesson.get("query") or "",
            tags=tags,
            agent_id=lesson.get("agent_id") or "query-agent",
            namespace="global",
        )
    )


def search_lessons(query: str, limit: int = 5) -> list[Any]:
    store = get_learning_machine().learned_knowledge_store
    if store is None or not query.strip():
        return []
    return store.search(query=query, namespace="global", limit=limit)
