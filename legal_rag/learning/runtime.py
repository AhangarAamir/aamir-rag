"""Shared agentic learned-knowledge store.

The query agent may search this store. It cannot save into it.
A reviewer approval is the only write path.
"""

from __future__ import annotations

from typing import Any

from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
from agno.run import RunContext

from legal_rag.agents.messages import latest_user_text
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


def _optional_learning_query() -> None:
    """A tool call with no query uses the current user question."""
    from agno.learn.stores.learned_knowledge import LearnedKnowledgeStore

    if getattr(LearnedKnowledgeStore, "_legal_optional_query", False):
        return

    def _install(method_name: str, is_async: bool) -> None:
        original = getattr(LearnedKnowledgeStore, method_name)

        def factory(self, namespace, user_id=None):
            inner = original(self, namespace, user_id)

            def _question(run_context: RunContext, query: str) -> str:
                return (query or "").strip() or latest_user_text(run_context)

            if is_async:

                async def search_learnings(run_context: RunContext, query: str = "", limit: int = 5) -> str:
                    """Search approved lessons. An empty query uses the current user question."""
                    text = _question(run_context, query)
                    if not text:
                        return "No query was provided."
                    return await inner(query=text, limit=limit)

            else:

                def search_learnings(run_context: RunContext, query: str = "", limit: int = 5) -> str:
                    """Search approved lessons. An empty query uses the current user question."""
                    text = _question(run_context, query)
                    if not text:
                        return "No query was provided."
                    return inner(query=text, limit=limit)

            search_learnings.__name__ = "search_learnings"
            return search_learnings

        setattr(LearnedKnowledgeStore, method_name, factory)

    _install("_create_search_learnings_tool", False)
    _install("_create_async_search_learnings_tool", True)
    LearnedKnowledgeStore._legal_optional_query = True


def get_learning_machine() -> LearningMachine:
    global _machine
    _optional_learning_query()
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
