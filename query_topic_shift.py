"""Drop prior chat from a query turn when the new question is self-contained.

Agno always injects session history and the session summary once
add_history_to_context is on. That leaks the last clause/instrument into a
new, unrelated question. These hooks classify the current message first,
then hide history only for that run. Follow-ups that need pronouns still
see prior turns. Stored session rows are left intact.
"""

from __future__ import annotations

import re
import threading
from textwrap import dedent
from typing import Callable, List, Optional

from agno.run import RunContext
from agno.run.agent import RunInput
from agno.session.agent import AgentSession
from agno.utils.log import log_debug, log_warning
from pydantic import BaseModel, Field

from config import OPENAI_MODEL

_HISTORY_GUARD = "_legal_query_history_guard"

# Phrase-level continuation cues. Bare this/that in a short message is
# handled in looks_like_follow_up so "this Agreement" in a long standalone
# question is not treated as a follow-up.
_FOLLOW_UP_RE = re.compile(
    r"(?i)\b("
    r"the same|same (?:clause|article|contract|document|instrument)|"
    r"as above|as mentioned|you said|you mentioned|"
    r"previous(?:ly)?|earlier|"
    r"follow[- ]up|"
    r"what about|how about|"
    r"that clause|this clause|the clause|"
    r"that article|this article|"
    r"the (?:above|prior|last) (?:answer|clause|article|contract|question)"
    r")\b"
)

_SHORT_ANAPHORA_RE = re.compile(r"(?i)\b(this|that|those|these)\b")

_SOCIAL_RE = re.compile(
    r"(?i)^(hi|hello|hey|thanks|thank you|bye|goodbye|ok|okay)[\s!.]*$"
)

_CLASSIFIER_INSTRUCTIONS = dedent(
    """
    Decide whether a new legal-research question depends on earlier chat.

    depends_on_prior is true only when the current message cannot be
    interpreted or answered correctly without the prior user questions.
    Examples: this, that, the clause, the article, the same, what about,
    continue, or a question that omits the instrument/party/topic already
    established.

    depends_on_prior is false when the current message is self-contained:
    it names its own document, clause, party, heading, or topic, or it is
    otherwise a complete new question. Same domain (oil and gas, PSC, JOA)
    is not a follow-up. Switching to a different clause, contract, or
    subject without referring back is a new question.

    If the current text is enough to search and answer on its own, return
    false. When unsure, prefer false whenever a concrete subject is named.
    """
).strip()


class QueryRelation(BaseModel):
    depends_on_prior: bool = Field(
        ...,
        description=(
            "True only if the current message cannot be answered without "
            "the earlier user questions."
        ),
    )
    reason: str = Field(..., description="Short reason for the decision.")


DependsOnPriorFn = Callable[[str, List[str]], bool]

_classifier_agent = None
_classifier_lock = threading.Lock()


def input_text(run_input: RunInput) -> str:
    raw = run_input.input_content_string() if run_input is not None else ""
    return (raw or "").strip()


def prior_user_texts(session: AgentSession, limit: int = 6) -> List[str]:
    texts: List[str] = []
    for run in session.runs or []:
        run_input = getattr(run, "input", None)
        if run_input is None:
            continue
        if hasattr(run_input, "input_content_string"):
            raw = run_input.input_content_string()
        else:
            raw = getattr(run_input, "input_content", "")
        if isinstance(raw, str) and raw.strip():
            texts.append(raw.strip())
    if limit <= 0:
        return texts
    return texts[-limit:]


def looks_like_follow_up(message: str) -> bool:
    text = (message or "").strip()
    if not text or _SOCIAL_RE.match(text):
        return False
    if _FOLLOW_UP_RE.search(text):
        return True
    words = text.split()
    return len(words) <= 8 and _SHORT_ANAPHORA_RE.search(text) is not None


def _get_classifier_agent():
    global _classifier_agent
    if _classifier_agent is None:
        from agno.agent import Agent
        from agno.models.openai import OpenAIResponses

        _classifier_agent = Agent(
            id="query-topic-shift",
            name="Query Topic Shift Classifier",
            model=OpenAIResponses(id=OPENAI_MODEL),
            output_schema=QueryRelation,
            instructions=_CLASSIFIER_INSTRUCTIONS,
            markdown=False,
        )
    return _classifier_agent


def classify_depends_on_prior(current: str, prior: List[str]) -> bool:
    """LLM fallback when heuristics do not already mark a follow-up."""
    numbered = "\n".join(f"{i}. {item}" for i, item in enumerate(prior, start=1))
    prompt = (
        f"Earlier user questions in this chat:\n{numbered}\n\n"
        f"Current user message:\n{current}\n"
    )
    with _classifier_lock:
        response = _get_classifier_agent().run(prompt)
    content = response.content
    if isinstance(content, QueryRelation):
        return content.depends_on_prior
    if isinstance(content, dict):
        return bool(content.get("depends_on_prior"))
    if isinstance(content, str):
        lowered = content.strip().lower()
        if "true" in lowered and "false" not in lowered:
            return True
        parsed = QueryRelation.model_validate_json(content)
        return parsed.depends_on_prior
    raise RuntimeError(f"Unexpected classifier content: {content!r}")


def should_isolate_from_history(
    current: str,
    prior: List[str],
    *,
    classify: Optional[DependsOnPriorFn] = None,
) -> bool:
    """True when this turn must not see prior chat."""
    current = (current or "").strip()
    if not current or not prior:
        return False
    if _SOCIAL_RE.match(current):
        return False
    if looks_like_follow_up(current):
        return False
    classify_fn = classify or classify_depends_on_prior
    try:
        depends = classify_fn(current, prior)
    except Exception as exc:
        log_warning(f"Topic-shift classifier failed; treating as standalone: {exc}")
        depends = False
    return not depends


def hide_prior_context(session: AgentSession) -> None:
    if getattr(session, _HISTORY_GUARD, None):
        return
    setattr(
        session,
        _HISTORY_GUARD,
        {
            "get_messages": session.get_messages,
            "summary": session.summary,
        },
    )
    session.get_messages = lambda *args, **kwargs: []  # type: ignore[method-assign]
    session.summary = None
    log_debug("Hiding session history for a standalone query turn")


def restore_prior_context(session: AgentSession) -> None:
    guard = getattr(session, _HISTORY_GUARD, None)
    if not guard:
        return
    session.get_messages = guard["get_messages"]
    session.summary = guard["summary"]
    delattr(session, _HISTORY_GUARD)
    log_debug("Restored session history after standalone query turn")


def isolate_unrelated_query(
    run_input: RunInput,
    session: AgentSession,
    run_context: RunContext,
    classify: Optional[DependsOnPriorFn] = None,
) -> None:
    """Pre-hook: hide history when the current question is not a follow-up."""
    current = input_text(run_input)
    prior = prior_user_texts(session)
    isolate = should_isolate_from_history(current, prior, classify=classify)
    state = run_context.session_state
    if state is None:
        state = {}
        run_context.session_state = state
    state["standalone_query"] = isolate
    if isolate:
        hide_prior_context(session)


def restore_query_history(session: AgentSession) -> None:
    """Post-hook: put history back before session summary and persist."""
    restore_prior_context(session)
