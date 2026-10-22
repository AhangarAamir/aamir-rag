"""Ingest agent: loads files into the shared knowledge base.

Uses Agno KnowledgeManagementTools (ingest_path / list_content / ingest_status).
Does not answer legal questions. Each ingest_path run writes a timestamped
failure log under data/import_logs.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.tools.knowledge import KnowledgeManagementTools

from legal_rag.config import OPENAI_MODEL, SESSIONS_DB_FILE
from legal_rag.knowledge.import_log import (
    attach_log_path,
    failed_files_from_folder,
    write_import_log,
)
from legal_rag.knowledge.store import get_knowledge
from legal_rag.observability.adapter import attach_observability
from legal_rag.prompts import INGEST_INSTRUCTIONS


class LoggingKnowledgeManagementTools(KnowledgeManagementTools):
    """Folder ingest that writes one log file per run."""

    def _record_import(self, run_context: Optional[RunContext], path: str, started: datetime, result: str) -> str:
        finished = datetime.now().astimezone()
        try:
            report = json.loads(result)
        except json.JSONDecodeError:
            report = {"ok": False, "error": result, "status": "failed"}
        if not isinstance(report, dict):
            report = {"ok": False, "error": result, "status": "failed"}

        failures: list[dict[str, str]] = []
        site_id = report.get("site_id")
        if site_id:
            try:
                owner = self._owner_id(run_context)
                failures = failed_files_from_folder(self.knowledge, str(site_id), owner)
            except Exception as exc:
                failures = [{"path": path, "error": f"Could not read failure rows: {exc}"}]
        if not failures and report.get("ok") is False:
            failures = [
                {
                    "path": path,
                    "error": str(report.get("error") or report.get("status_message") or "ingest failed"),
                }
            ]
        loaded = report.get("pages")
        if not isinstance(loaded, int):
            loaded = 0
        status = str(report.get("status") or ("failed" if report.get("ok") is False else "completed"))
        note = ""
        if report.get("ok") is False and not site_id:
            note = str(report.get("error") or "")
        log_path = write_import_log(
            source_path=path,
            started_at=started,
            finished_at=finished,
            status=status,
            loaded=loaded,
            failures=failures,
            note=note,
        )
        return attach_log_path(result, log_path)

    def ingest_path(self, run_context: RunContext, path: str) -> str:
        """Ingest a local file or folder into the knowledge base.

        A folder lands one row per file (nested folders included) under a folder row;
        re-running refreshes files whose content changed, retries failed ones, and
        removes rows for files deleted from the folder. A single file lands as one row.
        Writes one log per run at data/import_logs/import_YYYY-MM-DD_HHMMSS.log,
        including every failed file. The JSON result includes import_log.

        Args:
            path: A file or directory path readable by the server, e.g. /data/product-docs.

        Returns:
            JSON: ok, site_id, name, status, status_message, pages, failed, import_log.
        """
        started = datetime.now().astimezone()
        result = super().ingest_path(run_context, path)
        return self._record_import(run_context, path, started, result)

    async def aingest_path(self, run_context: RunContext, path: str) -> str:
        """Ingest a local file or folder into the knowledge base.

        A folder lands one row per file (nested folders included) under a folder row;
        re-running refreshes files whose content changed, retries failed ones, and
        removes rows for files deleted from the folder. A single file lands as one row.
        Writes one log per run at data/import_logs/import_YYYY-MM-DD_HHMMSS.log,
        including every failed file. The JSON result includes import_log.

        Args:
            path: A file or directory path readable by the server, e.g. /data/product-docs.

        Returns:
            JSON: ok, site_id, name, status, status_message, pages, failed, import_log.
        """
        started = datetime.now().astimezone()
        result = await super().aingest_path(run_context, path)
        return self._record_import(run_context, path, started, result)


def build_ingest_agent() -> Agent:
    knowledge = get_knowledge()
    return attach_observability(Agent(
        id="ingest-agent",
        name="Ingest Agent",
        model=OpenAIResponses(id=OPENAI_MODEL),
        db=SqliteDb(
            id="ingest-sessions-db",
            db_file=str(SESSIONS_DB_FILE),
            session_table="ingest_sessions",
        ),
        tools=[
            LoggingKnowledgeManagementTools(
                knowledge=knowledge,
                ingest_path=True,
                ingest_url=True,
                ingest_text=False,
                remove_content=False,
                instructions=INGEST_INSTRUCTIONS,
                add_instructions=True,
            )
        ],
        search_knowledge=False,
        markdown=True,
        instructions=INGEST_INSTRUCTIONS,
    ))
