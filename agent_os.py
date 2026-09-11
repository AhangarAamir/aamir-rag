"""AgentOS FastAPI runtime: exposes ingest and query agents, plus the query UI.

Run: `uv run python agent_os.py`  or  `uv run python main.py os`

Build the UI first: `cd ui && npm install && npm run build`
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

from pathlib import Path

from agno.db.sqlite import SqliteDb
from agno.os import AgentOS
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import DATA_DIR, require_api_key
from ingest_agent import build_ingest_agent
from knowledge_store import get_knowledge
from query_agent import build_query_agent

require_api_key()

UI_DIST = Path(__file__).resolve().parent / "ui" / "dist"
UI_INDEX = UI_DIST / "index.html"
UI_ASSETS = UI_DIST / "assets"

base_app = FastAPI(title="Legal RAG Query UI")


@base_app.get("/", include_in_schema=False)
async def query_ui():
    if UI_INDEX.is_file():
        return FileResponse(UI_INDEX)
    return HTMLResponse(
        "<!doctype html><html><body style='font-family:sans-serif;padding:2rem'>"
        "<p>Query UI is not built.</p>"
        "<p>Run <code>npm install &amp;&amp; npm run build</code> in <code>ui/</code>, "
        "then restart AgentOS.</p></body></html>",
        status_code=503,
    )


if UI_ASSETS.is_dir():
    base_app.mount("/assets", StaticFiles(directory=str(UI_ASSETS)), name="ui-assets")
else:

    @base_app.get("/assets/{asset_path:path}", include_in_schema=False)
    async def query_ui_assets(asset_path: str):
        target = (UI_ASSETS / asset_path).resolve()
        if UI_ASSETS.resolve() not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail="UI asset not found")
        return FileResponse(target)


ingest_agent = build_ingest_agent()
query_agent = build_query_agent()
knowledge = get_knowledge()

agent_os = AgentOS(
    id="legal-rag-os",
    name="Legal RAG AgentOS",
    description="Ingest agent writes contracts; query agent answers from knowledge tools.",
    agents=[ingest_agent, query_agent],
    knowledge=[knowledge],
    db=SqliteDb(id="agentos-db", db_file=str(DATA_DIR / "agentos.db")),
    base_app=base_app,
    on_route_conflict="preserve_base_app",
)
app = agent_os.get_app()


if __name__ == "__main__":
    agent_os.serve(app="agent_os:app", reload=True, port=8000)
