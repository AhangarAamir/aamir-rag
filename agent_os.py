"""Launcher: `uv run python agent_os.py`."""

from legal_rag.config import HOST, PORT
from legal_rag.server import agent_os, app

__all__ = ["agent_os", "app"]

if __name__ == "__main__":
    agent_os.serve(app="legal_rag.server:app", reload=True, host=HOST, port=PORT)
