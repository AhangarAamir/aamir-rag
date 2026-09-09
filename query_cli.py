"""CLI for the query agent: `uv run python query_cli.py`."""

from __future__ import annotations

import sys

from config import QUERY_SESSION_ID
from query_agent import build_query_agent


def main() -> None:
    agent = build_query_agent()
    session_kw = {"session_id": QUERY_SESSION_ID}
    if len(sys.argv) > 1:
        agent.print_response(" ".join(sys.argv[1:]), stream=True, **session_kw)
        return

    print("Query agent ready. Knowledge is used as a tool (think / search / analyze).")
    print(f"Session: {QUERY_SESSION_ID} (summary + last N turns persisted)")
    print("Ctrl-C to exit.\n")
    while True:
        try:
            message = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message or message.lower() in {"q", "quit", "exit"}:
            return
        agent.print_response(message, stream=True, **session_kw)


if __name__ == "__main__":
    main()
