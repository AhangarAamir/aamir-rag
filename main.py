"""Project entry: `uv run python main.py ingest` or `uv run python main.py query`."""

from __future__ import annotations

import sys


def main() -> None:
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]

    if mode in {"ingest", "in"}:
        from ingest_cli import main as ingest_main

        sys.argv = [sys.argv[0], *rest]
        ingest_main()
        return

    if mode in {"query", "ask", "q"}:
        from query_cli import main as query_main

        sys.argv = [sys.argv[0], *rest]
        query_main()
        return

    print("Legal RAG (Agno)")
    print("  uv run python main.py ingest [message]")
    print("  uv run python main.py query [question]")
    print("  uv run python ingest_cli.py")
    print("  uv run python query_cli.py")
    sys.exit(1)


if __name__ == "__main__":
    main()
