"""Project entry: `uv run python main.py <command>`."""

from __future__ import annotations

import sys


def main() -> None:
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]

    if mode in {"ingest", "in"}:
        from legal_rag.cli.ingest import ingest_main

        ingest_main(rest)
        return

    if mode == "reembed":
        from legal_rag.cli.ingest import reembed_main

        reembed_main(rest)
        return

    if mode == "links":
        from legal_rag.cli.ingest import links_main

        links_main(rest)
        return

    if mode == "status":
        from legal_rag.cli.ingest import status_main

        status_main(rest)
        return

    if mode in {"query", "ask", "q"}:
        from legal_rag.cli.query import main as query_main

        sys.argv = [sys.argv[0], *rest]
        query_main()
        return

    if mode in {"os", "serve", "agentos"}:
        from legal_rag.server import agent_os
        from legal_rag.config import HOST, PORT

        agent_os.serve(app="legal_rag.server:app", reload=True, host=HOST, port=PORT)
        return

    print("Legal RAG (Agno)")
    print("  uv run python main.py ingest [path] [--force] [--no-llm] [--prune] [--workers N]")
    print("  uv run python main.py reembed [--workers N]")
    print("  uv run python main.py links")
    print("  uv run python main.py status")
    print("  uv run python main.py query [question]")
    print("  uv run python main.py os")
    sys.exit(1)


if __name__ == "__main__":
    main()
