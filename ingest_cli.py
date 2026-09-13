"""CLI for the ingest agent: `uv run python ingest_cli.py`."""

from __future__ import annotations

import sys

from ingest_agent import build_ingest_agent


def main() -> None:
    agent = build_ingest_agent()
    if len(sys.argv) > 1:
        agent.print_response(" ".join(sys.argv[1:]), stream=True)
        return

    print("Ingest agent ready. Prefer: ingest the tree in data/incoming")
    print("Example: ingest all PDFs in data/incoming")
    print("Ctrl-C to exit.\n")
    while True:
        try:
            message = input("ingest> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message or message.lower() in {"q", "quit", "exit"}:
            return
        agent.print_response(message, stream=True)


if __name__ == "__main__":
    main()
