"""Ingest CLI: index files under CORPUS_DIR into data/index.db.

    uv run python main.py ingest [path] [--force] [--no-llm] [--prune] [--workers N]
    uv run python main.py reembed [--workers N]
    uv run python main.py links          # rebuild documents.link after changing base URLs
    uv run python main.py status
"""

from __future__ import annotations

import argparse
import sys


def ingest_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="main.py ingest", description="Index files under the corpus folder.")
    parser.add_argument("path", nargs="?", default="", help="file or folder relative to CORPUS_DIR")
    parser.add_argument("--force", action="store_true", help="re-index unchanged files")
    parser.add_argument("--no-llm", action="store_true", help="skip the per-document profile call")
    parser.add_argument("--prune", action="store_true", help="drop index rows for deleted files")
    parser.add_argument("--workers", type=int, help="files processed in parallel (default 1)")
    args = parser.parse_args(argv)

    from legal_rag.config import require_api_key
    from legal_rag.ingest.pipeline import ingest_path, remove_missing

    require_api_key()

    def progress(event: dict) -> None:
        if event["type"] == "file":
            print(f"[{event['index']}/{event['total']}] {event['file']}", flush=True)

    report = ingest_path(
        args.path or None,
        force=args.force,
        use_llm=False if args.no_llm else None,
        on_progress=progress,
        workers=args.workers,
    )
    removed = report.removed or (remove_missing() if args.prune else [])
    print(
        f"\nadded {len(report.added)} · updated {len(report.updated)} · unchanged {len(report.skipped)}"
        f" · failed {len(report.failed)} · removed {len(removed)} · pages {report.pages} · chunks {report.chunks}"
    )
    for failure in report.failed:
        print(f"  FAILED {failure['file']}: {failure['error']}")
    for warning in report.warnings:
        print(f"  warning: {warning}")
    sys.exit(1 if report.failed else 0)


def reembed_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="main.py reembed", description="Embed chunks with EMBEDDING_MODEL.")
    parser.add_argument("--workers", type=int, help="embedding batches in parallel (default 1)")
    args = parser.parse_args(argv)

    from legal_rag.config import require_api_key
    from legal_rag.ingest.pipeline import reembed

    require_api_key()
    print(reembed(on_progress=lambda e: print(f"embedded {e['done']}/{e['total']}", flush=True), workers=args.workers))


def links_main(argv: list[str]) -> None:
    from legal_rag.index import store

    print(f"updated documents.link for {store.refresh_links()} document(s)")


def status_main(argv: list[str]) -> None:
    from legal_rag.config import CORPUS_DIR, EMBEDDING_MODEL, INDEX_DB, OPENAI_MODEL
    from legal_rag.index import store

    print(f"corpus:     {CORPUS_DIR}")
    print(f"index:      {INDEX_DB}")
    print(f"chat model: {OPENAI_MODEL} (OpenAI Responses)")
    print(f"embeddings: {EMBEDDING_MODEL} (OpenAI)")
    print(f"stats:      {store.stats()}")


if __name__ == "__main__":
    ingest_main(sys.argv[1:])
