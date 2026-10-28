"""Command-line interface for local ChatDKU ingestion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChatDKU's local Qdrant document index.")
    parser.add_argument(
        "source_dir",
        type=Path,
        nargs="?",
        help="Directory containing documents. Defaults to CHATDKU_DOCUMENT_DIR.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the existing local collection before indexing.",
    )
    args = parser.parse_args()

    from ..config import get_settings
    from .pipeline import ingest_directory

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for logger_name in ("httpcore", "httpx", "llama_index"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    result = ingest_directory(args.source_dir, replace=args.replace, harness_settings=settings)
    print(
        f"Indexed {result.document_count} documents from {result.source_file_count} files "
        f"in collection {result.collection_name!r}.\n"
        f"Qdrant data: {result.index_path}\nManifest: {result.manifest_path}"
    )


if __name__ == "__main__":
    main()
