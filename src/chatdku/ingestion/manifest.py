"""Write a small, reviewable record of each completed ingestion run."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from .loader import SourceFile


MANIFEST_NAME = "manifest.json"


def write_manifest(
    index_dir: Path,
    collection_name: str,
    source_files: Iterable[SourceFile],
    document_count: int,
) -> Path:
    """Atomically persist the source inventory for the local Qdrant index."""
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / MANIFEST_NAME
    payload = {
        "collection": collection_name,
        "created_at": datetime.now(UTC).isoformat(),
        "document_count": document_count,
        "source_files": [
            {
                **asdict(source),
                "path": source.relative_path,
            }
            for source in source_files
        ],
    }
    temporary_path = manifest_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest_path
