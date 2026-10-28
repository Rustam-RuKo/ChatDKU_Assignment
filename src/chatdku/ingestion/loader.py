"""Discover and load documents from a local directory."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import logging
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader


SUPPORTED_SUFFIXES = frozenset({".docx", ".html", ".htm", ".json", ".md", ".pdf", ".txt"})
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A source file and the metadata recorded with its indexed chunks."""

    path: Path
    relative_path: str
    sha256: str
    size_bytes: int


def discover_files(source_dir: Path) -> list[SourceFile]:
    """Find supported, non-hidden files in a source directory."""
    source_dir = source_dir.resolve()
    logger.info("Looking for supported documents in %s", source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {source_dir}")

    files: list[SourceFile] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.relative_to(source_dir).parts):
            continue
        files.append(
            SourceFile(
                path=path,
                relative_path=path.relative_to(source_dir).as_posix(),
                sha256=_sha256(path),
                size_bytes=path.stat().st_size,
            )
        )

    if not files:
        extensions = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise FileNotFoundError(f"No supported documents found in {source_dir} ({extensions}).")
    logger.info("Located %d supported source file(s)", len(files))
    for source_file in files:
        logger.debug("Located source file: %s (%d bytes)", source_file.relative_path, source_file.size_bytes)
    return files


def load_documents(source_dir: Path) -> tuple[list[Document], list[SourceFile]]:
    """Load source files and attach stable source metadata to each document."""
    source_files = discover_files(source_dir)
    logger.info("Loading %d source file(s) with LlamaIndex", len(source_files))
    by_path = {str(item.path.resolve()): item for item in source_files}

    reader = SimpleDirectoryReader(
        input_files=[str(item.path) for item in source_files],
        filename_as_id=True,
    )
    documents = reader.load_data()
    logger.info("Loaded %d document record(s) from source files", len(documents))
    for document in documents:
        file_path = document.metadata.get("file_path")
        source = by_path.get(str(Path(file_path).resolve())) if file_path else None
        if source:
            document.metadata.update(
                {
                    "source_path": source.relative_path,
                    "source_sha256": source.sha256,
                    "source_size_bytes": source.size_bytes,
                }
            )
    return documents, source_files


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
