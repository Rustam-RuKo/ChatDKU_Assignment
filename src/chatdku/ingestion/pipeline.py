"""Build and open a local Qdrant-backed ChatDKU document index."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterator, Generator

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from ..config import Settings, get_settings
from ..setup import setup, verify_embedding_provider
from .loader import load_documents
from .manifest import write_manifest


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """The reviewable outcome of an ingestion run."""

    collection_name: str
    source_file_count: int
    document_count: int
    index_path: Path
    manifest_path: Path


def ingest_directory(
    source_dir: Path | str | None = None,
    *,
    replace: bool = False,
    harness_settings: Settings | None = None,
) -> IngestionResult:
    """Embed a directory of documents into an embedded local Qdrant index.

    Existing collections are protected by default. Pass ``replace=True`` only
    when you intentionally want to rebuild the collection from its sources.
    """
    active_settings = setup(harness_settings or get_settings())
    source_path = Path(source_dir).expanduser().resolve() if source_dir else active_settings.document_dir
    logger.info(
        "Starting ingestion into collection %r from %s",
        active_settings.vector_collection,
        source_path,
    )
    logger.info("Checking that the configured embedding provider can accept requests")
    dimension = verify_embedding_provider(active_settings)
    logger.info("Embedding provider is reachable; received vectors with %d dimensions", dimension)
    documents, source_files = load_documents(source_path)

    logger.info("Opening embedded Qdrant storage at %s", qdrant_path(active_settings))
    client = _client(active_settings)
    try:
        if client.collection_exists(active_settings.vector_collection):
            if not replace:
                raise FileExistsError(
                    "An index already exists. Pass replace=True to rebuild "
                    f"{active_settings.vector_collection!r}."
                )
            logger.info("Removing existing collection %r before rebuilding", active_settings.vector_collection)
            client.delete_collection(active_settings.vector_collection)

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=active_settings.vector_collection,
            enable_hybrid=False,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info(
            "Embedding and indexing %d document record(s) from %d source file(s)",
            len(documents),
            len(source_files),
        )
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        manifest_path = write_manifest(
            active_settings.index_dir,
            active_settings.vector_collection,
            source_files,
            len(documents),
        )
        logger.info(
            "Ingestion complete: collection %r contains %d document record(s); manifest written to %s",
            active_settings.vector_collection,
            len(documents),
            manifest_path,
        )
        return IngestionResult(
            collection_name=active_settings.vector_collection,
            source_file_count=len(source_files),
            document_count=len(documents),
            index_path=qdrant_path(active_settings),
            manifest_path=manifest_path,
        )
    finally:
        client.close()


@contextmanager
def open_index(harness_settings: Settings | None = None) -> Generator[VectorStoreIndex]:
    """Open an existing local Qdrant collection for the duration of a query."""
    active_settings = setup(harness_settings or get_settings())
    client = _client(active_settings)
    try:
        if not client.collection_exists(active_settings.vector_collection):
            raise FileNotFoundError(
                "No local index exists. Run ingestion before querying the harness."
            )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=active_settings.vector_collection,
            enable_hybrid=False,
        )
        yield VectorStoreIndex.from_vector_store(vector_store=vector_store)
    finally:
        client.close()


def qdrant_path(harness_settings: Settings) -> Path:
    """Return the embedded Qdrant storage directory for the active index."""
    return harness_settings.index_dir / "qdrant"


def _client(harness_settings: Settings) -> QdrantClient:
    return QdrantClient(path=str(qdrant_path(harness_settings)))
