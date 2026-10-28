"""Local document ingestion for the ChatDKU agent harness."""

from .pipeline import IngestionResult, ingest_directory, open_index

__all__ = ["IngestionResult", "ingest_directory", "open_index"]
