"""Initialize the local retrieval environment for ChatDKU."""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core import Settings as LlamaIndexSettings

from .config import Settings, get_settings


logger = logging.getLogger(__name__)


def _local_embedding_model(harness_settings: Settings) -> Any:
    """Create the configured local embedding model without a model service."""
    if not harness_settings.local_embedding_model:
        raise RuntimeError("No local embedding model is configured.")
    logger.info("Using local embedding model: %s", harness_settings.local_embedding_model)

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require llama-index-embeddings-huggingface. "
        ) from exc

    return HuggingFaceEmbedding(
        model_name=harness_settings.local_embedding_model,
        cache_folder=str(harness_settings.cache_dir / "models"),
    )


def _tei_embedding_model(harness_settings: Settings) -> Any:
    """Create the same Text Embeddings Inference client used by the backend."""
    if not harness_settings.embedding_base_url or not harness_settings.tei_embedding_model:
        raise RuntimeError("The Text Embeddings Inference endpoint is not fully configured.")
    logger.info(
        "Using TEI embedding model %s at %s/embed",
        harness_settings.tei_embedding_model,
        harness_settings.embedding_base_url,
    )

    try:
        from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
    except ImportError as exc:
        raise RuntimeError(
            "TEI embeddings require llama-index-embeddings-text-embeddings-inference."
        ) from exc

    return TextEmbeddingsInference(
        model_name=harness_settings.tei_embedding_model,
        base_url=harness_settings.embedding_base_url,
    )


def build_embedding_model(harness_settings: Settings | None = None) -> Any:
    """Build the selected embedding provider."""
    active_settings = harness_settings or get_settings()
    if active_settings.uses_tei_embeddings:
        return _tei_embedding_model(active_settings)
    return _local_embedding_model(active_settings)


def setup(harness_settings: Settings | None = None) -> Settings:
    """Configure LlamaIndex for local ingestion and retrieval."""
    active_settings = harness_settings or get_settings()
    active_settings.ensure_workspace()
    LlamaIndexSettings.embed_model = build_embedding_model(active_settings)
    LlamaIndexSettings.chunk_size = active_settings.chunk_size
    LlamaIndexSettings.chunk_overlap = active_settings.chunk_overlap
    return active_settings


def verify_embedding_provider(harness_settings: Settings | None = None) -> int:
    """Make one embedding request and return its vector dimension.

    This is useful for confirming a local model or TEI endpoint before running
    ingestion. It deliberately performs a real request rather than only
    validating configuration values.
    """
    active_settings = harness_settings or get_settings()
    model = build_embedding_model(active_settings)
    try:
        vector = model.get_text_embedding("ChatDKU embedding health check")
    except Exception as exc:
        provider = active_settings.resolved_embedding_provider
        logger.error("Embedding health check failed for %s provider: %s", provider, exc)
        raise RuntimeError(
            f"Embedding health check failed for provider {provider!r}."
        ) from exc
    dimension = len(vector)
    logger.info("Embedding health check passed; vector dimension: %d", dimension)
    return dimension


__all__ = ["build_embedding_model", "setup", "verify_embedding_provider"]
    
