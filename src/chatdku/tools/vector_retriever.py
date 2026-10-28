"""Semantic retrieval from the embedded local Qdrant collection."""

from __future__ import annotations

from ..config import Settings, get_settings
from ..ingestion.pipeline import open_index
from .retrieval import RetrievedDocument, format_documents


class VectorRetriever:
    """Retrieve semantically similar document chunks through LlamaIndex."""

    def __init__(self, harness_settings: Settings | None = None) -> None:
        self.settings = harness_settings or get_settings()

    def query(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        limit = top_k or self.settings.retrieval_top_k
        with open_index(self.settings) as index:
            nodes = index.as_retriever(similarity_top_k=limit).retrieve(query)
        return [
            RetrievedDocument(
                node_id=node.node_id,
                text=node.get_content(),
                metadata=dict(node.metadata),
                score=float(node.score or 0.0),
            )
            for node in nodes
        ]

    def __call__(self, semantic_query: str, top_k: int | None = None) -> str:
        """Search the document collection by meaning and return source-grounded passages."""
        return format_documents(self.query(semantic_query, top_k))
