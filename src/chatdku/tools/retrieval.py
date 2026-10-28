"""Shared result types and Qdrant payload helpers for retrieval tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from llama_index.core.vector_stores.utils import metadata_dict_to_node


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    node_id: str
    text: str
    metadata: dict[str, Any]
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
        }


def serialize_documents(documents: list[RetrievedDocument]) -> str:
    """Carry existing document fields intact across the string tool boundary."""
    return json.dumps([document.as_dict() for document in documents], ensure_ascii=False)


def document_from_payload(node_id: str, payload: dict[str, Any], score: float) -> RetrievedDocument:
    """Convert the Qdrant payload produced by LlamaIndex back into a document."""
    node = metadata_dict_to_node(payload)
    return RetrievedDocument(
        node_id=node_id,
        text=node.get_content(),
        metadata=dict(node.metadata),
        score=score,
    )


def format_documents(documents: list[RetrievedDocument]) -> str:
    """Format results for use as a DSPy tool observation."""
    if not documents:
        return "No matching documents found."
    blocks = []
    for document in documents:
        source = document.metadata.get("source_path") or document.metadata.get("file_name") or "Unknown source"
        url = document.metadata.get("url", "No URL")
        blocks.append(
            f"Source: {source}\nURL: {url}\nScore: {document.score:.4f}\n\n{document.text}"
        )
    return "\n\n---\n\n".join(blocks)
