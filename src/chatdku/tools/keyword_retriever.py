"""Keyword retrieval over documents stored in the embedded local Qdrant index."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import PureWindowsPath

from qdrant_client import QdrantClient

from ..config import Settings, get_settings
from ..ingestion.pipeline import qdrant_path
from .retrieval import RetrievedDocument, document_from_payload, format_documents


class KeywordRetriever:
    """Use BM25-style term matching over the Qdrant collection's stored text."""

    def __init__(self, harness_settings: Settings | None = None) -> None:
        self.settings = harness_settings or get_settings()

    def query(self, query: str, top_k: int | None = None, *, source_name: str | None = None) -> list[RetrievedDocument]:
        limit = top_k or self.settings.retrieval_top_k
        terms = _terms(query)
        if not terms:
            return []
        client = QdrantClient(path=str(qdrant_path(self.settings)))
        try:
            if not client.collection_exists(self.settings.vector_collection):
                raise FileNotFoundError("No local index exists. Run ingestion before querying the harness.")
            records, _ = client.scroll(
                collection_name=self.settings.vector_collection,
                with_payload=True,
                with_vectors=False,
                limit=10_000,
            )
            documents = [document_from_payload(str(record.id), record.payload, 0.0) for record in records]
        finally:
            client.close()
        if source_name is not None:
            # Filter before ranking: source names in query text are not a
            # restriction, and a missing source must not select another one.
            fragment = source_name.strip().casefold()
            if not fragment:
                return []
            filenames = sorted({PureWindowsPath(str(document.metadata.get("file_name", ""))).name
                                for document in documents} - {""})
            documents = [document for document in documents if fragment in
                         str(document.metadata.get("file_name", "")).casefold()]
            if not documents and filenames:
                # Control feedback only: Planner catches this error and never
                # registers it as documentary evidence. Bound catalog size.
                catalog = ", ".join(filenames[:20])
                suffix = " (additional filenames omitted)" if len(filenames) > 20 else ""
                raise ValueError(f"No matching source filename. Available filenames: {catalog}{suffix}. "
                                 "Use a distinctive filename fragment or search without a source filter.")
        return _rank(documents, terms)[:limit]

    def __call__(self, keyword_query: str, top_k: int | None = None) -> str:
        """Search the document collection for exact words, names, phrases, and codes."""
        return format_documents(self.query(keyword_query, top_k))


def _terms(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.casefold(), flags=re.UNICODE)


def _rank(documents: list[RetrievedDocument], terms: list[str]) -> list[RetrievedDocument]:
    if not documents:
        return []
    tokenized = [_terms(document.text) for document in documents]
    document_frequency = Counter(term for tokens in tokenized for term in set(tokens))
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    query_terms = Counter(terms)
    ranked: list[RetrievedDocument] = []
    for document, tokens in zip(documents, tokenized):
        frequencies = Counter(tokens)
        score = 0.0
        for term, query_frequency in query_terms.items():
            if term not in frequencies:
                continue
            inverse_frequency = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            numerator = frequencies[term] * 2.5
            denominator = frequencies[term] + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / max(average_length, 1))
            score += query_frequency * inverse_frequency * numerator / denominator
        if score:
            ranked.append(RetrievedDocument(document.node_id, document.text, document.metadata, score))
    return sorted(ranked, key=lambda document: document.score, reverse=True)
