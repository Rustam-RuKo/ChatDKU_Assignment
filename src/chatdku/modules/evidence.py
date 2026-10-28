"""Separate tool observations from model-written planning text."""

import json
import math

from ..tools.retrieval import RetrievedDocument, serialize_documents


def _documents(observation: str) -> list[RetrievedDocument]:
    """Validate a whole retrieval batch; never silently drop a malformed clause.

    Shape and source identity do not establish truth or question relevance.
    Custom tools must use the same RetrievedDocument serialization contract.
    """
    try:
        records = json.loads(observation)
        if not isinstance(records, list):
            return []
        documents = []
        for record in records:
            if not isinstance(record, dict):
                return []
            node_id, text = record.get("node_id"), record.get("text")
            metadata, score = record.get("metadata"), record.get("score")
            if not all(isinstance(value, str) and value.strip() for value in (node_id, text)):
                return []
            if not isinstance(metadata, dict) or not any(
                isinstance(metadata.get(key), str) and metadata[key].strip()
                for key in ("source_path", "file_name", "url")
            ):
                return []
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                return []
            documents.append(RetrievedDocument(node_id, text, metadata, score))
        return documents
    except (TypeError, ValueError, OverflowError):
        return []


def evidence_from_trajectory(trajectory: str) -> str:
    """Return observation-only JSON; malformed input is not evidence.

    These are tool results, not validated facts. Relevance and conflicting
    source claims still require assessment by the gate and synthesizer.
    """
    try:
        events = json.loads(trajectory)
    except (TypeError, ValueError):
        return "[]"
    if not isinstance(events, list):
        return "[]"
    observations = []
    for event in events:
        if not isinstance(event, dict):
            return "[]"
        name, text = event.get("tool_name"), event.get("observation")
        if not isinstance(name, str) or not isinstance(text, str):
            continue
        if not name.strip() or name == "finish" or not text.strip():
            continue
        documents = _documents(text)
        if not documents:
            continue
        observations.append({"tool_name": name, "observation": serialize_documents(documents)})
    return json.dumps(observations, ensure_ascii=False, indent=2)
