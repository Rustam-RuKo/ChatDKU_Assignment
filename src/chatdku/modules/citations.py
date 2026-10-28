"""Per-question registry and deterministic citation identity validation."""
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re

from ..tools.retrieval import RetrievedDocument
from .evidence import _documents


class CitationError(ValueError):
    """The draft or retrieved identity cannot be safely rendered."""


def _basename(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).name


class EvidenceInventory:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    def add(self, document: RetrievedDocument) -> str:
        metadata = document.metadata
        source = metadata.get("source_path") or metadata.get("file_name")
        if not isinstance(source, str) or not source.strip():
            raise CitationError("Missing document identity")
        name = _basename(source)
        if not name or name in {".", ".."} or len(name) > 255 or any(ord(c) < 32 for c in name):
            raise CitationError("Malformed document name")
        filename = metadata.get("file_name")
        if filename is not None and (not isinstance(filename, str) or _basename(filename) != name):
            raise CitationError("Inconsistent document name")
        digest = metadata.get("source_sha256")
        if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
            raise CitationError("Malformed document hash")
        page = metadata.get("page_label") if name.lower().endswith(".pdf") else None
        if page is not None and (not isinstance(page, str) or not re.fullmatch(r"(?:[0-9]{1,6}|[ivxlcdmIVXLCDM]{1,12})", page)):
            raise CitationError("Malformed PDF page label")
        identity = digest.lower() if digest else source
        citation = "E" + sha256(json.dumps([identity, document.node_id]).encode()).hexdigest()[:12]
        record = dict(id=citation, node_id=document.node_id, document_id=identity if digest else
                      sha256(identity.encode()).hexdigest(), filename=name, page_label=page, text=document.text)
        if citation in self.records and self.records[citation] != record:
            raise CitationError("Inconsistent record for the same chunk")
        self.records[citation] = record
        return citation

    def add_observation(self, observation: str) -> list[str]:
        return list(dict.fromkeys(self.add(document) for document in _documents(observation)))

    def context(self) -> str:
        # No absolute paths, URLs, or generated summaries in this model input.
        return json.dumps(list(self.records.values()), ensure_ascii=False, separators=(",", ":"))

    def render_blocks(self, blocks: list) -> str:
        """Render only support explicitly selected for each answer block.

        No free-form uncited heading/aside channel: labels are authored by code.
        This checks attribution structure, not whether the text follows from it.
        """
        if not isinstance(blocks, list) or not blocks:
            raise CitationError("Missing answer blocks")
        paragraphs = []
        for index, block in enumerate(blocks, 1):
            if hasattr(block, "model_dump"):
                block = block.model_dump()
            if not isinstance(block, dict) or set(block) != {"text", "evidence_ids"}:
                raise CitationError(f"Malformed answer block {index}")
            text, refs = block["text"], block["evidence_ids"]
            if (not isinstance(text, str) or not text.strip() or "[" in text or "]" in text
                    or not isinstance(refs, list) or not refs
                    or any(not isinstance(ref, str) or ref not in self.records for ref in refs)):
                raise CitationError(f"Missing or invalid support in answer block {index}")
            # One explicit support set applies to one paragraph; whitespace is
            # formatting, not a reason to guess references for separate claims.
            text = " ".join(text.split())
            paragraphs.append(text + " " + " ".join(f"[{ref}]" for ref in dict.fromkeys(refs)))
        return self.render("\n\n".join(paragraphs))

    def render(self, answer: str) -> str:
        """Validate references, not entailment; never repair an unsupported draft."""
        if not isinstance(answer, str) or not answer.strip() or not self.records:
            raise CitationError("Missing answer or evidence")
        # Source/page labels and links belong to the renderer, not model prose.
        if re.search(r"https?://|file://|[A-Za-z]:[\\/]|\\\\|(?<!\w)/(?:[^\s/]+/)+|"
                     r"\b(?:pages?|p\.)\s*(?:\d|[ivxlcdm]+\b)|第\s*[\d一二三四五六七八九十百]+\s*页|"
                     r"\b\S+\.(?:pdf|docx)\b", answer, re.I):
            raise CitationError("Model-authored source label or path")
        references = re.findall(r"\[([^\[\]]*)\]", answer)
        if not references or any(ref not in self.records for ref in references):
            raise CitationError("Missing or unknown citation")
        remaining = re.sub(r"\[E[0-9a-f]{12}\]", "", answer)
        if "[" in remaining or "]" in remaining:
            raise CitationError("Malformed citation")
        if any(not re.search(r"\[E[0-9a-f]{12}\]", block) for block in answer.strip().split("\n\n") if block.strip()):
            raise CitationError("Uncited answer paragraph")
        labels = []
        for ref in dict.fromkeys(references):
            record = self.records[ref]
            name = re.sub(r"([\\`*_{}\[\]()<>#!|])", r"\\\1", record["filename"])
            page = f" — PDF page {record['page_label']}" if record["page_label"] else ""
            labels.append(f"[{ref}] {name}{page}")
        return answer.strip() + "\n\nSources:\n" + "\n".join(labels)
