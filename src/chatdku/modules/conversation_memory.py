"""Conversation memory compatible with the ChatDKU DSPy agent flow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import dspy

from .prompts import CONVERSATION_MEMORY_INSTRUCTIONS


@dataclass(frozen=True, slots=True)
class ConversationMemoryEntry:
    role: str
    content: str


class CompressConversationMemorySignature(dspy.Signature):
    __doc__ = CONVERSATION_MEMORY_INSTRUCTIONS

    history_to_discard: str = dspy.InputField(desc="Discarded messages in JSON Lines format.")
    previous_summary: str = dspy.InputField(desc="Previous summary, which may be empty.")
    current_summary: str = dspy.OutputField(desc="Updated summary of discarded history.")


class ConversationMemory(dspy.Module):
    """Keep recent turns verbatim and summarize older ones when necessary."""

    def __init__(self, max_history_chars: int = 12_000) -> None:
        super().__init__()
        self.compressor = dspy.Predict(CompressConversationMemorySignature)
        self.max_history_chars = max_history_chars
        self.history: list[ConversationMemoryEntry] = []
        self.summary = ""

    def history_str(self, left: int = 0, right: int | None = None) -> str:
        entries = self.history[left:right]
        return "\n".join(json.dumps(asdict(entry), ensure_ascii=False) for entry in entries)

    def forward(
        self, role: str, content: str, max_history_size: int | None = None
    ) -> None:
        if max_history_size is not None:
            self.max_history_chars = max_history_size
        self.history.append(ConversationMemoryEntry(role=role, content=content))
        self._compress_if_needed()

    def register_history(self, role: str, content: str) -> None:
        self.forward(role, content)

    def _compress_if_needed(self) -> None:
        history = self.history_str()
        if len(history) <= self.max_history_chars or len(self.history) < 2:
            return
        discarded: list[ConversationMemoryEntry] = []
        while len(self.history_str()) > self.max_history_chars and len(self.history) > 1:
            discarded.append(self.history.pop(0))
        discarded_text = "\n".join(json.dumps(asdict(entry), ensure_ascii=False) for entry in discarded)
        self.summary = self.compressor(
            history_to_discard=discarded_text,
            previous_summary=self.summary,
        ).current_summary
