"""Tool-call memory compatible with the ChatDKU DSPy agent flow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import dspy

from .conversation_memory import ConversationMemory
from .prompts import TOOL_MEMORY_INSTRUCTIONS


@dataclass(frozen=True, slots=True)
class ToolMemoryEntry:
    name: str
    arguments: dict[str, Any]
    result: Any


class CompressToolMemorySignature(dspy.Signature):
    __doc__ = TOOL_MEMORY_INSTRUCTIONS

    current_user_message: str = dspy.InputField()
    conversation_history: str = dspy.InputField()
    conversation_summary: str = dspy.InputField()
    history_to_discard: str = dspy.InputField()
    previous_summary: str = dspy.InputField()
    current_summary: str = dspy.OutputField()


class ToolMemory(dspy.Module):
    """Keep recent tool observations and summarize older evidence."""

    def __init__(self, max_history_chars: int = 16_000) -> None:
        super().__init__()
        self.compressor = dspy.Predict(CompressToolMemorySignature)
        self.max_history_chars = max_history_chars
        self.reset()

    def reset(self) -> None:
        self.history: list[ToolMemoryEntry] = []
        self.summary = ""

    def history_str(self, left: int = 0, right: int | None = None) -> str:
        return "\n".join(
            json.dumps(asdict(entry), ensure_ascii=False, default=str)
            for entry in self.history[left:right]
        )

    def forward(
        self,
        current_user_message: str,
        conversation_memory: ConversationMemory,
        name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        self.history.append(ToolMemoryEntry(name=name, arguments=arguments, result=result))
        if len(self.history_str()) <= self.max_history_chars or len(self.history) < 2:
            return
        discarded: list[ToolMemoryEntry] = []
        while len(self.history_str()) > self.max_history_chars and len(self.history) > 1:
            discarded.append(self.history.pop(0))
        self.summary = self.compressor(
            current_user_message=current_user_message,
            conversation_history=conversation_memory.history_str(),
            conversation_summary=conversation_memory.summary,
            history_to_discard="\n".join(
                json.dumps(asdict(entry), ensure_ascii=False, default=str) for entry in discarded
            ),
            previous_summary=self.summary,
        ).current_summary
