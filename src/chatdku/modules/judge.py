"""Judge whether the agent has enough evidence to answer."""

from __future__ import annotations

import re

import dspy

from .conversation_memory import ConversationMemory
from .prompts import JUDGE_INSTRUCTIONS
from .tool_memory import ToolMemory


class JudgeSignature(dspy.Signature):
    __doc__ = JUDGE_INSTRUCTIONS

    current_user_message: str = dspy.InputField()
    conversation_history: str = dspy.InputField()
    conversation_summary: str = dspy.InputField()
    tool_history: str = dspy.InputField()
    tool_summary: str = dspy.InputField()
    judgement: str = dspy.OutputField(desc='Reply with exactly "Yes" or "No".')


class Judge(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.judge = dspy.Predict(JudgeSignature)

    def forward(
        self,
        current_user_message: str,
        conversation_memory: ConversationMemory,
        tool_memory: ToolMemory,
    ) -> dspy.Prediction:
        prediction = self.judge(
            current_user_message=current_user_message,
            conversation_history=conversation_memory.history_str(),
            conversation_summary=conversation_memory.summary,
            tool_history=tool_memory.history_str(),
            tool_summary=tool_memory.summary,
        )
        decision = re.sub(r"<think>.*?</think>", "", prediction.judgement, flags=re.DOTALL)
        return dspy.Prediction(judgement=decision.strip().removesuffix(".") == "Yes")
