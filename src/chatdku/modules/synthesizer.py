"""DSPy module for source-grounded ChatDKU responses."""

from __future__ import annotations

from datetime import date

import dspy
from pydantic import BaseModel, ConfigDict

from .prompts import RESPONSE_SYNTHESIS_INSTRUCTIONS


class AnswerBlock(BaseModel):
    """A paragraph and its model-selected support, never a generated locator."""
    model_config = ConfigDict(extra="forbid")
    text: str
    evidence_ids: list[str]


class ResponseSynthesisSignature(dspy.Signature):
    __doc__ = RESPONSE_SYNTHESIS_INSTRUCTIONS + """

The trajectory is a registered evidence inventory. Return a short list of answer
blocks. Each block has text (one concise substantive paragraph) and evidence_ids
(the exact IDs supporting that paragraph). Put references ONLY in evidence_ids,
not in the text. Do not produce headings, filenames, page numbers, URLs, footers,
or a separate reference block: code renders the references. Never select an ID
merely because it discusses the same topic; its passage must support the claim.
Separate sources with different rules into separately supported blocks. Follow
the document/year scope requested by the user, without asserting institutional
precedence. Distinguish credits from courses and total requirements from amounts
countable toward a total; do not merge figures from incompatible passages.
Report supported wording and differences without inventing why a rule changed.
Omit unsupported details. If no answer block is supported, return an empty list.
Never use conversation history as source evidence.
"""

    conversation_history: str = dspy.InputField(desc="Recent conversation history in JSON Lines format.")
    conversation_summary: str = dspy.InputField(desc="Summary of older conversation history.")
    gate_instruction: str = dspy.InputField(desc="Fixed control rule; contains no source facts.")
    trajectory: str = dspy.InputField(desc="Registered current-question evidence IDs, original text and source identity.")
    trajectory_summary: str = dspy.InputField(desc="Legacy context, not factual evidence; empty in the Agent workflow.")
    current_date: str = dspy.InputField(desc="Today's date in ISO format.")
    current_user_message: str = dspy.InputField(desc="The user's original question.")
    response: list[AnswerBlock] = dspy.OutputField(
        desc="Answer blocks, each with text and its supporting evidence_ids; no manually formatted citations."
    )


class ResponseSynthesizer(dspy.Module):
    """Produce a grounded final answer from retrieved document passages."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(ResponseSynthesisSignature)

    def forward(
        self,
        current_user_message: str,
        *,
        conversation_history: str = "",
        conversation_summary: str = "",
        gate_instruction: str = "Use only the evidence directly relevant to the question.",
        trajectory: str = "",
        trajectory_summary: str = "",
        current_date: date | None = None,
    ) -> dspy.Prediction:
        return self.predict(
            current_user_message=current_user_message.strip(),
            conversation_history=conversation_history,
            conversation_summary=conversation_summary,
            gate_instruction=gate_instruction,
            trajectory=trajectory,
            trajectory_summary=trajectory_summary,
            current_date=(current_date or date.today()).isoformat(),
        )


Synthesizer = ResponseSynthesizer
SynthesizerSignature = ResponseSynthesisSignature
