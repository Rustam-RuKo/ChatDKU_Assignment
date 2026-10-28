"""Evidence-scope gate for source-grounded responses."""

from __future__ import annotations

import re

import dspy

from .evidence import evidence_from_trajectory
from .capacity import CapacityError
from .prompts import (
    CLARIFY_RESPONSE,
    GROUNDED_SYNTHESIS_INSTRUCTION,
    INSUFFICIENT_RESPONSE,
    PROGRAM_GATE_INSTRUCTIONS,
)


class ProgramGateSignature(dspy.Signature):
    __doc__ = PROGRAM_GATE_INSTRUCTIONS

    current_user_message: str = dspy.InputField()
    conversation_history: str = dspy.InputField(desc="Resolve user references only; never documentary policy evidence.")
    conversation_summary: str = dspy.InputField(desc="Older context only; cannot supply missing policy evidence.")
    trajectory: str = dspy.InputField(desc="Current-question raw tool observations, not verified facts.")
    decision: str = dspy.OutputField(desc="Exactly ALLOW, CLARIFY, or INSUFFICIENT.")


class ProgramGate(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(ProgramGateSignature)

    def forward(
        self, current_user_message: str, trajectory: str, trajectory_summary: str = "",
        *, conversation_history: str = "", conversation_summary: str = ""
    ) -> dspy.Prediction:
        # Keep the optional summary argument for compatibility, but never use
        # model-generated summaries as evidence for this decision.
        evidence = evidence_from_trajectory(trajectory)
        try:
            prediction = self.predict(
                current_user_message=current_user_message,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                trajectory=evidence,
            )
            raw = prediction.decision.strip().upper() if isinstance(prediction.decision, str) else ""
            decision = raw if raw in {"ALLOW", "CLARIFY", "INSUFFICIENT"} else "INSUFFICIENT"
        except CapacityError:
            raise
        except Exception:
            decision = "INSUFFICIENT"
        if decision == "ALLOW" and evidence == "[]":
            decision = "INSUFFICIENT"
        # Generated prose is not a second answer channel or an authoritative
        # instruction. Only the validated control label crosses this boundary.
        safe_response = _safe_response(decision, current_user_message)
        return dspy.Prediction(
            decision=decision,
            guardrail=GROUNDED_SYNTHESIS_INSTRUCTION,
            safe_response=safe_response,
            allow=decision == "ALLOW",
        )


def _safe_response(decision: str, question: str) -> str:
    """Select authored messages from question cues, never generated gate prose."""
    if decision == "ALLOW":
        return ""
    chinese = bool(re.search(r"[\u4e00-\u9fff]", question))
    if decision == "CLARIFY":
        if re.search(r"\b(?:this|that) course\b", question, re.IGNORECASE) or any(
            cue in question for cue in ("这门课", "那门课")
        ):
            return ("请提供课程名称或课程代码，以便我查找相关规定。" if chinese else
                    "What is the course name or code? That will help me find the applicable policy.")
        return ("请说明你指的是哪门课程、哪个项目或哪个学年（提供与问题相关的信息即可）。" if chinese else
                CLARIFY_RESPONSE)
    return ("目前检索到的资料不足以确认这一点。请提供相关规定的段落，或补充问题的具体背景，以便继续核查。"
            if chinese else INSUFFICIENT_RESPONSE +
            " Please share the relevant policy passage or add context so I can check again.")


def unresolved_reference(question: str, conversation_history: str = "", conversation_summary: str = "") -> str:
    """Clarify explicit references only when there is no context to resolve them.

    This is a conservative fast path, not general missing-slot inference. Any
    history, quoted name, or course code delegates resolution to the workflow.
    """
    if conversation_history.strip() or conversation_summary.strip():
        return ""
    if re.search(r"\b[A-Za-z]{2,}\s*\d{2,}\b|[\"“‘].+?[\"”’]", question):
        return ""
    if re.search(r"\b(?:this|that) course\b", question, re.I) or any(cue in question for cue in ("这门课", "那门课")):
        return _safe_response("CLARIFY", question)
    chinese = bool(re.search(r"[\u4e00-\u9fff]", question))
    if (re.search(r"\bthat (?:academic )?year\b", question, re.I) or "那个学年" in question) and not re.search(r"\b(?:19|20)\d{2}\b", question):
        return "你指的是哪个学年？" if chinese else "Which academic year do you mean?"
    if re.search(r"\bthat program\b", question, re.I) or "那个项目" in question:
        return "你指的是哪个学位项目？" if chinese else "Which academic program do you mean?"
    return ""
