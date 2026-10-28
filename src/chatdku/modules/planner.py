"""Multi-step DSPy planner for the public ChatDKU agent harness."""

from __future__ import annotations

import json
from typing import Any

import dspy

from .conversation_memory import ConversationMemory
from .evidence import evidence_from_trajectory
from .citations import EvidenceInventory, CitationError
from .capacity import CapacityError
from .prompts import CHATDKU_ROLE, RETRIEVAL_PLAN_INSTRUCTIONS


class PlannerSignature(dspy.Signature):
    __doc__ = RETRIEVAL_PLAN_INSTRUCTIONS

    current_user_message: str = dspy.InputField()
    retrieval_hint: str = dspy.InputField(desc="Optional search wording; never replaces the original question's scope.")
    conversation_history: str = dspy.InputField()
    conversation_summary: str = dspy.InputField()
    chatbot_role: str = dspy.InputField()


class SummarizerSignature(dspy.Signature):
    __doc__ = """Summarize discarded tool observations while preserving evidence relevant to the user question."""

    current_user_message: str = dspy.InputField()
    trajectory_to_discard: str = dspy.InputField()
    previous_summary: str = dspy.InputField()
    new_summary: str = dspy.OutputField()


class Planner(dspy.Module):
    """Run a bounded sequence of named tools and return its trajectory."""

    def __init__(self, tools: list[Any], max_iterations: int = 5, *, evidence_max_bytes: int = 24000) -> None:
        super().__init__()
        normalized = [tool if isinstance(tool, dspy.Tool) else dspy.Tool(tool) for tool in tools]
        self.tools = {tool.name: tool for tool in normalized}
        self.tools["finish"] = dspy.Tool(
            lambda: "Completed.",
            name="finish",
            desc="Finish after collecting enough information to answer the user.",
            args={},
        )
        tool_list = "\n".join(_describe_tool(name, tool) for name, tool in self.tools.items())
        signature = (
            dspy.Signature({**PlannerSignature.input_fields}, f"{RETRIEVAL_PLAN_INSTRUCTIONS}\n\nAvailable tools:\n{tool_list}")
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=str)
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )
        self.predict = dspy.Predict(signature)
        self.summarizer = dspy.Predict(SummarizerSignature)
        self.max_iterations = max_iterations
        if evidence_max_bytes < 1:
            raise ValueError("evidence_max_bytes must be positive")
        self.evidence_max_bytes = evidence_max_bytes
        self.inventory = EvidenceInventory()
        self.trajectory_summary = ""

    def reset(self) -> None:
        """Discard question-specific planning state."""
        self.trajectory_summary = ""
        self.inventory = EvidenceInventory()

    def forward(
        self, current_user_message: str, conversation_memory: ConversationMemory, *, retrieval_hint: str = ""
    ) -> dspy.Prediction:
        self.reset()
        trajectory: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        # Cache only successful documentary requests within this turn. Failed
        # tools remain retryable; no evidence is carried across questions.
        completed_requests: set[str] = set()
        inputs = {
            "current_user_message": current_user_message,
            "retrieval_hint": retrieval_hint,
            "conversation_history": conversation_memory.history_str(),
            "conversation_summary": conversation_memory.summary,
            "chatbot_role": CHATDKU_ROLE,
        }
        for _ in range(self.max_iterations):
            plan = self.predict(**inputs, trajectory=_format_trajectory(trajectory))
            name = str(plan.next_tool_name).strip()
            arguments = plan.next_tool_args if isinstance(plan.next_tool_args, dict) else {}
            event: dict[str, Any] = {
                "thought": str(plan.next_thought),
                "tool_name": name,
                "tool_args": arguments,
            }
            if name not in self.tools:
                event["observation"] = f"Unknown tool: {name}. Choose one of {', '.join(self.tools)}."
                trajectory.append(event)
                break
            request_key = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False, default=str)
            if request_key in completed_requests:
                event["observation"] = "Request already completed; evidence remains in earlier observations."
                event["retrieval_progress"] = {"status": "duplicate_request", "new_evidence_ids": []}
                trajectory.append(event)
                continue
            try:
                event["observation"] = self.tools[name](**arguments)
                # Preserve raw results independently of working-history
                # compression. Do not promote thoughts or summaries to facts.
                if name != "finish":
                    observations.append({"tool_name": name, "observation": event["observation"]})
                    previous_ids = set(self.inventory.records)
                    identifiers = self.inventory.add_observation(event["observation"])
                    new_ids = [key for key in identifiers if key not in previous_ids]
                    event["retrieval_progress"] = {
                        "status": "new_evidence" if new_ids else "no_new_evidence",
                        "new_evidence_ids": new_ids,
                    }
                    size = len(_format_trajectory(observations).encode("utf-8"))
                    if size > self.evidence_max_bytes:
                        raise CapacityError("raw observations", size, self.evidence_max_bytes)
                    if identifiers:
                        completed_requests.add(request_key)
                        # All source text stays intact. Avoid nested JSON
                        # escaping and internal filesystem metadata in prompts.
                        event["observation"] = [self.inventory.records[key] for key in identifiers]
            except (CapacityError, CitationError):
                raise
            except Exception as error:
                event["observation"] = f"Execution error in {name}: {error}"
            trajectory.append(event)
            # Keep complete observations. The raw-byte and formatted-request
            # guards bound capacity; the old summary was never consumed by a
            # subsequent planner prediction, so generating it added only cost.
            if name == "finish":
                break
        return dspy.Prediction(
            trajectory=_format_trajectory(trajectory), summary=self.trajectory_summary,
            evidence=evidence_from_trajectory(_format_trajectory(observations)),
            inventory=self.inventory,
        )


def _format_trajectory(trajectory: list[dict[str, Any]]) -> str:
    return json.dumps(trajectory, ensure_ascii=False, default=str, indent=2)


def _describe_tool(name: str, tool: dspy.Tool) -> str:
    """Render each callable's exact schema in the planner's instructions."""
    arguments = ", ".join(
        f"{argument_name}: {argument.get('type', 'value')}"
        for argument_name, argument in tool.args.items()
    ) or "no arguments"
    return f"- {name}({arguments}): {tool.desc}"


RetrievalPlanner = Planner
