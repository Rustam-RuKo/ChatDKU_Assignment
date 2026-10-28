"""The public ChatDKU agent harness."""

from __future__ import annotations

import json
import logging
from typing import Any

# DSPy installs lazy imports for optional packages. Qdrant imports
# ``numpy.typing`` during startup, so load NumPy eagerly before DSPy to avoid
# its lazy-import proxy interfering with NumPy's internal module setup.
import numpy  # noqa: F401
import dspy

from .config import Settings, get_settings
from .modules import (
    ConversationMemory,
    Planner,
    ProgramGate,
    QueryRewrite,
    Synthesizer,
    ToolMemory,
)
from .tools import KeywordRetriever, VectorRetriever
from .tools.retrieval import serialize_documents
from .modules.capacity import BudgetedLM, CapacityError
from .modules.citations import CitationError
from .modules.program_gate import unresolved_reference


logger = logging.getLogger(__name__)


def configure_dspy(harness_settings: Settings | None = None) -> Settings:
    """Configure DSPy for the selected OpenAI-compatible chat endpoint."""
    active_settings = harness_settings or get_settings()
    if not active_settings.llm_model:
        raise ValueError("Set CHATDKU_LLM_MODEL before running the agent.")
    model = active_settings.llm_model
    if not model.startswith("openai/"):
        model = f"openai/{model}"
    dspy.configure(
        lm=BudgetedLM(
            model=model,
            api_base=active_settings.llm_base_url,
            api_key=active_settings.llm_api_key or "not-required",
            model_type="chat",
            max_tokens=active_settings.llm_max_tokens,
            temperature=active_settings.llm_temperature,
            context_tokens=active_settings.context_tokens,
            safety_tokens=active_settings.context_safety_tokens,
            tokenizer_path=active_settings.tokenizer_path,
        )
    )
    return active_settings


class Agent(dspy.Module):
    """Run ChatDKU's retrieval, evidence-gating, and response-synthesis loop."""

    def __init__(
        self,
        *,
        max_iterations: int = 3,
        rewrite_query: bool = True,
        tools: list[Any] | None = None,
        harness_settings: Settings | None = None,
    ) -> None:
        super().__init__()
        self.settings = configure_dspy(harness_settings)
        self.rewrite_query = rewrite_query
        self.conversation_memory = ConversationMemory()
        self.tool_memory = ToolMemory()
        self.query_rewriter = QueryRewrite()
        self.program_gate = ProgramGate()
        self.synthesizer = Synthesizer()
        self.vector_retriever = VectorRetriever(self.settings)
        self.keyword_retriever = KeywordRetriever(self.settings)
        self.planner = Planner(tools or [self.vector_search, self.keyword_search], max_iterations,
                               evidence_max_bytes=self.settings.evidence_max_bytes)
        self.previous_response: str | None = None

    def reset(self) -> None:
        self.conversation_memory = ConversationMemory()
        self.tool_memory.reset()
        self.planner.reset()
        self.previous_response = None

    def vector_search(self, semantic_query: str, top_k: int | None = None) -> str:
        """Search indexed DKU documents by semantic similarity.

        Args:
            semantic_query: A natural-language query for conceptual searches.
            top_k: Optional maximum number of results to return.
        """
        return serialize_documents(self.vector_retriever.query(semantic_query, top_k))

    def keyword_search(self, keyword_query: str, top_k: int | None = None, source_name: str | None = None) -> str:
        """Search indexed DKU documents for exact words, names, phrases, and codes.

        Args:
            keyword_query: Specific terms or phrases for BM25 keyword matching.
            top_k: Optional maximum number of results to return.
            source_name: Optional case-insensitive filename fragment to restrict
                results to a requested source. A missing filename reports
                available filenames without returning documentary evidence.
        """
        options = {} if source_name is None else {"source_name": source_name}
        return serialize_documents(self.keyword_retriever.query(keyword_query, top_k, **options))

    def forward(self, current_user_message: str) -> dspy.Prediction:
        """Answer one question and retain the completed turn as conversation memory."""
        self.planner.reset()
        # Memory compression mutates lists before its predictor can fail. Save
        # data only, not DSPy modules, so a retry starts at the same boundary.
        memories = (self.conversation_memory, self.tool_memory)
        snapshots = [(list(memory.history), memory.summary) for memory in memories]
        previous_response = self.previous_response
        try:
            return self._forward_turn(current_user_message)
        except Exception as error:
            for memory, (history, summary) in zip(memories, snapshots):
                memory.history = history
                memory.summary = summary
            self.previous_response = previous_response
            omitted = list(self.planner.inventory.records)
            self.planner.reset()
            if isinstance(error, (CapacityError, CitationError)):
                status = "capacity" if isinstance(error, CapacityError) else "invalid_citations"
                logger.warning("Turn withheld: %s; omitted evidence IDs=%s", error, omitted)
                chinese = any("\u4e00" <= char <= "\u9fff" for char in current_user_message)
                if status == "capacity":
                    response = ("完整证据和上下文超出本次处理容量。请缩小问题范围；我没有基于截断的证据作答。" if chinese else
                                "The complete evidence and context exceed this run's capacity. Please narrow the question; no partial-evidence answer was produced.")
                else:
                    response = ("我无法验证本次回答的来源引用，因此未提供该回答。请缩小问题范围或提供相关段落。" if chinese else
                                "I could not verify this answer's source references, so I withheld the draft. Please narrow the question or provide a relevant passage.")
                return dspy.Prediction(response=response, rewritten_query=current_user_message,
                                       trajectory="[]", gate_decision="INSUFFICIENT",
                                       citation_status=status, omitted_evidence_ids=omitted)
            raise

    def _forward_turn(self, current_user_message: str) -> dspy.Prediction:
        if self.previous_response:
            self.conversation_memory("assistant", self.previous_response)

        logger.debug("Agent received user message:\n%s", current_user_message)
        clarification = unresolved_reference(current_user_message, self.conversation_memory.history_str(),
                                             self.conversation_memory.summary)
        if clarification:
            self.conversation_memory("user", current_user_message)
            self.previous_response = clarification
            return dspy.Prediction(response=clarification, rewritten_query=current_user_message,
                                   trajectory="[]", gate_decision="CLARIFY", citation_status="not_applicable")
        planner_message = current_user_message
        if self.rewrite_query:
            rewrite_context = {
                "current_user_message": current_user_message,
                "conversation_history": self.conversation_memory.history_str(),
                "conversation_summary": self.conversation_memory.summary,
                "tool_history": self.tool_memory.history_str(),
                "tool_summary": self.tool_memory.summary,
            }
            _log_context("query rewrite", rewrite_context)
            rewrite = self.query_rewriter(
                current_user_message,
                conversation_history=rewrite_context["conversation_history"],
                conversation_summary=rewrite_context["conversation_summary"],
                tool_history=rewrite_context["tool_history"],
                tool_summary=rewrite_context["tool_summary"],
            )
            planner_message = rewrite.rewritten_query
            logger.debug("Query rewrite result:\n%s", planner_message)

        _log_context(
            "planner",
            {
                "current_user_message": current_user_message,
                "retrieval_hint": planner_message if self.rewrite_query else "",
                "conversation_history": self.conversation_memory.history_str(),
                "conversation_summary": self.conversation_memory.summary,
            },
        )
        plan = self.planner(current_user_message, self.conversation_memory,
                            retrieval_hint=planner_message if self.rewrite_query else "")
        logger.debug("Planner trajectory (tool calls and observations):\n%s", plan.trajectory)
        logger.debug("Planner trajectory summary:\n%s", plan.summary)
        self._record_tool_history(current_user_message, plan.trajectory)
        logger.debug("Current-question raw tool evidence:\n%s", plan.evidence)
        gate = self.program_gate(current_user_message, plan.evidence,
                                 conversation_history=self.conversation_memory.history_str(),
                                 conversation_summary=self.conversation_memory.summary)
        logger.debug(
            "Program gate decision: %s\nGuardrail: %s\nSafe response: %s",
            gate.decision,
            gate.guardrail,
            gate.safe_response,
        )

        if gate.allow:
            synthesis_context = {
                "current_user_message": current_user_message,
                "conversation_history": self.conversation_memory.history_str(),
                "conversation_summary": self.conversation_memory.summary,
                "gate_instruction": gate.guardrail,
                "trajectory": plan.inventory.context(),
                "trajectory_summary": "",
            }
            _log_context("response synthesis", synthesis_context)
            response = self.synthesizer(
                current_user_message,
                conversation_history=synthesis_context["conversation_history"],
                conversation_summary=synthesis_context["conversation_summary"],
                gate_instruction=synthesis_context["gate_instruction"],
                trajectory=synthesis_context["trajectory"],
                trajectory_summary=synthesis_context["trajectory_summary"],
            ).response
            response = (plan.inventory.render_blocks(response) if isinstance(response, list)
                        else plan.inventory.render(response))
        else:
            response = gate.safe_response

        logger.debug("Final agent response:\n%s", response)
        self.conversation_memory("user", current_user_message)
        self.previous_response = response
        return dspy.Prediction(
            response=response,
            rewritten_query=planner_message,
            trajectory=plan.trajectory,
            gate_decision=gate.decision,
            citation_status="verified" if gate.allow else "not_applicable",
        )

    def _record_tool_history(self, current_user_message: str, trajectory: str) -> None:
        """Record completed tool calls without exposing them in the final answer."""
        try:
            events = json.loads(trajectory)
        except json.JSONDecodeError:
            return
        for event in events:
            name = event.get("tool_name")
            if name and name != "finish":
                observation = event.get("observation", "")
                if isinstance(observation, list):
                    # Cross-turn memory records retrieval actions, not another
                    # copy of source text that cannot authorize a later answer.
                    observation = json.dumps({"evidence_ids": [record["id"] for record in observation]})
                self.tool_memory(
                    current_user_message,
                    self.conversation_memory,
                    name,
                    event.get("tool_args", {}),
                    observation,
                )


def _log_context(stage: str, context: dict[str, str]) -> None:
    """Emit the exact non-secret inputs supplied to one agent stage at DEBUG."""
    logger.debug(
        "%s context window:\n%s",
        stage.capitalize(),
        json.dumps(context, ensure_ascii=False, indent=2, default=str),
    )
