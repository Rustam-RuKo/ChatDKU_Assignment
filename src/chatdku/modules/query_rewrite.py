"""DSPy module for producing a retrieval-ready search query."""

from __future__ import annotations

import dspy

from .prompts import CHATDKU_ROLE, QUERY_REWRITE_INSTRUCTIONS


class QueryRewriteSignature(dspy.Signature):
    __doc__ = QUERY_REWRITE_INSTRUCTIONS

    role_prompt: str = dspy.InputField(desc="System prompt describing ChatDKU's role.")
    current_user_message: str = dspy.InputField(desc="The user's original question.")
    conversation_history: str = dspy.InputField(desc="Recent conversation history in JSON Lines format.")
    conversation_summary: str = dspy.InputField(desc="Summary of older conversation history.")
    tool_history: str = dspy.InputField(desc="Previous tool calls and observations.")
    tool_summary: str = dspy.InputField(desc="Summary of older tool calls and observations.")
    rewritten_query: str = dspy.OutputField(
        desc="One concise standalone search query for semantic retrieval."
    )


class QueryRewriter(dspy.Module):
    """Turn a user question into a query suitable for the local index."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(QueryRewriteSignature)

    def forward(
        self,
        current_user_message: str,
        *,
        conversation_history: str = "",
        conversation_summary: str = "",
        tool_history: str = "",
        tool_summary: str = "",
        role_prompt: str = CHATDKU_ROLE,
    ) -> dspy.Prediction:
        return self.predict(
            role_prompt=role_prompt,
            current_user_message=current_user_message.strip(),
            conversation_history=conversation_history,
            conversation_summary=conversation_summary,
            tool_history=tool_history,
            tool_summary=tool_summary,
        )


QueryRewrite = QueryRewriter
