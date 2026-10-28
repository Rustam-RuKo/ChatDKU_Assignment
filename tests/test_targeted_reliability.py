"""Targeted retrieval regressions with synthetic documents only."""
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy  # noqa: F401
from chatdku.tools.keyword_retriever import KeywordRetriever
from test_source_evidence import document
from test_evidence_gating import step
from chatdku.modules.planner import Planner
from chatdku.modules.conversation_memory import ConversationMemory
from chatdku.tools.retrieval import serialize_documents


class SourceFilterTests(unittest.TestCase):
    def test_filter_precedes_ranking_and_missing_source_never_falls_back(self):
        docs = [document(node="a", name="Handbook.pdf", text="SYNTH 123 eligibility"),
                document(node="b", name="Student FAQ.docx", page=None, text="First year eligibility")]
        client = Mock()
        client.scroll.return_value = ([SimpleNamespace(id=str(i), payload={}) for i in range(2)], None)
        retriever = KeywordRetriever(SimpleNamespace(
            retrieval_top_k=8, index_dir=Path("synthetic-index"), vector_collection="synthetic"))
        with patch("chatdku.tools.keyword_retriever.QdrantClient", return_value=client), \
             patch("chatdku.tools.keyword_retriever.document_from_payload", side_effect=lambda node, *_: docs[int(node)]):
            self.assertEqual(retriever.query("eligibility", 1, source_name="fAq")[0].node_id, "b")
            with self.assertRaisesRegex(ValueError, "Available filenames:.*Handbook.pdf.*Student FAQ.docx"):
                retriever.query("eligibility", source_name="missing")
            self.assertEqual(len(retriever.query("eligibility")), 2)
        self.assertEqual(client.close.call_count, 3)


class ProgressTests(unittest.TestCase):
    def test_duplicate_request_is_not_executed_twice_and_reset_allows_retry(self):
        executions = []
        def search(query: str):
            executions.append(query)
            return serialize_documents([document()])
        planner = Planner([search], max_iterations=3)
        planner.predict = Mock(side_effect=[step("search", query="scope"), step("search", query="scope"), step()])
        result = planner("Compare policy scopes", ConversationMemory())
        self.assertEqual(executions, ["scope"])
        events = json.loads(result.trajectory)
        self.assertEqual(events[1]["retrieval_progress"]["status"], "duplicate_request")
        self.assertEqual(len(planner.inventory.records), 1)
        planner.predict = Mock(side_effect=[step("search", query="scope"), step()])
        planner("New turn", ConversationMemory())
        self.assertEqual(executions, ["scope", "scope"])

    def test_different_queries_report_new_identity_not_execution_success(self):
        def search(query: str):
            return serialize_documents([document(node="second" if query == "new" else "first")])
        planner = Planner([search], max_iterations=3)
        planner.predict = Mock(side_effect=[step("search", query="one"), step("search", query="same evidence"),
                                           step("search", query="new")])
        result = planner("Compare requirements", ConversationMemory())
        events = json.loads(result.trajectory)
        self.assertEqual([e["retrieval_progress"]["status"] for e in events],
                         ["new_evidence", "no_new_evidence", "new_evidence"])
        self.assertEqual(len(json.loads(result.evidence)), 3)
        self.assertEqual(len(planner.inventory.records), 2)

    def test_empty_results_are_not_progress_and_transient_errors_can_retry(self):
        executions = []
        def search(query: str):
            executions.append(query)
            if len(executions) == 1:
                raise RuntimeError("Synthetic transient error")
            return "[]"
        planner = Planner([search], max_iterations=3)
        planner.predict = Mock(return_value=step("search", query="missing"))
        result = planner("Missing requirement", ConversationMemory())
        self.assertEqual(len(executions), 3)
        self.assertEqual(result.evidence, "[]")
        self.assertEqual(json.loads(result.trajectory)[-1]["retrieval_progress"]["status"], "no_new_evidence")

    def test_agent_exposes_source_filter_without_changing_document_metadata(self):
        from test_evidence_gating import agent
        app = agent()
        app.keyword_retriever.query = Mock(return_value=[document()])
        result = json.loads(app.keyword_search("eligibility", 2, source_name="policy"))
        app.keyword_retriever.query.assert_called_once_with("eligibility", 2, source_name="policy")
        self.assertEqual(result[0]["metadata"], document().metadata)

    def test_filename_feedback_cannot_authorize_a_policy_answer(self):
        import dspy
        from test_evidence_gating import agent
        app = agent()
        def search(query: str):
            raise ValueError("No matching source filename. Available filenames: handbook.pdf")
        app.planner = Planner([search], max_iterations=2)
        app.planner.predict = Mock(side_effect=[step("search", query="eligibility"), step()])
        app.program_gate.predict = Mock(return_value=dspy.Prediction(decision="ALLOW"))
        app.synthesizer.predict = Mock(side_effect=AssertionError("Filename feedback is not evidence"))
        result = app("What are the eligibility requirements?")
        self.assertEqual(result.gate_decision, "INSUFFICIENT")
        self.assertFalse(app.planner.inventory.records)
