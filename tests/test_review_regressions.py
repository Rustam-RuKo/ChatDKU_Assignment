"""Review regressions: synthetic documents and fake predictions only."""
import json
import unittest
from unittest.mock import Mock

import numpy  # noqa: F401
import dspy

from test_evidence_gating import agent, evidence, step, cited_answer
from chatdku.modules.evidence import evidence_from_trajectory
from chatdku.modules.program_gate import ProgramGate
from chatdku.tools.retrieval import RetrievedDocument, serialize_documents


class EvidenceReviewTests(unittest.TestCase):
    def test_document_metadata_survives_both_agent_retrieval_wrappers(self):
        app = agent()
        document = RetrievedDocument("node-7", "6 credits, except approved transfers.",
                                     {"file_name": "synthetic.txt", "page_label": "7"}, 0.8)
        for retriever, search in ((app.vector_retriever, app.vector_search),
                                  (app.keyword_retriever, app.keyword_search)):
            retriever.query = Mock(return_value=[document])
            raw = search("transfer", 2)
            result = evidence_from_trajectory(json.dumps([dict(tool_name="search", observation=raw)]))
            self.assertEqual(json.loads(json.loads(result)[0]["observation"]), [document.as_dict()])
            retriever.query.assert_called_once_with("transfer", 2)

    def test_malformed_document_batch_is_rejected_whole(self):
        valid = RetrievedDocument("node-1", "Synthetic rule with an exception.",
                                  {"file_name": "synthetic.txt"}, 1.0).as_dict()
        for changes in ({"node_id": ""}, {"text": " "}, {"metadata": {}},
                        {"metadata": {"file_name": 1}}, {"score": float("nan")},
                        {"score": True}, {"score": "1"}):
            raw = json.dumps([dict(tool_name="search", observation=json.dumps([valid, valid | changes]))])
            self.assertEqual(evidence_from_trajectory(raw), "[]")

    def test_irrelevant_document_still_requires_semantic_gate(self):
        raw = serialize_documents([RetrievedDocument("weather", "It rains on the synthetic campus.",
                                                    {"file_name": "weather.txt"}, 1.0)])
        trajectory = json.dumps([dict(tool_name="search", observation=raw)])
        self.assertNotEqual(evidence_from_trajectory(trajectory), "[]")
        gate = ProgramGate()
        gate.predict = Mock(return_value=dspy.Prediction(decision="INSUFFICIENT"))
        self.assertFalse(gate("How many credits?", trajectory).allow)

    def test_successful_non_document_output_is_not_evidence(self):
        for observation in ("Connection refused", "HTTP 500", "hello", "42", "null",
                            "{}", "[]", '{"error":"timeout"}', '[{"text":"6 credits"}]'):
            with self.subTest(observation=observation):
                raw = json.dumps([dict(tool_name="search", observation=observation)])
                self.assertEqual(evidence_from_trajectory(raw), "[]")

    def test_successful_status_tool_cannot_authorize_synthesis(self):
        app = agent()
        app.planner.tools["status"] = dspy.Tool(lambda: "Server healthy", name="status", args={})
        app.planner.predict = Mock(side_effect=[step("status"), step()])
        app.program_gate.predict = Mock(return_value=dspy.Prediction(decision="ALLOW"))
        app.synthesizer.predict = Mock()
        self.assertEqual(app("What is required?").gate_decision, "INSUFFICIENT")
        app.synthesizer.predict.assert_not_called()

    def test_clarification_uses_original_question_without_generated_facts(self):
        for question, expected in (("Can I take this course?", "course name"),
                                   ("这门课可以选吗？", "课程名称")):
            gate = ProgramGate()
            gate.predict = Mock(return_value=dspy.Prediction(decision="CLARIFY", safe_response="999 credits"))
            answer = gate(question, evidence()).safe_response
            self.assertIn(expected, answer)
            self.assertNotIn("999", answer)

    def test_chinese_insufficiency_is_actionable(self):
        gate = ProgramGate()
        gate.predict = Mock(return_value=dspy.Prediction(decision="INSUFFICIENT"))
        self.assertIn("相关", gate("我的成绩是多少？", "[]").safe_response)


class RecoveryTests(unittest.TestCase):
    def test_conversation_compression_failure_restores_discarded_entries(self):
        app = agent()
        app.conversation_memory("user", "Previous question")
        app.previous_response = "Previous answer"
        before = app.conversation_memory.history_str()
        app.conversation_memory.max_history_chars = 1
        app.conversation_memory.compressor = Mock(side_effect=RuntimeError("compression failed"))
        with self.assertRaises(RuntimeError):
            app("New question")
        self.assertEqual(app.conversation_memory.history_str(), before)
        self.assertEqual(app.previous_response, "Previous answer")

    def test_gate_predictor_failure_is_a_completed_abstaining_turn(self):
        app = agent()
        app.previous_response = "Previous answer"
        app.planner.predict = Mock(return_value=step())
        app.program_gate.predict = Mock(side_effect=RuntimeError("endpoint failure"))
        app.synthesizer.predict = Mock()
        self.assertEqual(app("New question").gate_decision, "INSUFFICIENT")
        self.assertEqual(app.conversation_memory.history_str().count("Previous answer"), 1)
        app.synthesizer.predict.assert_not_called()

    def test_failed_turn_rolls_back_all_memory_and_retry_does_not_duplicate(self):
        for stage in ("rewrite", "planner", "gate", "synthesis", "tool_memory", "conversation_memory"):
            with self.subTest(stage=stage):
                app = agent()
                app.conversation_memory("user", "Previous question")
                app.previous_response = "Previous answer"
                app.conversation_memory.summary = "Prior conversation summary"
                app.tool_memory.summary = "Prior tool summary"
                before = (app.conversation_memory.history_str(), app.tool_memory.history_str())
                app.planner.predict = Mock(side_effect=[step("synthetic_search", query="short"), step()])
                app.program_gate.predict = Mock(return_value=dspy.Prediction(decision="ALLOW"))
                app.synthesizer.predict = Mock(side_effect=cited_answer)
                fail = Mock(side_effect=RuntimeError("synthetic failure"))
                if stage == "rewrite":
                    app.rewrite_query = True
                    app.query_rewriter.forward = fail
                elif stage == "planner":
                    app.planner.predict = fail
                elif stage == "gate":
                    app.program_gate.forward = fail
                elif stage == "synthesis":
                    app.synthesizer.predict = fail
                elif stage == "tool_memory":
                    app.tool_memory("old", app.conversation_memory, "search", {}, "old")
                    before = (app.conversation_memory.history_str(), app.tool_memory.history_str())
                    app.tool_memory.max_history_chars = 1
                    app.tool_memory.compressor = fail
                else:
                    # Fail after the final user entry was appended, not at turn start.
                    original = app.conversation_memory.forward
                    def append_then_fail(role, content):
                        original(role, content)
                        if role == "user":
                            raise RuntimeError("synthetic failure")
                    app.conversation_memory.forward = append_then_fail
                with self.assertRaises(RuntimeError):
                    app("Retry question")
                self.assertEqual((app.conversation_memory.history_str(), app.tool_memory.history_str()), before)
                self.assertEqual(app.conversation_memory.summary, "Prior conversation summary")
                self.assertEqual(app.tool_memory.summary, "Prior tool summary")
                self.assertEqual(app.previous_response, "Previous answer")
                self.assertEqual(app.planner.trajectory_summary, "")
                # Restore only the failed dependency, then retry the same Agent.
                app.rewrite_query = False
                if stage == "gate":
                    del app.program_gate.forward
                if stage == "conversation_memory":
                    del app.conversation_memory.forward
                app.tool_memory.max_history_chars = 16000
                app.planner.predict = Mock(side_effect=[step("synthetic_search", query="short"), step()])
                app.synthesizer.predict = Mock(side_effect=cited_answer)
                self.assertEqual(app("Retry question").response.split(" [")[0], "6 credits.")
                self.assertEqual(app.conversation_memory.history_str().count("Previous answer"), 1)
                self.assertEqual(app.conversation_memory.history_str().count("Retry question"), 1)


if __name__ == "__main__":
    unittest.main()
