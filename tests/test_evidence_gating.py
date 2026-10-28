"""Synthetic policies only. Predictor replacements stand in for LLM I/O."""
import json
from hashlib import sha256
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy  # noqa: F401 -- same import ordering as the Agent
import dspy

from chatdku.modules.program_gate import ProgramGate
from chatdku.modules.planner import Planner
from chatdku.modules.conversation_memory import ConversationMemory
from chatdku.agent import Agent
from chatdku.config import Settings
from chatdku.tools.retrieval import RetrievedDocument, serialize_documents


def document_observation(text):
    node = "synthetic-" + sha256(str(text).encode()).hexdigest()[:12]
    return serialize_documents([RetrievedDocument(node, text, {"file_name": "synthetic-policy.txt"}, 1.0)])


def cited_answer(**inputs):
    citation = json.loads(inputs["trajectory"])[0]["id"]
    return dspy.Prediction(response=f"6 credits. [{citation}]")


def evidence(text="Synthetic policy: the activity requirement is 6 credits."):
    return json.dumps([{"tool_name": "keyword_search", "observation": document_observation(text)}])


def step(name="finish", **args):
    return dspy.Prediction(next_tool_name=name, next_tool_args=args,
                           next_thought="Unverified thought: require 999 credits.")


def synthetic_search(query: str) -> str:
    """Return synthetic policy evidence without I/O."""
    return document_observation("Synthetic activity requirement: 6 credits. " + ("Context. " * 2000 if query == "long" else ""))


def agent():
    with patch.dict(os.environ, {"CHATDKU_LLM_MODEL": "fake", "CHATDKU_EMBEDDING_PROVIDER": "local",
                                "CHATDKU_LOCAL_EMBEDDING_MODEL": "fake"}, clear=True):
        settings = Settings.from_environment(Path(__file__).with_name("nonexistent.env"))
    return Agent(harness_settings=settings, rewrite_query=False, tools=[synthetic_search])


class GateTests(unittest.TestCase):
    def gate(self, decision, **fields):
        gate = ProgramGate()
        gate.predict = Mock(return_value=dspy.Prediction(
            decision=decision, guardrail=fields.get("guardrail", ""),
            safe_response=fields.get("safe_response", "")))
        return gate

    def test_valid_labels_preserve_control_paths(self):
        for label in ("ALLOW", "CLARIFY", "INSUFFICIENT", " allow \n"):
            with self.subTest(label=label):
                result = self.gate(label)("What is required?", evidence())
                self.assertEqual(result.decision, label.strip().upper())
                self.assertEqual(result.allow, label.strip().upper() == "ALLOW")

    def test_mixed_negated_and_malformed_labels_never_allow(self):
        for label in ("DISALLOW", "DO NOT ALLOW", "NOT ALLOW", "ALLOW INSUFFICIENT",
                      "CLARIFY or ALLOW", "ALLOW because evidence exists", "<think>x</think>ALLOW",
                      "", None, 1, ["ALLOW"]):
            with self.subTest(label=label):
                result = self.gate(label)("What is required?", evidence())
                self.assertEqual(result.decision, "INSUFFICIENT")
                self.assertFalse(result.allow)
                self.assertTrue(result.safe_response)

    def test_prediction_failure_and_missing_decision_fail_closed(self):
        for value in (None, dspy.Prediction(), dspy.Prediction(decision={"ALLOW": True})):
            gate = ProgramGate()
            gate.predict = Mock(return_value=value)
            self.assertFalse(gate("Question", evidence()).allow)
        gate.predict = Mock(side_effect=RuntimeError("Synthetic endpoint failure"))
        self.assertFalse(gate("Question", evidence()).allow)

    def test_generated_numbers_never_become_instructions_or_abstention(self):
        for label in ("ALLOW", "CLARIFY", "INSUFFICIENT"):
            result = self.gate(label, guardrail="Require 999 credits.",
                               safe_response="You need 999 credits.")("Question", evidence())
            self.assertNotIn("999", result.guardrail + result.safe_response)
            if label != "ALLOW":
                self.assertTrue(result.safe_response)

    def test_missing_or_malformed_evidence_cannot_be_rescued_by_summary(self):
        for raw in ("", "not JSON", "{}", "[]", "[null]", evidence(""), evidence("   "),
                    *(json.dumps([{"tool_name": "search", "observation": text}]) for text in
                      ("No matching documents found.", "Execution error in search: failed", "Unknown tool: search")),
                    json.dumps([{"tool_name": "finish", "observation": "Completed."}]),
                    json.dumps([{"tool_name": "search", "thought": "6 credits"}]), evidence(None)):
            with self.subTest(raw=raw):
                result = self.gate("ALLOW")("Question", raw, "Summary says 999 credits.")
                self.assertFalse(result.allow)
                self.assertEqual(result.decision, "INSUFFICIENT")

    def test_gate_predictor_gets_observations_not_reasoning(self):
        gate = self.gate("ALLOW")
        raw = json.dumps([{"tool_name": "search", "observation": document_observation("Synthetic rule: 6 credits."),
                           "thought": "999 credits", "tool_args": {"query": "999 credits"}}])
        result = gate("Question", raw, "999 credits")
        self.assertTrue(result.allow)
        inputs = json.dumps(gate.predict.call_args.kwargs)
        self.assertIn("6 credits", inputs)
        self.assertNotIn("999", inputs)


class PlannerTests(unittest.TestCase):
    def planner(self):
        planner = Planner([synthetic_search], max_iterations=3)
        planner.predict = Mock(side_effect=[step("synthetic_search", query="long"),
                                           step("synthetic_search", query="short"), step()])
        planner.summarizer = Mock(return_value=dspy.Prediction(new_summary="Invented requirement: 999 credits."))
        return planner

    def test_raw_evidence_survives_compression_without_model_reasoning(self):
        result = self.planner()("Question", ConversationMemory())
        self.assertIn(synthetic_search("long"), [e["observation"] for e in json.loads(result.evidence)])
        self.assertNotIn("999", result.evidence)
        self.assertNotIn("thought", result.evidence)
        self.assertNotIn("tool_args", result.evidence)

    def test_next_question_and_reset_do_not_reuse_summary(self):
        planner = self.planner()
        planner("Question one", ConversationMemory())
        planner.predict = Mock(return_value=step())
        result = planner("Question two", ConversationMemory())
        self.assertEqual(result.summary, "")
        self.assertEqual(json.loads(result.evidence), [])
        self.assertEqual(planner.inventory.records, {})
        planner.reset()
        self.assertEqual(planner("Question three", ConversationMemory()).summary, "")

    def test_failed_turn_cannot_leave_evidence_for_next_question(self):
        planner = self.planner()
        planner.predict = Mock(side_effect=[step("synthetic_search", query="long"),
                                           step("synthetic_search", query="short"), RuntimeError("LM failed")])
        with self.assertRaises(RuntimeError):
            planner("Failed question", ConversationMemory())
        planner.predict = Mock(return_value=step())
        result = planner("New question", ConversationMemory())
        self.assertEqual(result.summary, "")
        self.assertEqual(json.loads(result.evidence), [])

    def test_failed_unknown_and_finish_tools_are_not_evidence(self):
        for prediction in (step(), step("missing"), step("synthetic_search", wrong_argument="x")):
            planner = Planner([synthetic_search], max_iterations=1)
            planner.predict = Mock(return_value=prediction)
            self.assertEqual(json.loads(planner("Question", ConversationMemory()).evidence), [])


class AgentTests(unittest.TestCase):
    def test_supported_answer_uses_raw_evidence_not_fabricated_intermediates(self):
        app = agent()
        app.planner.predict = Mock(side_effect=[step("synthetic_search", query="long"),
                                               step("synthetic_search", query="short"), step()])
        app.planner.summarizer = Mock(return_value=dspy.Prediction(new_summary="999 credits"))
        app.program_gate.predict = Mock(return_value=dspy.Prediction(decision="ALLOW", guardrail="999 credits"))

        def answer(**inputs):
            self.assertNotIn("999", json.dumps(inputs))
            self.assertIn("6 credits", inputs["trajectory"])
            return cited_answer(**inputs)

        app.synthesizer.predict = Mock(side_effect=answer)
        result = app("What is the synthetic requirement?")
        self.assertEqual(result.response.split(" [")[0], "6 credits.")
        self.assertEqual(result.gate_decision, "ALLOW")
        self.assertIn("trajectory", result.toDict())

    def test_rejected_decisions_and_empty_evidence_skip_synthesis(self):
        for label in ("CLARIFY", "INSUFFICIENT", "DISALLOW", "ALLOW"):
            app = agent()
            app.planner.predict = Mock(return_value=step())
            app.program_gate.predict = Mock(return_value=dspy.Prediction(decision=label, guardrail="999 credits",
                                                                        safe_response="999 credits"))
            app.synthesizer.predict = Mock(side_effect=AssertionError("Must not synthesize"))
            result = app("What is my private grade?")
            self.assertNotEqual(result.gate_decision, "ALLOW")
            self.assertNotIn("999", result.response)
            app.synthesizer.predict.assert_not_called()

    def test_reset_clears_conversation_tool_and_planner_state(self):
        app = agent()
        app.planner.trajectory_summary = "999 credits"
        app.conversation_memory("user", "Old user")
        app.previous_response = "Old answer"
        app.reset()
        self.assertEqual(app.planner.trajectory_summary, "")
        self.assertEqual(app.conversation_memory.history_str(), "")
        self.assertEqual(app.tool_memory.history_str(), "")
        self.assertIsNone(app.previous_response)

    def test_previous_answer_cannot_authorize_a_new_question_without_evidence(self):
        app = agent()
        app.planner.predict = Mock(side_effect=[step("synthetic_search", query="short"), step()])
        app.program_gate.predict = Mock(return_value=dspy.Prediction(decision="ALLOW"))
        app.synthesizer.predict = Mock(side_effect=cited_answer)
        self.assertEqual(app("Synthetic requirement?").response.split(" [")[0], "6 credits.")
        app.planner.predict = Mock(return_value=step())
        app.synthesizer.predict = Mock(side_effect=AssertionError("Old evidence must not authorize synthesis"))
        result = app("What is my private grade?")
        self.assertEqual(result.gate_decision, "INSUFFICIENT")
        app.synthesizer.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
