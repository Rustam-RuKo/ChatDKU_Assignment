"""Integrated reliability contracts using synthetic policies only."""
import json
import unittest
from unittest.mock import Mock
import numpy  # noqa: F401
import dspy
from chatdku.modules.citations import EvidenceInventory, CitationError
from test_evidence_gating import agent, step
from test_source_evidence import document


class StructuredAnswerTests(unittest.TestCase):
    def test_real_adapter_parses_typed_blocks_and_renderer_keeps_locators(self):
        from chatdku.modules.synthesizer import ResponseSynthesisSignature
        registry = EvidenceInventory(); ref = registry.add(document())
        output = "[[ ## response ## ]]\n" + json.dumps([
            {"text": "Synthetic requirement: 6 credits.", "evidence_ids": [ref]}
        ]) + "\n\n[[ ## completed ## ]]"
        prediction = dspy.ChatAdapter().parse(ResponseSynthesisSignature, output)
        result = registry.render_blocks(prediction["response"])
        self.assertIn("PDF page 7", result)
        self.assertIn("6 credits. [" + ref + "]", result)

    def test_invalid_block_support_never_gets_automatic_references(self):
        registry = EvidenceInventory(); ref = registry.add(document())
        for blocks in ([], [{"text": "Claim", "evidence_ids": []}],
                       [{"text": "Claim", "evidence_ids": [ref, "E000000000000"]}],
                       [{"text": "Claim", "evidence_ids": ref}],
                       [{"text": "Claim", "evidence_ids": [ref], "heading": "Uncited claim"}],
                       [{"text": "Claim [E000000000000]", "evidence_ids": [ref]}],
                       [{"text": "On page 99 this is required", "evidence_ids": [ref]}]):
            with self.subTest(blocks=blocks), self.assertRaises(CitationError):
                registry.render_blocks(blocks)

    def test_blocks_preserve_conflicts_but_do_not_prove_entailment(self):
        registry = EvidenceInventory(); a = registry.add(document())
        b = registry.add(document(node="second", name="other.docx", page=None, text="Other scope: 9 credits."))
        result = registry.render_blocks([{"text": "First scope: 6 credits.", "evidence_ids": [a]},
                                         {"text": "Other scope: 9 credits.", "evidence_ids": [b]}])
        self.assertIn("PDF page 7", result)
        self.assertIn("other.docx", result)
        self.assertIn("999 credits", registry.render_blocks([{"text": "999 credits", "evidence_ids": [a]}]))

    def test_agent_renders_explicit_block_support_without_model_brackets(self):
        app = agent()
        app.planner.predict = Mock(side_effect=[step("synthetic_search", query="short"), step()])
        app.program_gate.predict = Mock(return_value=dspy.Prediction(decision="ALLOW"))
        def synthesize(**inputs):
            ref = json.loads(inputs["trajectory"])[0]["id"]
            return dspy.Prediction(response=[{"text": "Synthetic requirement: 6 credits.", "evidence_ids": [ref]}])
        app.synthesizer.predict = Mock(side_effect=synthesize)
        result = app("Synthetic requirement?")
        self.assertEqual(result.citation_status, "verified")
        self.assertIn("Synthetic requirement: 6 credits. [E", result.response)
        self.assertIn("Sources:", result.response)


class ScopeTests(unittest.TestCase):
    def test_clarification_failure_rolls_back_and_retry_records_one_turn(self):
        app = agent()
        original = app.conversation_memory.forward
        def fail_after_append(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("Synthetic memory failure")
        app.conversation_memory.forward = fail_after_append
        with self.assertRaises(RuntimeError):
            app("Can I take this course?")
        self.assertEqual(app.conversation_memory.history_str(), "")
        self.assertIsNone(app.previous_response)
        del app.conversation_memory.forward
        self.assertEqual(app("Can I take this course?").gate_decision, "CLARIFY")
        self.assertEqual(app.conversation_memory.history_str().count("Can I take this course?"), 1)

    def test_tool_memory_retains_ids_not_policy_text_across_turns(self):
        from test_evidence_gating import cited_answer
        app = agent()
        app.planner.predict = Mock(side_effect=[step("synthetic_search", query="short"), step()])
        app.program_gate.predict = Mock(return_value=dspy.Prediction(decision="ALLOW"))
        app.synthesizer.predict = Mock(side_effect=cited_answer)
        app("Synthetic requirement?")
        self.assertIn("evidence_ids", app.tool_memory.history_str())
        self.assertNotIn("6 credits", app.tool_memory.history_str())
        app.planner.predict = Mock(return_value=step())
        result = app("What is my personal GPA?")
        self.assertEqual(result.gate_decision, "INSUFFICIENT")

    def test_original_scope_and_context_reach_planner_and_gate_despite_drift(self):
        app = agent(); app.rewrite_query = True
        app.conversation_memory("user", "I mean SYNTH 123 in the 2030 undergraduate program.")
        question = "Can I take this course under those rules? Compare the conflicting statements, not why they changed."
        app.query_rewriter.predict = Mock(return_value=dspy.Prediction(rewritten_query="Why did policy change?"))
        app.planner.predict = Mock(side_effect=[step("synthetic_search", query="short"), step()])
        app.program_gate.predict = Mock(return_value=dspy.Prediction(decision="ALLOW"))
        from test_evidence_gating import cited_answer
        app.synthesizer.predict = Mock(side_effect=cited_answer)
        result = app(question)
        self.assertEqual(result.gate_decision, "ALLOW")
        self.assertEqual(app.planner.predict.call_args.kwargs["current_user_message"], question)
        self.assertIn("SYNTH 123", app.program_gate.predict.call_args.kwargs["conversation_history"])
        self.assertEqual(app.program_gate.predict.call_args.kwargs["current_user_message"], question)

    def test_missing_explicit_year_program_references_and_chinese(self):
        from chatdku.modules.program_gate import unresolved_reference
        for question, expected in (("Can I apply that academic year's rules?", "academic year"),
                                   ("What are the rules for that program?", "program"),
                                   ("这门课可以选吗？", "课程")):
            with self.subTest(question=question):
                self.assertIn(expected, unresolved_reference(question))
        for question in ("What are the general graduation requirements?", "Can I take this course SYNTH 123?",
                         "What is my current personal GPA?", "Why did the policy change?"):
            self.assertEqual(unresolved_reference(question), "")
        self.assertEqual(unresolved_reference("Can I take this course?", 'Prior user context'), "")

    def test_bounded_planner_keeps_evidence_without_unused_summary_call(self):
        from chatdku.modules.planner import Planner
        from chatdku.modules.conversation_memory import ConversationMemory
        from test_evidence_gating import synthetic_search
        planner = Planner([synthetic_search], max_iterations=3)
        planner.predict = Mock(side_effect=[step("synthetic_search", query="long"),
                                           step("synthetic_search", query="short"), step()])
        planner.summarizer = Mock(side_effect=AssertionError("Unused summary must not cost a call"))
        result = planner("Question", ConversationMemory())
        self.assertEqual(len(json.loads(result.trajectory)), 3)
        self.assertEqual(result.summary, "")
        planner.summarizer.assert_not_called()

    def test_unresolved_course_is_clarified_without_retrieval_or_model_calls(self):
        app = agent(); app.rewrite_query = True
        app.query_rewriter.predict = Mock(side_effect=AssertionError("No model call needed"))
        app.planner.predict = Mock(side_effect=AssertionError("No course to retrieve"))
        result = app("Can I take this course CR/NC?")
        self.assertEqual(result.gate_decision, "CLARIFY")
        self.assertIn("course name or code", result.response)
        app.query_rewriter.predict.assert_not_called()
        app.planner.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
