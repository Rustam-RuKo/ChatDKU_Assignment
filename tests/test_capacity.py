"""Request-level safety checks with no provider traffic."""
import unittest
import tempfile
from pathlib import Path
import io
import json
from hashlib import sha256
from unittest.mock import patch
import numpy  # noqa: F401
import dspy
from chatdku.modules.capacity import BudgetedLM, CapacityError, request_bound, load_counter
from chatdku.modules.planner import Planner
from chatdku.modules.conversation_memory import ConversationMemory
from test_evidence_gating import synthetic_search, step, agent


class CapacityTests(unittest.TestCase):
    def test_counter_requires_matching_asset_model_and_runtime_template(self):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        raw = Tokenizer(BPE(vocab={"x": 0}, merges=[])).to_str()
        envelope = dict(model="fake", pre="qwen2", model_digest="digest", tokenizer=raw,
                        tokenizer_sha256=sha256(raw.encode()).hexdigest(),
                        template_sha256=sha256(b"template").hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counter.json"; path.write_text(json.dumps(envelope))
            def replies(template):
                return [io.BytesIO(json.dumps({"models": [{"name": "fake", "digest": "digest"}]}).encode()),
                        io.BytesIO(json.dumps({"template": template}).encode())]
            with patch("urllib.request.urlopen", side_effect=replies("template")):
                self.assertIsNotNone(load_counter(path, "openai/fake", "http://127.0.0.1:11434/v1"))
            with patch("urllib.request.urlopen", side_effect=replies("changed template")):
                self.assertIsNone(load_counter(path, "openai/fake", "http://127.0.0.1:11434/v1"))
            envelope["tokenizer_sha256"] = "corrupted"; path.write_text(json.dumps(envelope))
            with patch("urllib.request.urlopen", side_effect=AssertionError("No network")) as network:
                self.assertIsNone(load_counter(path, "openai/fake", "http://127.0.0.1:11434/v1"))
                network.assert_not_called()

    def test_untrusted_counter_configuration_falls_back_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counter.json"
            path.write_text('{"model":"wrong-model"}')
            with patch("urllib.request.urlopen", side_effect=AssertionError("No metadata call")) as network:
                self.assertIsNone(load_counter(path, "openai/fake", "http://127.0.0.1:11434/v1"))
                self.assertIsNone(load_counter(path, "openai/fake", "https://external.invalid/v1"))
                network.assert_not_called()

    def test_cumulative_overflow_keeps_all_ids_and_stops_planning(self):
        planner = Planner([synthetic_search], evidence_max_bytes=1000)
        planner.predict = unittest.mock.Mock(side_effect=[step("synthetic_search", query="short"),
                                                          step("synthetic_search", query="long"), step()])
        with self.assertRaises(CapacityError):
            planner("Question", ConversationMemory())
        self.assertEqual(planner.predict.call_count, 2)
        self.assertEqual(len(planner.inventory.records), 2)

    def test_synthesis_history_cannot_bypass_request_guard(self):
        app = agent()
        with patch.object(dspy.LM, "__call__", side_effect=AssertionError("No request")) as provider:
            with self.assertRaises(CapacityError):
                app.synthesizer("Question", conversation_history="x" * 20000)
            provider.assert_not_called()

    def test_real_dspy_rewrite_guard_refuses_huge_chinese_context_without_provider_call(self):
        app = agent(); app.rewrite_query = True
        app.previous_response = "Old response"
        app.planner.inventory.add_observation(synthetic_search("short"))
        app.planner.predict = unittest.mock.Mock()
        with patch.object(dspy.LM, "__call__", side_effect=AssertionError("No provider traffic")) as provider:
            result = app("中" * 20000)
        self.assertEqual(result.citation_status, "capacity")
        self.assertEqual(result.omitted_evidence_ids, [])
        self.assertEqual(app.previous_response, "Old response")
        self.assertEqual(app.conversation_memory.history_str(), "")
        app.planner.predict.assert_not_called(); provider.assert_not_called()

    def test_capacity_in_gate_propagates_to_explicit_agent_refusal(self):
        app = agent()
        app.planner.predict = unittest.mock.Mock(side_effect=[step("synthetic_search", query="short"), step()])
        app.program_gate.predict = unittest.mock.Mock(side_effect=CapacityError("model request", 20000, 16384))
        app.synthesizer.predict = unittest.mock.Mock()
        result = app("Question")
        self.assertEqual(result.citation_status, "capacity")
        self.assertEqual(len(result.omitted_evidence_ids), 1)
        app.synthesizer.predict.assert_not_called()
        self.assertEqual(app.planner.inventory.records, {})

    def test_observation_capacity_rolls_back_memory_and_names_all_omitted_ids(self):
        app = agent(); app.planner.evidence_max_bytes = 100
        app.planner.predict = unittest.mock.Mock(return_value=step("synthetic_search", query="short"))
        app.program_gate.predict = unittest.mock.Mock(); app.synthesizer.predict = unittest.mock.Mock()
        result = app("Question")
        self.assertEqual(result.citation_status, "capacity")
        self.assertTrue(result.omitted_evidence_ids)
        self.assertEqual(app.conversation_memory.history_str(), "")
        app.program_gate.predict.assert_not_called(); app.synthesizer.predict.assert_not_called()

    def test_multibyte_and_schema_overhead_and_unsupported_requests(self):
        plain = [{"role": "user", "content": "abc"}]
        chinese = [{"role": "user", "content": "中文文"}]
        self.assertGreater(request_bound(chinese), request_bound(plain))
        self.assertGreater(request_bound(plain, {"schema": "x" * 100}), request_bound(plain))
        for messages in ([{"role": "user", "content": [{"image": "x"}]}],
                         [{"role": "tool", "content": "x"}], []):
            with self.assertRaises(CapacityError):
                request_bound(messages)

    def test_local_counter_still_reserves_template_schema_and_output_space(self):
        class Counter:
            def encode(self, text):
                return type("Encoding", (), {"ids": list(range(len(text)))})()
        messages = [{"role": "user", "content": "中文"}]
        self.assertEqual(request_bound(messages, counter=Counter()), 2 + 4 + 256)
        self.assertGreater(request_bound(messages), request_bound(messages, counter=Counter()))

    def test_oversized_first_observation_stops_before_second_model_call(self):
        planner = Planner([synthetic_search], evidence_max_bytes=100)
        planner.predict = unittest.mock.Mock(return_value=step("synthetic_search", query="long"))
        with self.assertRaises(CapacityError):
            planner("Question", ConversationMemory())
        self.assertEqual(planner.predict.call_count, 1)
        self.assertEqual(len(planner.inventory.records), 1)

    def test_formatted_messages_and_output_reservation_are_checked_before_provider(self):
        messages = [{"role": "system", "content": "instructions"}, {"role": "user", "content": "中文规则"}]
        bound = request_bound(messages)
        lm = BudgetedLM("openai/fake", context_tokens=bound + 32 + 10, safety_tokens=10, max_tokens=32)
        with patch.object(dspy.LM, "__call__", return_value=["ok"]) as provider:
            self.assertEqual(lm(messages=messages), ["ok"])
            messages[-1]["content"] += "x"
            with self.assertRaises(CapacityError):
                lm(messages=messages)
            self.assertEqual(provider.call_count, 1)


if __name__ == "__main__":
    unittest.main()
