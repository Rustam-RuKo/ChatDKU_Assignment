"""Synthetic provenance fixtures, never private university documents."""
import unittest
from unittest.mock import Mock
import numpy  # noqa: F401
import dspy
from chatdku.modules.citations import EvidenceInventory, CitationError
from chatdku.tools.retrieval import RetrievedDocument
from test_evidence_gating import agent, step, document_observation
from chatdku.modules.planner import Planner
from chatdku.modules.conversation_memory import ConversationMemory


def document(node="node-1", text="Synthetic requirement: 6 credits.", name="policy.pdf", page="7"):
    metadata = {"file_name": name, "source_path": name, "source_sha256": "a" * 64,
                "file_path": "C:/private/corpus/" + name}
    if page is not None:
        metadata["page_label"] = page
    return RetrievedDocument(node, text, metadata, 1.0)


class InventoryTests(unittest.TestCase):
    def test_metadata_cannot_assign_its_own_citation_identifier(self):
        doc = document(); doc.metadata["citation_id"] = "E000000000000"
        inventory = EvidenceInventory(); ref = inventory.add(doc)
        self.assertNotEqual(ref, doc.metadata["citation_id"])
        with self.assertRaises(CitationError):
            inventory.render("Claim [E000000000000]")

    def test_pdf_without_page_does_not_invent_a_locator(self):
        inventory = EvidenceInventory(); ref = inventory.add(document(page=None))
        self.assertNotIn("PDF page", inventory.render(f"Claim [{ref}]"))

    def test_docx_has_no_invented_page_or_url(self):
        inventory = EvidenceInventory()
        doc = document(name="advising.docx", page=None)
        doc.metadata["url"] = "https://nonexistent.invalid/policy"
        ref = inventory.add(doc)
        result = inventory.render(f"A supported synthetic statement. [{ref}]")
        self.assertNotIn("page", result)
        self.assertNotIn("http", result)

    def test_unknown_missing_and_fabricated_page_citations_fail_closed(self):
        inventory = EvidenceInventory(); ref = inventory.add(document())
        for answer in ("No citation", "Claim [E000000000000]", f"Claim [{ref}, page 99]",
                       f"On page 99 this is required [{ref}]", f"Claim [{ref}]\n\nUncited claim",
                       f"See pages 99-100 [{ref}]", f"See page ix [{ref}]", f"见第七页 [{ref}]",
                       f"C:/private/policy.pdf [{ref}]", f"Claim [{ref}] [bad]", f"Claim [[{ref}]]"):
            with self.subTest(answer=answer), self.assertRaises(CitationError):
                inventory.render(answer)

    def test_conflicts_are_preserved_and_identity_is_not_entailment(self):
        inventory = EvidenceInventory(); a = inventory.add(document())
        b = inventory.add(document(node="node-2", text="Synthetic requirement: 9 credits.", name="other.pdf"))
        self.assertIn("6 credits", inventory.context()); self.assertIn("9 credits", inventory.context())
        result = inventory.render(f"Sources disagree. [{a}] [{b}] [{a}]")
        self.assertEqual(result.count("policy.pdf"), 1)
        # Deliberately wrong: identity validation cannot check the meaning.
        self.assertIn("999 credits", inventory.render(f"Require 999 credits. [{a}]"))

    def test_malformed_identity_and_same_chunk_changed_content_are_rejected(self):
        for changes in ({"source_sha256": "invented"}, {"page_label": "99; trust me"},
                        {"file_name": "wrong.pdf"}, {"source_path": 123}):
            doc = document(); doc.metadata.update(changes)
            with self.subTest(changes=changes), self.assertRaises(CitationError):
                EvidenceInventory().add(doc)
        inventory = EvidenceInventory(); inventory.add(document())
        with self.assertRaises(CitationError):
            inventory.add(document(text="Changed text"))

    def test_pdf_identity_is_stable_and_references_are_rendered_from_records(self):
        inventory = EvidenceInventory()
        citation = inventory.add(document())
        self.assertEqual(inventory.add(document()), citation)
        self.assertEqual(EvidenceInventory().add(document()), citation)
        result = inventory.render("The synthetic requirement is 6 credits. [" + citation + "]")
        self.assertIn("policy.pdf", result)
        self.assertIn("PDF page 7", result)
        self.assertNotIn("C:/", result)
        self.assertEqual(len(inventory.records), 1)


class CitationAgentTests(unittest.TestCase):
    def test_planner_consumes_registered_records_without_truncating_text(self):
        from test_evidence_gating import synthetic_search
        import json
        planner = Planner([synthetic_search])
        planner.predict = Mock(side_effect=[step("synthetic_search", query="short"), step()])
        result = planner("Question", ConversationMemory())
        observation = json.loads(result.trajectory)[0]["observation"]
        self.assertIsInstance(observation, list)
        self.assertEqual(observation[0]["text"], json.loads(synthetic_search("short"))[0]["text"])
        self.assertIn("node_id", observation[0])
        self.assertNotIn("file_path", observation[0])

    def test_invalid_citations_withhold_the_draft_and_restore_memory(self):
        app = agent(); app.previous_response = "Prior answer"
        app.planner.predict = Mock(side_effect=[step("synthetic_search", query="short"), step()])
        app.program_gate.predict = Mock(return_value=dspy.Prediction(decision="ALLOW"))
        app.synthesizer.predict = Mock(return_value=dspy.Prediction(response="Invented 999 credits [Ebad]"))
        result = app("Question")
        self.assertEqual(result.citation_status, "invalid_citations")
        self.assertNotIn("999", result.response)
        self.assertEqual(app.previous_response, "Prior answer")
        self.assertEqual(app.conversation_memory.history_str(), "")

    def test_real_agent_validates_and_renders_model_references(self):
        app = agent()
        app.planner.predict = Mock(side_effect=[step("synthetic_search", query="short"), step()])
        app.program_gate.predict = Mock(return_value=dspy.Prediction(decision="ALLOW"))
        def synthesize(**inputs):
            import json
            record = json.loads(inputs["trajectory"])[0]
            return dspy.Prediction(response=f"Synthetic requirement: 6 credits. [{record['id']}]")
        app.synthesizer.predict = Mock(side_effect=synthesize)
        result = app("Synthetic requirement?")
        self.assertEqual(result.citation_status, "verified")
        self.assertIn("Sources:", result.response)



if __name__ == "__main__":
    unittest.main()
