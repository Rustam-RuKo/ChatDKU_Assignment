# My contribution to ChatDKU

For this assignment, I focused on making the existing ChatDKU agent more reliable when answering questions from documents. My submission builds on the public repository at commit `a60f11dcc902b9970cad1b8e371bc64feb03e535`. I kept its DSPy workflow, CLI, ingestion pipeline, Qdrant index, and keyword and vector retrieval tools. I do not claim those existing components as my own work.

I chose two areas: preventing generated reasoning from becoming authoritative evidence, and making citations traceable to retrieved passages. I wanted focused changes that could be tested and explained without replacing the architecture.

## 1. Evidence gating and turn isolation

The original workflow could accept incorrect gate decisions through substring matching. Generated summaries and gate instructions could also introduce unsupported policy figures. Failed turns could leave state that affected later questions.

With AI coding assistance, I developed changes that:

- Accept only normalized `ALLOW`, `CLARIFY`, and `INSUFFICIENT` decisions, and fail closed on malformed predictions or prediction errors.
- Separate gate decisions from factual evidence and prevent gate-generated policy figures from becoming factual instructions.
- Validate documentary observations instead of treating successful tool execution or generated planner text as evidence.
- Reset question-specific evidence and restore memory after failed turns.
- Provide authored clarification for narrowly identifiable missing information without another model call.

The main implementation is in `agent.py`, `modules/program_gate.py`, and `modules/evidence.py`. Synthetic tests cover decision parsing, invalid observations, unsupported gate numbers, reset behavior, retries, and supported answers. These protections address specific control and state failures; they do not guarantee that the model interprets every passage correctly.

## 2. Source-aware evidence and citation identity

I wanted a reviewer to be able to identify the passage behind a citation. Formatted retrieval text alone made that difficult, and a plausible source label did not prove that the source had actually been retrieved.

The implementation preserves the existing `RetrievedDocument` fields, assigns stable evidence IDs, and retains distinct passages from conflicting sources. Synthesis selects evidence IDs, and the renderer checks references and produces source labels from metadata. Unknown references and invented locators are rejected. DOCX passages do not receive invented page numbers.

Capacity checks bound complete observations and formatted model requests. When evidence cannot fit, the agent withholds the answer rather than silently cutting policy clauses. A validated local tokenizer export allows more precise budgeting, with a conservative fallback when validation is unavailable.

The main files are `tools/retrieval.py`, `modules/citations.py`, `modules/synthesizer.py`, and `modules/capacity.py`. A valid citation establishes source identity, not factual support: an answer can cite the right passage and still misinterpret it.

## Focused follow-up changes

The planner retains the original question separately from the rewritten retrieval hint. Source filtering, filename recovery feedback, repeated-request suppression, and evidence-progress reporting reuse the existing tools and evidence inventory. Unused summary generation was removed, and cross-turn tool memory stores evidence IDs rather than duplicate passage text. No evaluation question or expected policy answer is hard-coded into production logic.

## How I evaluated the work

The recorded acceptance suite passed 61 deterministic tests across six modules, using synthetic policies and fake predictions or retrieval outputs. Clean-package checks also covered source compilation, CLI help, and dependency consistency. I distinguish those checks from live factual accuracy.

I personally ran additional local comparisons and inspected the DSPy traces in Phoenix, with AI guidance. In the original agent, I traced a wrong credit-counting claim from planner reasoning into the final answer even though retrieval had returned the relevant restriction. In the improved agent, the same broad question exceeded the evidence budget and returned an abstention. A narrower, single-source question produced a correct answer with a supporting citation, but its planner was already correct. That run therefore does not prove that downstream safeguards corrected faulty reasoning.

The earlier recorded evaluation also retains unresolved M01 and C01 failures. I have not treated a passing test suite, a valid citation, or a refusal as proof that factual reliability is solved. The [evaluation summary](EVALUATION_SUMMARY.md) gives the recorded results and their limitations.

## My role and AI assistance

I used AI assistance for brainstorming, implementation, test development, setup and debugging, and documentation. I also used it to help me run Phoenix comparisons.


## Remaining limitations

The local model can still mix source claims and misinterpret policy figures. Broad searches can collect enough evidence to trigger abstention, and source-name mistakes add retries. CPU inference is slow. The submission demonstrates focused reliability protections and a reproducible investigation; it is not a production-ready academic adviser.
