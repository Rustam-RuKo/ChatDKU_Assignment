# Evaluation summary

The original public baseline and final targeted evaluation use the supplied two-document local corpus (844 chunks), local E5 embeddings and Ollama Qwen `qwen3:4b-instruct-2507-q4_K_M`, a quantized 4B model on an i7-1355U CPU with 16 GB RAM. Context: 16384 tokens; output: 2048; temperature: 0.2. The README records model identities and setup constraints. No document content was sent to an external model service.

## Before and after

Times are seconds. Baseline request counts are observed HTTP completions from local server log intervals, not instrumented logical counts. Final counts are logical requests / observed HTTP completions.

| Case / purpose | Original baseline | Final targeted outcome | Baseline seconds / HTTP | Final seconds / logical / HTTP |
|---|---|---|---:|---:|
| F01: straightforward degree-total fact | Correct answer; document named without precise page citation | Correct supported answer with registered PDF citation | 424.302 / 5 | 538.043 / 6 / 6 |
| M01: multiple population requirements and PE accounting | Wrong PE figures and units despite available policy material | Relevant accounting passage retrieved; earlier eight-credit error absent, but incomplete scoped explanation and insufficient claim support remain | 920.740 / 7 | 528.803 / 6 / 6 |
| A01: unspecified course eligibility | Overbroad advice loses policy scope rather than asking a targeted question | Authored course-name/code clarification, no LLM request | 640.130 / 5 | 7.061 / 0 / 0 |
| C01: conflicting FAQ/Bulletin wording | Partial comparison through the gate's prose; no definitive answer without a reason for change | Both relevant passages retrieved, but unnecessary generic CLARIFY; no synthesis | 745.252 / 6 | 670.204 / 5 / 5 |
| U01: unavailable personal records | Appropriate abstention | Appropriate abstention; no invented private information | 83.743 / 3 | 54.637 / 3 / 3 |

## What the measurements establish

- **Deterministic tests:** 61 passing tests over synthetic policies, fake predictions and fake retrieval. The original 55 tests remain unchanged. Tests cover exact gate parsing, malformed outputs, errors/retries/reset, structured documentary evidence, identity/citation validation, capacity bounds and retrieval progress. Passing tests do not establish real-model factual correctness.
- **Citation identity:** final F01 and M01 pass reference validation; saved registry records match the unchanged index and rendered citations reproduce exactly. C01/A01/U01 have no generated citations. A valid reference does not prove the claim follows from its passage.
- **Source coverage:** F01 has the relevant degree-total passage. M01 has both total requirements and the detailed accounting passage. C01 has both the relevant FAQ eligibility passage and Bulletin exception. Five/five/zero/ten/zero chunks were registered for F01/M01/A01/C01/U01. Presence alone is not sufficiency.
- **Factual support:** F01 is supported. M01 omits the explicit mainland compulsory PE course/credit requirement, combines differently scoped athletics and degree-accounting wording, cites an athletics passage for a cap established elsewhere, and adds an unestablished explanation about remaining credits. It is not a complete factual fix.
- **Clarification/abstention:** A01 is targeted and useful; U01 correctly avoids unavailable records. C01 asks for context already supplied despite retrieved comparison evidence. Its safe output avoids the prior invented precedence/rationale but fails useful answering. The decision-only gate trace cannot reveal its internal reason.
- **Cost:** final total 1798.747 seconds, 20 logical/HTTP requests, 48197 prompt and 2553 completion tokens. No final capacity refusal or truncation occurred. Largest reserved request bound was 11385/16384; all observed prompt counts were below the configured local-counter bound.

## Method and limitations

The original baseline exercised both existing retrievers. Final agent runs used keyword retrieval; separate unchanged-tool diagnostics tested four targeted queries through both keyword and vector search and found the missing policy passages. No-rewrite C01 still failed useful answering, demonstrating that rewriting was not the sole problem. Filename recovery was tested after initial source-filter attempts failed on guessed filenames; those unsuccessful attempts remain preserved privately, not substituted for the final rows.

Final DSPy memory/disk response caches were disabled; server prefix caches may remain. Baseline response-cache conditions differed. Runs are single observations, with CPU startup/load and some concurrent offline validation work; no strong speedup or statistical accuracy claim is justified. The final suite retains the same questions and acceptance criteria. It is a small selected diagnostic set, not comprehensive academic-policy validation.

Raw questions, source-bearing traces, indexed records and drafts remain private and are excluded from the source package. A reviewer can rerun synthetic tests immediately after dependency setup, but reproducing private-corpus results requires separately authorized documents, the matching index/model setup and local services. The final private batch-summary read encountered a Windows default-encoding error after completed results were saved; UTF-8 analysis recovered them without relabeling an agent failure.

**Conclusion:** the submission demonstrates two integrated improvements with testable engineering guarantees and disclosed negative live results. M01 remains inadequately supported; C01 unnecessarily requests clarification. The agent is not a fully reliable academic adviser. Dates alone do not establish institutional precedence, and no such rule is implemented.
