# ChatDKU candidate assignment

Two reliability improvements integrated into the existing public ChatDKU agent: strict evidence gating and source-aware, verifiable citation identities. This is a local document-question-answering harness, not the production university service. **61 deterministic tests pass; M01 still contains inadequately supported claims and C01 unnecessarily requests clarification in the final live evaluation.**

Based on [ChatDKU-Public](https://github.com/Edge-Intelligence-Lab/ChatDKU-Public) commit `a60f11dcc902b9970cad1b8e371bc64feb03e535`. Read the [contribution](CANDIDATE_CONTRIBUTION.md) and [evaluation](EVALUATION_SUMMARY.md).

## Architecture and implemented improvements

The original public harness uses Python, DSPy, LlamaIndex, embedded local Qdrant, vector and BM25-style keyword retrieval, and an OpenAI-compatible LLM endpoint. It provides a CLI, not a web backend. Older Redis/ChromaDB production descriptions do not describe this public baseline.

**Improvement 1 — evidence gating.** The original substring decision parser could accept negated labels, and generated summaries/gate instructions could introduce unsupported policy figures. Exact normalized decisions now fail closed on malformed output or predictor failure. Gate-generated prose never becomes factual instructions. Gate and synthesis use current-question documentary observations rather than planner summaries. Turn rollback/reset protects retries and prevents stale evidence. Narrow English/Chinese clarification handles identifiable missing references without an LLM call.

**Improvement 2 — source identity and capacity.** Structured `RetrievedDocument` records enter a per-question inventory. Synthesis selects explicit IDs in typed answer blocks. Code validates IDs and renders safe filenames and actual PDF page labels; missing/unknown references and fabricated locators reject the draft. Whole-observation and complete-request bounds refuse overflow instead of silently truncating exceptions. Citation validity establishes source identity, **not factual entailment**.

A final refinement adds optional source-filename filtering to the existing keyword tool and deterministic new-evidence/no-progress feedback. Identical successful documentary requests are suppressed within a turn. Source coverage improved, but synthesis/gate errors remain. No extra verifier model, reranker, database or answer-specific rule was added.

```text
Approved files -> extraction/chunking -> embeddings -> embedded Qdrant
Original question -> optional rewrite hint -> bounded DSPy planner
  -> vector / keyword search -> current-question evidence inventory
  -> strict gate -> clarification / insufficiency OR typed synthesis
  -> citation-ID validation + source rendering -> response + turn memory
All LLM requests: capacity preflight. Failed turns: memory rollback.
```

## Requirements and Windows setup

A fresh checkout contains no corpus, index, models, tokenizer cache, credentials or running services. Obtain the approved corpus separately. Install Python 3.13+, uv, and a compatible model runtime; Git is needed if cloning. The evaluated machine used Python 3.13.3, uv 0.13.5, Ollama 0.40.9, an i7-1355U CPU and 16 GB RAM, without a discrete GPU.

From the extracted repository root in PowerShell:

```powershell
uv sync --locked --python 3.13
# REQUIRED for DOCX ingestion; absent from the upstream lockfile.
uv pip install --python .venv/Scripts/python.exe docx2txt==0.9
Copy-Item .env.example .env
```

**DOCX portability warning:** LlamaIndex's DOCX reader imports `docx2txt`, omitted by the upstream manifest/lock. The tested environment uses the pinned `docx2txt==0.9` overlay. This submission preserves the verified lock and documents the overlay because a consistent lock update could not be verified during packaging: the prior uv executable was unavailable. An exact `uv sync` may remove the overlay; reinstall it afterward. Use the venv executables below or `uv run --no-sync`. This is a documented setup limitation, not a self-contained locked installation.

### Local model service

Install Ollama separately. The evaluated model is `qwen3:4b-instruct-2507-q4_K_M`, approximately 2.5 GB to download. It fits the tested CPU machine, but document answers take several minutes. Runtime/model downloads require network access and disk space and are not included here. If missing, run `ollama pull qwen3:4b-instruct-2507-q4_K_M` against the local service.

Start a local server with this environment; do not start a second server on an occupied port:

```powershell
$env:OLLAMA_HOST='127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD='1'
$env:OLLAMA_CONTEXT_LENGTH='16384'
$env:OLLAMA_NUM_PARALLEL='1'
ollama serve
```

Use another terminal for the CLI. After loading, inspect `http://127.0.0.1:11434/api/ps` to verify the actual context. Configure ignored `.env`:

```dotenv
CHATDKU_LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M
CHATDKU_LLM_BASE_URL=http://127.0.0.1:11434/v1
CHATDKU_LLM_TEMPERATURE=0.2
CHATDKU_LLM_MAX_TOKENS=2048
CHATDKU_EMBEDDING_PROVIDER=local
CHATDKU_LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
CHATDKU_DATA_DIR=../private
CHATDKU_DOCUMENT_DIR=../private/corpus
CHATDKU_INDEX_DIR=../private/index
CHATDKU_CACHE_DIR=../private/cache
CHATDKU_CONTEXT_TOKENS=16384
CHATDKU_CONTEXT_SAFETY_TOKENS=1024
CHATDKU_EVIDENCE_MAX_BYTES=24000
CHATDKU_CHUNK_SIZE=512
CHATDKU_CHUNK_OVERLAP=64
CHATDKU_RETRIEVAL_TOP_K=8
```

**Timeout limitation:** `CHATDKU_LLM_TIMEOUT_SECONDS` is parsed and validated by configuration but is not forwarded to DSPy/LiteLLM in the current agent. Setting it does not control model-request timeout; provider/library behavior applies. The README does not promise a configurable inference deadline. No runtime change was made during acceptance.

No API key is required by the tested loopback service. Do not use external LLM/embedding endpoints for private documents without permission. DEBUG logs may expose retrieved text.

### Embeddings and offline use

The E5 model ID downloads public model assets on first use (approximately 500 MB), then computes embeddings locally. For exact reproduction, pre-provision `intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` and set `CHATDKU_LOCAL_EMBEDDING_MODEL` to that local snapshot in your private `.env`. The generic model-ID example is not revision-pinned. Only after assets exist, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for offline operation. Local embedding computation does not upload document content.

### Optional tokenizer export

With the evaluated model installed and Ollama running:

```powershell
.venv/Scripts/python.exe scripts/export_ollama_tokenizer.py --model qwen3:4b-instruct-2507-q4_K_M --output ../private/qwen-counter.json
```

Set `CHATDKU_TOKENIZER_PATH` in `.env` to the exported file's resolved path on your machine; never commit that path or cache. The roughly 9.4 MB export uses local model metadata and existing Transformers/tokenizers dependencies. Model digest, template and tokenizer hash must match. Evaluated model digest: `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`.

Without a matching export, the conservative UTF-8 byte bound can refuse document questions that fit with the validated counter. Answer coverage is therefore reduced. Neither mode establishes exact counting for arbitrary providers/templates. Keep the server context aligned with configuration and recreate the Agent after runtime changes; do not disable bounds to conceal refusals.

## Ingest and run

Put **only approved documents** in the private corpus directory. The evaluated corpus comprised the undergraduate Bulletin PDF and Advising FAQ DOCX, totaling 844 indexed chunks. Neither file is distributed. Never ingest the whole project workspace.

```powershell
.venv/Scripts/agent.exe --help
.venv/Scripts/chatdku-ingest.exe --help
.venv/Scripts/agent.exe --check-embeddings
.venv/Scripts/chatdku-ingest.exe ../private/corpus
.venv/Scripts/agent.exe "What requirements are stated in the supplied documents?"
.venv/Scripts/agent.exe --no-rewrite "Compare the relevant source statements."
.venv/Scripts/agent.exe
```

Help does not prove model or ingestion readiness. Ingestion refuses an existing collection; `--replace` deliberately deletes/rebuilds it. Rebuild after changing embedding model or chunking. Embedded Qdrant needs no server; avoid concurrent processes opening the same local index. On Windows use Ctrl-Z then Enter to end interactive input. See [ingestion](docs/ingestion.md) and [agent details](docs/agent.md).

## Automated tests and evaluation

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m compileall -q src tests scripts
uv pip check --python .venv/Scripts/python.exe
git diff --check
```

Tests use synthetic policies and fake DSPy predictions/retrieval, not private university passages or an inference service. They verify deterministic contracts, not real-model accuracy. Final live runs use the same local CPU-based 4B model, E5 assets and approved index as the baseline. Final DSPy response caches were disabled; baseline cache conditions differ, so timings do not establish speedups.

F01 preserves a correct cited answer; A01 asks for the missing course without inference; U01 abstains on unavailable private records. M01 retrieves the necessary accounting passage but still gives an incompletely supported explanation. C01 retrieves both relevant policy statements but unnecessarily requests clarification. See [evaluation results](EVALUATION_SUMMARY.md). Source-bearing raw traces are retained privately, not distributed.

## Structure and contribution boundaries

- `src/chatdku/agent.py`: orchestration and transactional turn state.
- `src/chatdku/modules/`: planner, gate, synthesizer, evidence, citations, capacity and memory.
- `src/chatdku/tools/`: existing retrieval tools and structured documentary transport.
- `src/chatdku/ingestion/`: existing readers, chunking and index creation.
- `scripts/export_ollama_tokenizer.py`: local counter export.
- `tests/`: six synthetic regression-test modules.

Upstream supplied the harness and ingestion/retrieval architecture. Candidate changes integrate into that workflow rather than a separate demo. AI coding assistance was used for implementation, tests, diagnostics and documentation; the candidate must be able to explain, modify and independently validate the code. See [contribution details](CANDIDATE_CONTRIBUTION.md).

The code-and-documentation task is prepared for reviewer inspection with disclosed limitations. The agent is **not production-ready for academic advising**. Application materials and deadline confirmation remain separate checklist items.
