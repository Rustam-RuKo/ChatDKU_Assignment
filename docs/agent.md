# Agent

The CLI agent uses the embedded Qdrant collection created by ingestion. It is a local harness: it has no backend API, tracing service, reranker, or external application database.

## Configure the language model

Set an OpenAI-compatible model endpoint in `.env`:

```dotenv
CHATDKU_LLM_MODEL=Qwen/Qwen3-4B
CHATDKU_LLM_BASE_URL=http://localhost:8000/v1
# CHATDKU_LLM_API_KEY=
```

An API key is optional for local servers. Set `CHATDKU_LLM_API_KEY` when the endpoint requires one. This repository does not provide a language-model endpoint, API key, or source documents.

Build an index before starting the agent. The retrieval tools raise a clear error if the configured Qdrant collection does not exist.

## Run it

Start an interactive session:

```bash
uv run --no-sync agent
```

Ask a single question and exit:

```bash
uv run --no-sync agent "What is ChatDKU?"
```

By default the agent rewrites a question using conversation and tool memory before retrieval. Use the original wording directly when debugging a query:

```bash
uv run --no-sync agent --no-rewrite "ChatDKU"
```

## Retrieval and response flow

1. The query rewriter incorporates relevant conversation and tool history.
2. The planner has up to three planning steps, including any finish action.
3. `vector_search` takes `semantic_query` for conceptual similarity search.
4. `keyword_search` takes `keyword_query` for exact names, phrases, and codes.
5. The program gate attempts to assess evidence sufficiency; its semantic classification can be wrong. Exact decision parsing and empty-evidence checks are deterministic.
6. The synthesizer creates a response from the approved evidence.

The planner receives each tool's exact argument schema and is instructed not to invent aliases such as `query`, `keywords`, or `search_term`.

## Inspect context and tool calls

Enable agent diagnostics with:

```bash
CHATDKU_LOG_LEVEL=DEBUG uv run --no-sync agent
```

For every question, the agent logs the context inputs given to query rewriting, planning, and response synthesis. It also logs the planner trajectory, tool arguments, retrieved observations, gate decision, and final response. The debug context contains user messages and retrieved document text, so use it only in a trusted terminal.

The CLI keeps `httpx`, `httpcore`, and LlamaIndex's own debug output quiet so the useful ChatDKU context logs remain readable.
# Evaluated context-capacity configuration

The capacity guard is evaluated for local text-only Ollama 0.40.9 with
`qwen3:4b-instruct-2507-q4_K_M`, a 16384-token server context, 2048 output tokens,
and a 1024-token safety margin. It is not a universal tokenizer or provider
guarantee. Match `CHATDKU_CONTEXT_TOKENS` to the server's actual context size.

For this configuration, set `CHATDKU_LLM_BASE_URL=http://127.0.0.1:11434/v1`
and `CHATDKU_LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M`. Export the **already
installed** model's tokenizer to a private directory outside the repository:

```powershell
python scripts/export_ollama_tokenizer.py --model qwen3:4b-instruct-2507-q4_K_M --output ../private/qwen-counter.json
```

Set `CHATDKU_TOKENIZER_PATH` in the ignored local `.env` to the absolute exported
path. The export uses local metadata, existing dependencies, and about 9.4 MB;
it downloads no model and sends no documents. Runtime model/template identities
and the cache hash must match. Restart the Agent after changing model/template.
Other endpoints/models require validation before relying on the capacity bound.

Without a valid cache the guard uses UTF-8 bytes as a conservative token bound.
This preserves refusal safety for the inspected byte-level model but was
impractical for the evaluated document questions: all four were refused in the
earlier byte-only run. Do not disable the guard or treat this fallback as equal
answer coverage. `CHATDKU_EVIDENCE_MAX_BYTES=24000` additionally bounds complete
raw retrieval results, including repeated passages and metadata. Overflows
refuse the entire turn and report omitted IDs; no clauses are trimmed to fit.

The final answer's `citation_status=verified` means registered-reference identity
passed validation, not that the policy interpretation was proven correct.
Synthesis now returns internal typed blocks with explicit supporting IDs; the
public Agent still returns rendered text in `response`. Model-generated headings
and reference lists are unnecessary. Missing/unknown support rejects the draft;
no additional verification LLM is called.

For the evaluated Windows setup and required `docx2txt==0.9` overlay, follow the [main README](../README.md). `uv run --no-sync` preserves the manually installed overlay. A fresh checkout requires separately provisioned models, corpus, configuration and services.
