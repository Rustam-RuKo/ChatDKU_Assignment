"""Conservative bounds for the pinned text-only Ollama/Qwen workflow."""
import json
from hashlib import sha256
import logging
from pathlib import Path
import urllib.request

import dspy


class CapacityError(RuntimeError):
    def __init__(self, stage: str, observed: int = 0, limit: int = 0):
        super().__init__(f"{stage}: capacity {observed} exceeds/does not satisfy {limit}")
        self.stage, self.observed, self.limit = stage, observed, limit


def load_counter(path, model, api_base):
    """Use a local GGUF-derived counter only for the matching local runtime.

    Any missing/mismatched asset or unavailable identity check falls back to
    the byte bound. These metadata requests contain no user/document text.
    """
    if not path:
        return None
    try:
        if api_base not in {"http://127.0.0.1:11434/v1", "http://localhost:11434/v1"}:
            raise ValueError("Counter is restricted to the inspected local Ollama endpoint")
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        name = model.removeprefix("openai/")
        if (envelope["model"] != name or envelope["pre"] != "qwen2" or
                sha256(envelope["tokenizer"].encode()).hexdigest() != envelope["tokenizer_sha256"]):
            raise ValueError("Counter identity mismatch")
        base = api_base.removesuffix("/v1")
        tags = json.load(urllib.request.urlopen(base + "/api/tags", timeout=5))
        digest = next(item["digest"] for item in tags["models"] if item["name"] == name)
        request = urllib.request.Request(base + "/api/show", headers={"Content-Type": "application/json"},
                                        data=json.dumps(dict(model=name)).encode())
        info = json.load(urllib.request.urlopen(request, timeout=5))
        if digest != envelope["model_digest"] or sha256(info["template"].encode()).hexdigest() != envelope["template_sha256"]:
            raise ValueError("Runtime identity mismatch")
        from tokenizers import Tokenizer
        return Tokenizer.from_str(envelope["tokenizer"])
    except Exception:
        logging.getLogger(__name__).warning("Local counter unavailable or mismatched; using conservative UTF-8 byte bound")
        return None


def request_bound(messages: list[dict], response_format=None, *, counter=None) -> int:
    """Upper bound, NOT an exact token estimate.

    Qwen's GPT-2 byte-level BPE cannot emit more ordinary tokens than input
    UTF-8 bytes. Reserve 256 tokens per message
    plus the configured margin for the inspected ChatML template. This
    contract is for text messages; model/template changes require review.
    """
    if not isinstance(messages, list) or not messages:
        raise CapacityError("unsupported request")
    for message in messages:
        if (not isinstance(message, dict) or set(message) - {"role", "content"}
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)):
            raise CapacityError("unsupported request")
    try:
        count = (lambda text: len(counter.encode(text).ids)) if counter is not None else (lambda text: len(text.encode("utf-8")))
        size = sum(count(message["content"]) for message in messages)
        size += count(json.dumps(response_format, ensure_ascii=False))
    except (TypeError, ValueError, UnicodeError) as error:
        raise CapacityError("unsupported request") from error
    return size + 256 * len(messages)


class BudgetedLM(dspy.LM):
    """Check DSPy's fully formatted messages, including fallback requests."""
    def __init__(self, *args, context_tokens=16384, safety_tokens=1024, tokenizer_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_tokens = context_tokens
        self.safety_tokens = safety_tokens
        self.counter = load_counter(tokenizer_path, self.model, self.kwargs.get("api_base"))
        self.count_mode = "local_qwen_bpe" if self.counter is not None else "utf8_bytes"

    def _check(self, prompt, messages, kwargs):
        merged = self.kwargs | kwargs
        if merged.get("tools") or merged.get("functions") or (prompt is not None and not isinstance(prompt, str)):
            raise CapacityError("unsupported request")
        messages = messages if messages is not None else [{"role": "user", "content": prompt or ""}]
        output = merged.get("max_completion_tokens") or merged.get("max_tokens")
        if not isinstance(output, int) or isinstance(output, bool) or output < 1:
            raise CapacityError("missing output reservation")
        observed = request_bound(messages, merged.get("response_format"), counter=self.counter) + output + self.safety_tokens
        if observed > self.context_tokens:
            raise CapacityError("model request", observed, self.context_tokens)

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        self._check(prompt, messages, kwargs)
        return super().__call__(prompt, messages=messages, **kwargs)

    async def acall(self, prompt=None, *, messages=None, **kwargs):
        self._check(prompt, messages, kwargs)
        return await super().acall(prompt, messages=messages, **kwargs)
