"""Export the installed Qwen2-pretokenized model's BPE for local counting.

No documents are sent. Example: python scripts/export_ollama_tokenizer.py
--model qwen3:4b-instruct-2507-q4_K_M --output ../private/qwen-counter.json
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

from tokenizers import Regex, Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from transformers.models.qwen2.tokenization_qwen2 import PRETOKENIZE_REGEX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = "http://127.0.0.1:11434"
    tags = json.load(urllib.request.urlopen(base + "/api/tags", timeout=10))
    digest = next(m["digest"] for m in tags["models"] if m["name"] == args.model)
    request = urllib.request.Request(base + "/api/show", headers={"Content-Type": "application/json"},
                                    data=json.dumps(dict(model=args.model, verbose=True)).encode())
    model = json.load(urllib.request.urlopen(request, timeout=30))
    info = model["model_info"]
    if info.get("tokenizer.ggml.pre") != "qwen2" or info.get("tokenizer.ggml.model") != "gpt2":
        raise ValueError("Only the inspected Qwen2 byte-level BPE is supported")
    vocabulary = {text: i for i, text in enumerate(info["tokenizer.ggml.tokens"])}
    merges = [tuple(pair.split(" ")) for pair in info["tokenizer.ggml.merges"]]
    tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=merges))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    # No normalizer or special-token collapsing: count literal text faithfully.
    raw = tokenizer.to_str()
    envelope = dict(model=args.model, model_digest=digest, tokenizer_sha256=sha256(raw.encode()).hexdigest(),
                    tokenizer=raw, pre="qwen2", template_sha256=sha256(model["template"].encode()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(envelope), encoding="utf-8")
    print(json.dumps({k: v for k, v in envelope.items() if k != "tokenizer"}))


if __name__ == "__main__":
    main()
