"""Local command-line interface for the ChatDKU agent harness."""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local ChatDKU agent a question.")
    parser.add_argument("question", nargs="*", help="Question to answer. Omit for an interactive session.")
    parser.add_argument("--no-rewrite", action="store_true", help="Send the original question directly to the planner.")
    parser.add_argument(
        "--check-embeddings",
        action="store_true",
        help="Send one test embedding request and exit.",
    )
    args = parser.parse_args()

    from .config import get_settings

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # LlamaIndex internals quiet so DEBUG output remains useful to an operator.
    for logger_name in ("httpcore", "httpx", "llama_index"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    if args.check_embeddings:
        from .setup import verify_embedding_provider

        try:
            dimension = verify_embedding_provider(settings)
        except RuntimeError as error:
            raise SystemExit(f"Embedding provider check failed: {error}") from error
        print(f"Embedding provider is ready; vector dimension: {dimension}")
        return

    from .agent import Agent

    agent = Agent(rewrite_query=not args.no_rewrite)
    if args.question:
        print(agent(" ".join(args.question)).response)
        return

    print("ChatDKU local harness. Press Ctrl-D to exit.")
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            print()
            return
        if question:
            print(f"ChatDKU: {agent(question).response}")
