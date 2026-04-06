"""
main.py — CLI interface for Portfolio AI Agent

Loads the portfolio documentation, initializes the Gemini-powered agent,
and runs an interactive question-answering loop in the terminal.

Usage:
    uv run python main.py
    uv run python main.py --no-chunk    # use full documents instead of chunks
    uv run python main.py --top-k 3     # return top 3 search results
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from dotenv import load_dotenv

load_dotenv()  # load .env file if present

import ingest
import logs
import search_agent
from search_tools import SearchTool

# ── Welcome message ───────────────────────────────────────────────────────────

WELCOME = """
==========================================================
      Portfolio AI Agent  --  rodnm.github.io
  Ask anything about the portfolio in Spanish or English
  Type 'exit' / 'salir' / 'quit' to stop
==========================================================
"""

EXIT_COMMANDS = {"exit", "salir", "quit", "q", "bye", "adios", "adios"}


# ── CLI argument parser ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portfolio AI Agent -- bilingual Q&A for rodnm.github.io"
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Disable document chunking (use full documents as records)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of search results to return (default: 5)",
    )
    return parser


# ── Main loop ─────────────────────────────────────────────────────────────────

async def chat_loop(agent, search_tool: SearchTool) -> None:
    """Interactive REPL: read user input -> run agent -> print response -> log."""
    print(WELCOME)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! / Hasta luego!")
            break

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Goodbye! / Hasta luego!")
            break

        print("\nSearching documentation...\n")
        start = time.time()

        try:
            result = await agent.run(user_prompt=question)
            elapsed = time.time() - start

            response_text = str(result.output)
            print(f"Agent:\n{response_text}")
            print(f"\n{'=' * 60}")
            print(f"  Response time: {elapsed:.1f}s")
            print(f"{'=' * 60}\n")

            # Log interaction to file
            try:
                log_path = logs.log_interaction_to_file(agent, result.new_messages())
                print(f"  [logged -> {log_path.name}]\n")
            except Exception as log_err:
                print(f"  [logging skipped: {log_err}]\n")

        except Exception as e:
            print(f"\nError: {e}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    chunk = not args.no_chunk

    print("Loading portfolio documentation...")
    try:
        index, records = ingest.read_repo_data(chunk=chunk)
    except Exception as e:
        print(f"Failed to load data: {e}", file=sys.stderr)
        sys.exit(1)

    search_tool = SearchTool(index=index, records=records, top_k=args.top_k)
    agent = search_agent.init_agent(search_tool)

    asyncio.run(chat_loop(agent, search_tool))


if __name__ == "__main__":
    main()
