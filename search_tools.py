"""
search_tools.py — Search engine wrapper for Portfolio AI Agent

Wraps a minsearch Index inside a SearchTool class whose .search() method
is registered as a Pydantic AI tool in search_agent.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from minsearch import Index


# ── SearchTool ────────────────────────────────────────────────────────────────

@dataclass
class SearchTool:
    """
    Encapsulates a fitted minsearch Index and exposes a search() method
    designed to be registered as a Pydantic AI tool.

    Attributes:
        index   - fitted minsearch.Index built by ingest.read_repo_data()
        records - raw list of all document records (used for metadata lookup)
        top_k   - number of results to return (default from env SEARCH_TOP_K or 5)
        boost   - minsearch field boost weights
    """

    index: Index
    records: list[dict] = field(default_factory=list)
    top_k: int = field(default_factory=lambda: int(os.getenv("SEARCH_TOP_K", "5")))
    boost: dict = field(
        default_factory=lambda: {"title": 2.0, "text": 1.0}
    )

    def search(self, query: str) -> list[dict]:
        """
        Search the portfolio documentation for information relevant to the query.

        Use this tool before composing every answer. Call it with the user's
        question or key terms from it. If the first search returns insufficient
        context, rephrase and call again.

        Args:
            query: the search terms or question to look up in the documentation

        Returns:
            A list of matching document chunks. Each chunk contains:
            - title: document title
            - text: the matching content excerpt
            - github_url: direct link to the source file on GitHub
            - doc_id: document identifier
        """
        results = self.index.search(
            query,
            boost_dict=self.boost,
            num_results=self.top_k,
        )
        # Return only the fields the agent needs (keep payload small)
        return [
            {
                "title": r.get("title", ""),
                "text": r.get("text", ""),
                "github_url": r.get("github_url", ""),
                "doc_id": r.get("doc_id", ""),
            }
            for r in results
        ]
