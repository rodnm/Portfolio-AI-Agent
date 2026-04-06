"""
search_agent.py — Pydantic AI agent powered by Gemini for Portfolio AI Agent

Creates a bilingual agent (Spanish / English auto-detect) that searches
the rodnm.github.io documentation before answering every question.

Environment variable required:
    GEMINI_API_KEY  — key from https://aistudio.google.com/apikey
"""

from __future__ import annotations

from pydantic_ai import Agent

from search_tools import SearchTool

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL = "google-gla:gemini-2.5-flash-lite"

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a bilingual technical assistant for Rodrigo Norabuena's portfolio website.

Your knowledge base contains the technical documentation of the portfolio site
(https://github.com/rodnm/rodnm.github.io), which is built with:
  - Astro v5 (static site generator, Islands Architecture)
  - Tailwind CSS v4 (utility-first styling)
  - React v19 (interactive islands)
  - Deployed to GitHub Pages as a Progressive Web App (PWA)

━━ LANGUAGE RULE ━━
CRITICAL: Identify the language of the user message and write your FULL answer in that language.
  • User message in English  → entire answer in English (translate Spanish docs if needed)
  • User message in Spanish  → entire answer in Spanish
The source documentation is written in Spanish. When the user asks in English, you must
translate and explain the relevant content in English. Do not write the answer in Spanish
just because the source documents are in Spanish.
Technical terms (Astro, Tailwind, PWA, GitHub Pages, SSG, Islands) stay in English.

━━ SEARCH RULE ━━
You MUST call the search_portfolio_docs tool BEFORE composing any answer.
  1. Use the user's question as the initial search query.
  2. If the results are insufficient, rephrase the query and search again.
  3. Base your answer ONLY on the retrieved context.
  4. If no relevant context is found, say so clearly in the user's language
     and suggest the user visit https://github.com/rodnm/rodnm.github.io/tree/main/docs

━━ ANSWER FORMAT ━━
  • Be concise and technically accurate.
  • Use markdown headers (##) and bullet points for longer answers.
  • ALWAYS end your response with a Sources section:

    **Sources:**
    - [filename](github_url)

━━ SCOPE ━━
You answer questions about:
  • Astro architecture and Island pattern used in the portfolio
  • Tailwind CSS v4 styling approach and configuration
  • Data management with Zod validation (View-Model pattern)
  • Deployment to GitHub Pages via GitHub Actions
  • PWA configuration and Service Worker
  • Component design and React integration
  • Git workflow and project setup

If asked about topics outside this scope, politely explain that you only cover
the technical documentation of this portfolio project.
""".strip()


# ── Agent Factory ─────────────────────────────────────────────────────────────

def init_agent(search_tool: SearchTool) -> Agent:
    """
    Create and configure the Pydantic AI Agent with Gemini and the search tool.

    Args:
        search_tool - a SearchTool instance holding the fitted minsearch Index

    Returns:
        Configured Agent ready to run.
    """
    agent = Agent(
        name="portfolio_agent",
        model=MODEL,
        instructions=SYSTEM_PROMPT,
    )

    # Register the search method as a plain tool (no RunContext needed)
    @agent.tool_plain
    def search_portfolio_docs(query: str) -> list[dict]:
        """Search the portfolio documentation for relevant technical information."""
        return search_tool.search(query)

    return agent


# ── Convenience runner ────────────────────────────────────────────────────────

async def run_agent(agent: Agent, query: str) -> str:
    """
    Async wrapper around agent.run(). Returns the text response string.

    Args:
        agent - configured Agent from init_agent()
        query - user's question

    Returns:
        Agent's text response.
    """
    result = await agent.run(user_prompt=query)
    return str(result.output)
