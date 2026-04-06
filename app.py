"""
app.py — Streamlit UI for Portfolio AI Agent

Bilingual chat interface for querying the rodnm.github.io documentation.
Powered by Google Gemini 2.5 Flash Lite via Pydantic AI.

Run locally:
    uv run streamlit run app.py

Deploy:
    Push to GitHub, connect on share.streamlit.io,
    add GEMINI_API_KEY to app Secrets.
"""

from __future__ import annotations

import asyncio
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import ingest
import logs
import search_agent
from search_tools import SearchTool

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio AI Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Load resources (cached — runs once per session) ───────────────────────────

@st.cache_resource(show_spinner="Loading portfolio documentation...")
def load_resources():
    """Download, index, and initialize the agent. Cached across reruns."""
    index, records = ingest.read_repo_data(chunk=True)
    search_tool = SearchTool(index=index, records=records)
    agent = search_agent.init_agent(search_tool)
    return agent, search_tool


agent, search_tool = load_resources()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Portfolio AI Agent")
    st.markdown(
        "Ask questions about how **Rodrigo Norabuena's** portfolio was built.\n\n"
        "Responds in **Spanish or English** automatically."
    )
    st.divider()

    st.markdown("**Model**")
    st.code("google-gla:gemini-2.5-flash-lite", language=None)

    st.markdown("**Knowledge base**")
    st.markdown(
        "[rodnm/rodnm.github.io](https://github.com/rodnm/rodnm.github.io)\n\n"
        "— `docs/` technical documentation\n\n"
        "— `README.md` overview"
    )

    st.markdown("**Topics covered**")
    st.markdown(
        "- Astro v5 & Islands Architecture\n"
        "- Tailwind CSS v4 styling\n"
        "- GitHub Pages deployment\n"
        "- PWA & Service Worker\n"
        "- React component design\n"
        "- Git workflow & CI/CD"
    )

    st.divider()

    if st.button("🗑️ Clear chat / Limpiar chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Built as the final project for the "
        "[7-Day AI Agents Crash Course](https://alexeygrigorev.com/aihero/) "
        "by Alexey Grigorev."
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🤖 Portfolio AI Agent")
st.caption(
    "Ask me anything about the technical architecture of "
    "[rodnm.github.io](https://github.com/rodnm/rodnm.github.io) "
    "— in **Spanish** or **English**."
)

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

placeholder = "Ask about the portfolio... / Pregunta sobre el portafolio..."

if prompt := st.chat_input(placeholder):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent and display response
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation... / Buscando en la documentación..."):
            start = time.time()
            try:
                result = asyncio.run(agent.run(user_prompt=prompt))
                elapsed = time.time() - start
                response_text = str(result.output)

                st.markdown(response_text)
                st.caption(f"⏱️ {elapsed:.1f}s  •  gemini-2.5-flash-lite")

                # Log the interaction
                try:
                    logs.log_interaction_to_file(agent, result.new_messages())
                except Exception:
                    pass  # logging is non-critical

            except Exception as e:
                response_text = f"⚠️ Error: {e}"
                st.error(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})

# ── Empty state hint ──────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.info(
        "**Example questions / Preguntas de ejemplo:**\n\n"
        "- ¿Qué es el patrón Islands de Astro?\n"
        "- How is the portfolio deployed to GitHub Pages?\n"
        "- ¿Cómo funciona la validación de datos con Zod?\n"
        "- What PWA features does the portfolio support?\n"
        "- ¿Cuáles son los estilos usados en el proyecto?"
    )
