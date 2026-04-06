"""
logs.py — Interaction logging for Portfolio AI Agent

Saves every agent interaction to a JSON file, following the same pattern
used in the AI Agents Crash Course (Day 5).

Log files are written to the directory specified by the LOGS_DIRECTORY
environment variable (default: ./logs).
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter

# ── Config ────────────────────────────────────────────────────────────────────

LOG_DIR = Path(os.getenv("LOGS_DIRECTORY", "logs"))
LOG_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serializer(obj):
    """JSON serializer that handles datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} is not JSON serializable")


def _log_entry(agent, messages, source: str = "user") -> dict:
    """Build a structured log dict from an agent and its message history."""
    tools = []
    for ts in agent.toolsets:
        tools.extend(ts.tools.keys())

    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)

    return {
        "agent_name": agent.name,
        "system_prompt": agent._instructions,
        "provider": agent.model.system,
        "model": agent.model.model_name,
        "tools": tools,
        "messages": dict_messages,
        "source": source,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def log_interaction_to_file(agent, messages, source: str = "user") -> Path:
    """
    Save a complete agent interaction to a JSON file.

    Args:
        agent    - Pydantic AI Agent instance
        messages - result.new_messages() from agent.run()
        source   - origin of the query: "user" | "ai-generated"

    Returns:
        Path to the created log file.
    """
    entry = _log_entry(agent, messages, source)

    # Build filename from last message timestamp + random hex
    last_ts = entry["messages"][-1].get("timestamp")
    if isinstance(last_ts, datetime):
        ts_str = last_ts.strftime("%Y%m%d_%H%M%S")
    elif isinstance(last_ts, str):
        try:
            dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            ts_str = dt.strftime("%Y%m%d_%H%M%S")
        except ValueError:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    rand_hex = secrets.token_hex(3)
    filename = f"{entry['agent_name']}_{ts_str}_{rand_hex}.json"
    filepath = LOG_DIR / filename

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, default=_serializer, ensure_ascii=False)

    return filepath


def load_log_file(log_file: str | Path) -> dict:
    """Load a JSON log file and return its content as a dict."""
    with open(log_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["log_file"] = str(log_file)
    return data


def list_log_files(source_filter: str | None = None) -> list[Path]:
    """
    Return all log files in LOG_DIR, sorted newest first.

    Args:
        source_filter - if provided, only return logs matching this source value
                        e.g. "user" or "ai-generated"
    """
    files = sorted(LOG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if source_filter is None:
        return files

    # Filter by source field inside each file
    filtered = []
    for f in files:
        try:
            data = load_log_file(f)
            if data.get("source") == source_filter:
                filtered.append(f)
        except Exception:
            continue
    return filtered
