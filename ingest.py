"""
ingest.py — Data pipeline for Portfolio AI Agent

Downloads rodnm/rodnm.github.io as a zip archive, extracts all .md files
from docs/, README.md, and src/content/blog/, parses frontmatter, optionally
chunks the documents with a sliding window, and builds a minsearch index.

Usage:
    python ingest.py          # test run, prints stats
"""

import io
import os
import zipfile

import frontmatter
import requests
from minsearch import Index
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

REPO_OWNER = "rodnm"
REPO_NAME = "rodnm.github.io"
REPO_ZIP_URL = os.getenv(
    "REPO_ZIP_URL",
    f"https://codeload.github.com/{REPO_OWNER}/{REPO_NAME}/zip/refs/heads/main",
)
GITHUB_BASE = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/blob/main/"
ZIP_PREFIX = f"{REPO_NAME}-main/"

# Only index .md files under these paths (relative to repo root)
TARGET_PREFIXES = (
    "docs/",
    "README.md",
    "src/content/blog/",
)


# ── Download ─────────────────────────────────────────────────────────────────

def download_repo_zip(url: str = REPO_ZIP_URL) -> bytes:
    """Download the repository zip archive from GitHub. Returns raw bytes."""
    print(f"Downloading {REPO_OWNER}/{REPO_NAME} from GitHub...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


# ── Extract ───────────────────────────────────────────────────────────────────

def extract_md_files(zip_bytes: bytes) -> list[dict]:
    """
    Open zip from bytes and return a list of dicts for every qualifying .md file.

    Each dict contains:
        zip_path      - original path inside the zip
        relative_path - path relative to repo root (without zip prefix)
        github_url    - full GitHub blob URL for the file
        raw_content   - decoded text content
    """
    results = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            name = info.filename

            # Only .md and .mdx files
            if not (name.lower().endswith(".md") or name.lower().endswith(".mdx")):
                continue

            # Strip the zip archive prefix (e.g. "rodnm.github.io-main/")
            if not name.startswith(ZIP_PREFIX):
                continue
            relative_path = name[len(ZIP_PREFIX):]

            # Only keep files in the target directories
            if not any(relative_path.startswith(p) for p in TARGET_PREFIXES):
                continue

            try:
                with zf.open(info) as f:
                    raw_content = f.read().decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"  Warning: could not read {name}: {e}")
                continue

            results.append(
                {
                    "zip_path": name,
                    "relative_path": relative_path,
                    "github_url": GITHUB_BASE + relative_path,
                    "raw_content": raw_content,
                }
            )

    return results


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_markdown(file_info: dict) -> dict:
    """
    Parse frontmatter and body from a raw markdown file dict.

    Adds to the dict:
        title         - from frontmatter["title"] or filename stem
        body          - markdown body without frontmatter
        doc_id        - slug derived from relative_path
        fm            - raw frontmatter dict
    """
    post = frontmatter.loads(file_info["raw_content"])
    body = post.content.strip()
    metadata = dict(post.metadata)

    # Derive title: prefer frontmatter, fall back to filename stem
    stem = file_info["relative_path"].rsplit("/", 1)[-1]
    stem = stem.replace(".md", "").replace(".mdx", "")
    title = str(metadata.get("title", stem.replace("-", " ").replace("_", " ")))

    # Stable doc_id from relative path
    doc_id = (
        file_info["relative_path"]
        .replace("/", "_")
        .replace(".mdx", "")
        .replace(".md", "")
    )

    return {
        **file_info,
        "title": title,
        "body": body,
        "doc_id": doc_id,
        "fm": metadata,
    }


# ── Chunking ──────────────────────────────────────────────────────────────────

def sliding_window(text: str, window_size: int = 300, step: int = 150) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Args:
        text        - input text
        window_size - words per chunk
        step        - words to advance between chunks

    Returns list of text strings (each chunk is window_size words or fewer).
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    for i in range(0, len(words), step):
        chunk = words[i : i + window_size]
        chunks.append(" ".join(chunk))
        if i + window_size >= len(words):
            break

    return chunks if chunks else [text]


def chunk_documents(
    docs: list[dict],
    chunk: bool = True,
    window_size: int = 300,
    step: int = 150,
) -> list[dict]:
    """
    Convert parsed documents into indexable records.

    If chunk=False: one record per document (text = full body).
    If chunk=True:  one record per sliding-window chunk.

    Each output record has:
        chunk_id, doc_id, title, text, relative_path, github_url
    """
    records = []

    for doc in docs:
        base = {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "relative_path": doc["relative_path"],
            "github_url": doc["github_url"],
        }

        if not chunk:
            records.append(
                {
                    **base,
                    "chunk_id": doc["doc_id"],
                    "text": doc["body"],
                }
            )
        else:
            chunks = sliding_window(doc["body"], window_size=window_size, step=step)
            for idx, chunk_text in enumerate(chunks):
                records.append(
                    {
                        **base,
                        "chunk_id": f"{doc['doc_id']}_chunk_{idx}",
                        "text": chunk_text,
                    }
                )

    return records


# ── Index ─────────────────────────────────────────────────────────────────────

def build_index(records: list[dict]) -> Index:
    """Create and fit a minsearch Index from the document records."""
    index = Index(
        text_fields=["title", "text"],
        keyword_fields=["doc_id", "chunk_id", "relative_path"],
    )
    index.fit(records)
    return index


# ── Public API ────────────────────────────────────────────────────────────────

def read_repo_data(
    chunk: bool = True,
    window_size: int = 300,
    step: int = 150,
    url: str = REPO_ZIP_URL,
) -> tuple[Index, list[dict]]:
    """
    Full pipeline: download → extract → parse → chunk → index.

    Returns:
        (index, records) — the fitted minsearch Index and the raw record list.

    This is the single public entry point imported by search_tools, main, and app.
    """
    zip_bytes = download_repo_zip(url)

    raw_files = extract_md_files(zip_bytes)
    print(f"Found {len(raw_files)} markdown files in repo")

    parsed = [parse_markdown(f) for f in raw_files]

    records = chunk_documents(parsed, chunk=chunk, window_size=window_size, step=step)
    print(f"Created {len(records)} index records (chunking={'on' if chunk else 'off'})")

    index = build_index(records)
    print("Index built successfully")

    return index, records


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    index, records = read_repo_data(chunk=True)

    print("\n--- Sample records ---")
    for rec in records[:3]:
        print(f"  [{rec['chunk_id']}]")
        print(f"  Title : {rec['title']}")
        print(f"  URL   : {rec['github_url']}")
        print(f"  Text  : {rec['text'][:120]}...")
        print()

    print("--- Quick search test ---")
    results = index.search("Astro Islands architecture", num_results=3)
    for r in results:
        print(f"  {r['title']} — {r['github_url']}")
