# %% [markdown]
# # Question Generation for Portfolio AI Agent Evaluation
# 
# This notebook generates **bilingual ground-truth Q&A pairs** (Spanish + English) from the portfolio documentation.
# 
# **Pipeline:**
# 1. Load all portfolio docs from GitHub (no chunking — full documents)
# 2. For each document, ask OpenAI to generate 3 Spanish + 3 English questions with expected answers
# 3. Save everything to `eval/ground_truth.csv`
# 
# **Requirements:** `OPENAI_API_KEY` must be set in your environment.

# %%
import sys
import os
import json
import pandas as pd

# ── Add parent directory to sys.path so ingest.py is importable ───────────────
try:
    NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    NOTEBOOK_DIR = os.path.abspath("")
PARENT_DIR = os.path.dirname(NOTEBOOK_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(PARENT_DIR, ".env"))

print(f"Parent dir added to sys.path: {PARENT_DIR}")
print(f"Python version: {sys.version}")

# %% [markdown]
# ## 1. Load Portfolio Documents
# 
# We load the documents **without chunking** (`chunk=False`) so each record represents one full document.
# This gives the LLM full context when generating questions.

# %%
from ingest import read_repo_data

# Load full documents — no sliding-window chunking
index, records = read_repo_data(chunk=False)

print(f"\nLoaded {len(records)} documents (unchunked)")

# %% [markdown]
# ## 2. Inspect a Sample Record
# 
# Let's look at the shape of one record before we start generating questions.

# %%
# Show a sample record to understand the data structure
sample = records[0]
print("Keys:", list(sample.keys()))
print(f"\ndoc_id      : {sample['doc_id']}")
print(f"title       : {sample['title']}")
print(f"github_url  : {sample['github_url']}")
print(f"text length : {len(sample['text'])} chars")
print(f"\nText preview:\n{sample['text'][:400]}...")

# %% [markdown]
# ## 3. Define the Question-Generation Prompt
# 
# The prompt instructs Groq to produce exactly **3 Spanish** and **3 English** questions per document,
# each paired with an expected answer grounded in the document content.
# 
# The model is asked to return **valid JSON** for easy parsing.

# %%
QUESTION_GEN_PROMPT_TEMPLATE = """\
You are an expert technical writer creating evaluation data for a bilingual AI assistant.

Below is a documentation excerpt from a portfolio website built with Astro, Tailwind CSS, React, and GitHub Pages.

Document title: {title}
Document ID: {doc_id}
GitHub URL: {github_url}

--- DOCUMENT CONTENT ---
{text}
--- END OF DOCUMENT ---

Generate exactly 6 question-answer pairs:
  - 3 questions in SPANISH
  - 3 questions in ENGLISH

Each answer must be:
  - Grounded exclusively in the document content above
  - Concise (1-3 sentences)
  - Technically accurate

Respond ONLY with a valid JSON array — no markdown fences, no extra text:
[
  {{"question": "...", "expected_answer": "...", "language": "es"}},
  {{"question": "...", "expected_answer": "...", "language": "en"}}
]
"""

print("Prompt template defined.")
print(f"Template length: {len(QUESTION_GEN_PROMPT_TEMPLATE)} chars")

# %% [markdown]
# ## 4. Generate Questions with OpenAI
# 
# We call the **OpenAI API directly** (not through the Pydantic AI agent) to keep this pipeline simple and deterministic.
# 
# - Model: `gpt-4o-mini`
# - Temperature: `0.3` for consistent, factual answers
# - We truncate documents longer than 6,000 chars to stay within context limits

# %%
from openai import OpenAI

# OpenAI client — reads OPENAI_API_KEY from .env
OPENAI_MODEL = "gpt-4o-mini"
MAX_DOC_CHARS = 6_000   # truncate very long documents to avoid token limits

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_qa_pairs(record: dict) -> list[dict]:
    """
    Call OpenAI to generate 6 Q&A pairs (3 ES + 3 EN) for a single document.

    Returns a list of dicts with keys:
        question, expected_answer, language
    Returns [] on any error so the pipeline continues.
    """
    text_snippet = record["text"][:MAX_DOC_CHARS]

    prompt = QUESTION_GEN_PROMPT_TEMPLATE.format(
        title=record["title"],
        doc_id=record["doc_id"],
        github_url=record["github_url"],
        text=text_snippet,
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        pairs = json.loads(raw)
        return pairs

    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse error for {record['doc_id']}: {e}")
        return []
    except Exception as e:
        print(f"  [WARN] API error for {record['doc_id']}: {e}")
        return []


print(f"generate_qa_pairs() function defined  (model={OPENAI_MODEL})")

# %% [markdown]
# ## 5. Run Generation Loop
# 
# Iterate over every document record and collect Q&A pairs.
# A `tqdm` progress bar shows how far along we are.

# %%
from tqdm.auto import tqdm

all_pairs = []
question_counter = 0

for record in tqdm(records, desc="Generating Q&A pairs"):
    pairs = generate_qa_pairs(record)

    for pair in pairs:
        all_pairs.append(
            {
                "question_id": f"q{question_counter:04d}",
                "question": pair.get("question", ""),
                "expected_answer": pair.get("expected_answer", ""),
                "language": pair.get("language", "en"),
                "doc_id": record["doc_id"],
                "github_url": record["github_url"],
            }
        )
        question_counter += 1

print(f"\nGeneration complete: {len(all_pairs)} Q&A pairs collected")

# %% [markdown]
# ## 6. Save to CSV
# 
# Save the ground truth pairs to `eval/ground_truth.csv` next to this notebook.

# %%
# Build DataFrame with the required columns
df = pd.DataFrame(
    all_pairs,
    columns=["question_id", "question", "expected_answer", "language", "doc_id", "github_url"],
)

# Output path — same directory as this notebook
OUTPUT_PATH = os.path.join(NOTEBOOK_DIR, "ground_truth.csv")
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"Saved {len(df)} rows to: {OUTPUT_PATH}")
print("\nFirst 3 rows:")
df.head(3)

# %% [markdown]
# ## 7. Dataset Statistics
# 
# A quick summary of what was generated.

# %%
total = len(df)
es_count = (df["language"] == "es").sum()
en_count = (df["language"] == "en").sum()
docs_covered = df["doc_id"].nunique()

print("=" * 45)
print(" Ground Truth Dataset Statistics")
print("=" * 45)
print(f"  Total Q&A pairs     : {total}")
print(f"  Spanish (es)        : {es_count}")
print(f"  English (en)        : {en_count}")
print(f"  Documents covered   : {docs_covered} / {len(records)}")
print(f"  Avg pairs per doc   : {total / max(docs_covered, 1):.1f}")
print("=" * 45)

print("\nPairs per document:")
print(df.groupby("doc_id")["question_id"].count().rename("num_questions").to_string())

# %% [markdown]
# ## Done!
# 
# The file `eval/ground_truth.csv` is now ready.
# 
# **Next step:** Open `evaluations.ipynb` to run the full retrieval + LLM-as-Judge evaluation pipeline.


