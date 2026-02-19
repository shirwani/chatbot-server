"""Create a ChromaDB collection from the client's FAQ text file.

This script reads the FAQ file for the configured CLIENT_SITE and stores
its contents in a persistent ChromaDB collection located under that
client's `chroma_db` directory.

The collection is called "faq".

Usage (from project root):

    python -m code.create_chromadb_faq_collection

or

    python code/create_chromadb_faq_collection.py

The script is idempotent with respect to document IDs: it will skip
FAQ entries whose IDs are already present in the collection, so you can
safely re-run it after updating the FAQ file.
"""

from __future__ import annotations
import os
import re
import sys
from typing import List, Dict
from utils import dbg_print
from llm_utils import (
    get_client_faq_file,
    get_client_name,
    set_client_name,
    set_faq_collection,
    get_faq_collection,
    get_faq_embedder,
    get_faq_collection_name,
    get_client_chroma_db_path,
    get_chroma_db_client,
)
from utils import read_from_text_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_faq_file_exists() -> None:
    """Validate that the FAQ file exists."""
    if not os.path.isfile(get_client_faq_file()):
        raise FileNotFoundError(
            f"FAQ file not found at {get_client_faq_file()}. "
            "Expected path: client_sites/CLIENT_SITE/faq/faq.txt"
        )


def _get_existing_ids(collection) -> set:
    """Fetch all existing IDs in the collection to support idempotent indexing."""
    existing_ids: set = set()
    res = collection.get(include=["metadatas"], limit=100000)
    for _id in res.get("ids", []) or []:
        existing_ids.add(str(_id))
    return existing_ids


def _parse_faq_entries(text: str) -> List[Dict[str, str]]:
    """Parse the FAQ text into structured entries.

    Expected format for each entry:

        Q: <question>\n
        Keywords: kw1, kw2, ...\n
        A: <answer>\n
    """
    entries: List[Dict[str, str]] = []

    # Split on blank lines between blocks
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue

        q_line = next((ln for ln in lines if ln.lower().startswith("q:")), None)
        k_line = next((ln for ln in lines if ln.lower().startswith("keywords:")), None)
        a_line = next((ln for ln in lines if ln.lower().startswith("a:")), None)

        if not (q_line and k_line and a_line):
            continue

        question = q_line.split(":", 1)[1].strip()
        keywords_raw = k_line.split(":", 1)[1].strip()
        answer = a_line.split(":", 1)[1].strip()

        keywords = [kw.strip() for kw in keywords_raw.split(",") if kw.strip()]

        entry = {
            "question": question,
            "keywords": keywords,
            "answer": answer,
        }
        entries.append(entry)

    return entries


def _build_text_for_entry(entry: Dict[str, str]) -> str:
    """Build a text representation for embedding using both question and answer.

    This helps the semantic search match queries against either the
    question wording or key phrases contained only in the answer.
    """
    question = (entry.get("question") or "").strip()
    answer = (entry.get("answer") or "").strip()
    if question and answer:
        return f"Q: {question} \nA: {answer}"
    return question or answer


# ---------------------------------------------------------------------------
# Main indexing logic
# ---------------------------------------------------------------------------

@dbg_print
def index_faq_to_chroma() -> None:
    """Index the client's FAQ entries into the Chroma `faq` collection.

    - One Chroma document per Q/A entry
    - Document text = question (for fast semantic search)
    - Metadata = question, answer, keywords, client_site
    """

    _ensure_faq_file_exists()
    set_faq_collection()
    existing_ids = _get_existing_ids(get_faq_collection())

    faq_text: str = read_from_text_file(get_client_faq_file())
    if not faq_text.strip():
        print("FAQ file is empty; nothing to index.")
        return

    entries = _parse_faq_entries(faq_text)
    if not entries:
        print("No FAQ entries parsed; check FAQ format.")
        return

    docs: List[str] = []
    metadatas: List[Dict[str, object]] = []
    ids: List[str] = []

    for idx, entry in enumerate(entries, start=1):
        doc_id = f"faq-{idx}"
        if doc_id in existing_ids:
            continue

        text = _build_text_for_entry(entry)
        docs.append(text)
        metadatas.append(
            {
                "question": entry.get("question", ""),
                "answer": entry.get("answer", ""),
                "keywords": entry.get("keywords", []),
                "client_site": get_client_name(),
                "source": "faq.txt",
            }
        )
        ids.append(doc_id)

    if not ids:
        print("All FAQ entries already indexed; nothing new to add.")
        return

    embeddings = get_faq_embedder().encode(docs, convert_to_numpy=True)
    get_faq_collection().add(
        documents=docs,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    print(
        f"Indexed {len(ids)} FAQ entries for client '{get_client_name()}' into Chroma collection "
        f"'{get_faq_collection_name()}' at '{get_client_chroma_db_path()}'."
    )


@dbg_print
def drop_faq_collection() -> None:
    """Delete the `faq` collection for the configured client site if present."""
    try:
        get_chroma_db_client().delete_collection(get_faq_collection_name())
        print(f"Deleted '{get_faq_collection_name()}' collection.")
    except Exception as e:
        print(
            f"Could not delete '{get_faq_collection_name()}' collection (may not exist yet): {e}"
        )


if __name__ == "__main__":

    args = sys.argv[1:]

    # kwargs: anything containing "="
    kwargs = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            kwargs[key] = value

    client_name = kwargs.get("client", None)

    if not client_name:
        print("No client specified. Use client=CLIENT_NAME to specify the client.")
        sys.exit(1)

    set_client_name(client_name)
    drop_faq_collection()
    index_faq_to_chroma()
