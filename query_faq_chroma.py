from __future__ import annotations
import os
from typing import List, Dict, Any
from llm_utils import (
    get_faq_collection,
    get_faq_embedder,
    get_client_faq_synth_prompt_file,
    generate_with_single_input,
)
from utils import dbg_print, read_from_text_file

def _get_collection():
    return get_faq_collection()


def _get_faq_embedder():
    return get_faq_embedder()


def _normalize_keywords(kw_list: List[str]) -> List[str]:
    return [kw.strip().lower() for kw in kw_list if isinstance(kw, str) and kw.strip()]


def _keyword_match(user_query: str, metadatas: List[Dict[str, Any]]) -> List[int]:
    """Return indices of FAQ entries whose keywords list hits the user query.

    Uses simple case-insensitive substring matching between the query and
    each keyword phrase for robustness (e.g. "weekend hours" hits
    keywords like "open", "weekend", "hours").
    """
    if not user_query:
        return []

    query_norm = " ".join(_normalize_keywords([user_query]))
    if not query_norm:
        return []

    hit_indices: List[int] = []
    for idx, meta in enumerate(metadatas):
        kws = meta.get("keywords", []) or []
        if not kws:
            continue
        for kw in _normalize_keywords(kws):
            if kw and kw in query_norm:
                hit_indices.append(idx)
                break

    return hit_indices


def _synthesize_answer(query: str, matches: List[Dict[str, Any]]) -> str:
    """Use the LLM to synthesize a final answer from FAQ matches and the query."""
    # Build a simple context block of Q/A pairs
    faq_context_lines: List[str] = []
    for i, m in enumerate(matches, start=1):
        q = (m.get("question") or "").strip()
        a = (m.get("answer") or "").strip()
        if not (q and a):
            continue
        faq_context_lines.append(f"FAQ {i} - Q: {q}\nFAQ {i} - A: {a}")
    faq_context = "\n\n".join(faq_context_lines)

    # Try loading a client-specific synthesis prompt template if present
    synth_prompt_path = get_client_faq_synth_prompt_file()
    if os.path.isfile(synth_prompt_path):
        prompt_template = read_from_text_file(synth_prompt_path)
        prompt = prompt_template.format(faq_context=faq_context, query=query)
    else:
        # Fallback generic prompt
        prompt = (
            "You are a helpful assistant answering user questions based on an FAQ.\n\n"
            "Here are relevant FAQ entries (question and answer):\n\n"
            f"{faq_context}\n\n"
            "User question: {query}\n\n"
            "Provide a concise, direct answer to the user, based only on the FAQ information."
            "If the answer is not in the FAQ, return '' (empty string)."
        ).format(query=query)

    # Use relatively low temperature for factual FAQ-style responses
    answer = generate_with_single_input(
        prompt=prompt,
        temperature=0.2,
        top_p=0.8,
        max_tokens=1024,
    )
    return answer


@dbg_print
def query_faq_chroma(query: str, top_k: int = 3):
    """Query the FAQ ChromaDB collection using keyword + semantic search and synthesize an answer.

    Returns:
      - dict with keys {"matches", "answer"} where:
          * matches: list of {question, answer, score}
          * answer: synthesized answer string
      - None if no FAQ match is found.
    """
    # Auto-correct obvious spelling mistakes in the user's query while
    # preserving punctuation and casing as much as possible. This helps
    # both keyword matching and semantic retrieval.
    raw_query = (query or "").strip()
    if not raw_query:
        return None

    collection = _get_collection()

    # Fetch all docs & metadatas once (FAQ set is small)
    res = collection.get(include=["documents", "metadatas"], limit=100000)
    ids: List[str] = [str(i) for i in res.get("ids", []) or []]
    docs: List[str] = res.get("documents", []) or []
    metas: List[Dict[str, Any]] = res.get("metadatas", []) or []

    if not docs or not metas:
        return None

    # Keyword pre-filter
    hit_indices = _keyword_match(query, metas)
    if not hit_indices:
        return None

    # Build subset for semantic search
    subset_docs = [docs[i] for i in hit_indices]
    subset_metas = [metas[i] for i in hit_indices]

    embedder = _get_faq_embedder()
    query_emb = embedder.encode([query], convert_to_numpy=True)
    subset_embs = embedder.encode(subset_docs, convert_to_numpy=True)

    # Manual cosine similarity for full control and speed (small N)
    import numpy as np

    q_vec = query_emb[0]
    # Normalize
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    s_norm = subset_embs / (np.linalg.norm(subset_embs, axis=1, keepdims=True) + 1e-12)

    sims = (s_norm @ q_norm).astype(float)
    # Top-k indices sorted by similarity desc
    k = min(top_k, len(sims))
    top_idx = np.argsort(-sims)[:k]

    matches: List[Dict[str, Any]] = []
    for rank, i in enumerate(top_idx):
        meta = subset_metas[int(i)]
        score = float(sims[int(i)])
        matches.append(
            {
                "question": meta.get("question", ""),
                "answer": meta.get("answer", ""),
                "score": score,
            }
        )

    if not matches:
        return None

    synthesized = _synthesize_answer(query, matches)

    print(f"    - query_faq_chroma() -> matches: {matches}, synthesized answer: {synthesized}")
    return synthesized


if __name__ == "__main__":
    # Small smoke test
    test_queries = [
        #"Are you hiring?",
        "What is your customer service contact?",
        "Do you sell gift cards?",
        "What are your weakand hours?",
        "Do you give free estimates?",
        #"Something unrelated to faq",
    ]
    for q in test_queries:
        print("Query:", q)
        result = query_faq_chroma(q)
        print("Result:", result)
        print("-" * 80)
