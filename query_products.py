from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from llm_utils import (
    get_products_embedder,
    get_products_collection,
    get_client_filter_on_list_file,
    get_client_metadata_fields_list,
    get_client_system_prompts_path,
    get_params_for_task,
    generate_params_dict,
    generate_with_single_input,
    get_cross_encoder_reranker,
)
from utils import dbg_print, read_file_as_list, read_file_as_tuple, read_from_text_file
import numpy as np
from metadata_filters import generate_serializeable_metadata_filters_from_query
from technical_or_creative import technical_or_creative
from typing import List
import os


# ---------------------------------------------------------------------------
# Cached file readers — avoid re-reading the same config files from disk on
# every single query.  The cache is keyed on the file path so it
# automatically invalidates if the client changes (different path).
# ---------------------------------------------------------------------------

@dbg_print
@lru_cache(maxsize=16)
def _cached_read_file_as_list(path: str) -> list[str]:
    return read_file_as_list(path)


@dbg_print
@lru_cache(maxsize=16)
def _cached_read_file_as_tuple(path: str) -> tuple[str, ...]:
    return read_file_as_tuple(path)


@dbg_print
@lru_cache(maxsize=16)
def _cached_read_from_text_file(path: str) -> str:
    return read_from_text_file(path)

@dbg_print
def _build_chroma_where_from_filters(filters: list[dict] | None) -> dict:
    """Convert our internal filter list into a ChromaDB `where` dict.

    Each filter is expected to be of the form:
      {"field": str, "operator": one of ">", "<", "in", "value": Any}

    We map these to Chroma comparison operators and wrap them under a
    single top-level `$and` operator, as required by Chroma's filter
    validation (`where` must have exactly one operator key).
    """
    if not filters:
        # Return an empty dict to signal "no filters" to the caller;
        # callers must treat this as "omit where entirely" rather than
        # passing `{}` into Chroma, which will raise a validation error.
        return {}

    clauses: list[dict] = []
    for f in filters:
        field = f.get("field")
        op = f.get("operator")
        value = f.get("value")
        if not field or op is None:
            continue

        if op == ">":
            clause = {field: {"$gt": value}}
        elif op == "<":
            clause = {field: {"$lt": value}}
        elif op == "in":
            if not isinstance(value, (list, tuple, set)):
                value = [value]
            clause = {field: {"$in": list(value)}}
        else:
            continue

        clauses.append(clause)

    if not clauses:
        return {}

    # Chroma expects a single top-level operator like {$and: [ ... ]}
    if len(clauses) == 1:
        # A single clause can be returned directly without $and
        return clauses[0]

    return {"$and": clauses}


@dbg_print
def _rerank_with_cross_encoder(query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
    """Re-score candidates using a Cross-Encoder and return the top-k items.

    The Cross-Encoder compares the raw query against each candidate's document
    text, producing a relevance score that is far more accurate than the
    initial bi-encoder (embedding) similarity.

    Args:
        query: The original user query.
        candidates: List of product dicts as returned by ``_run_chroma_query``.
        top_k: Number of top items to keep after reranking.

    Returns:
        The *top_k* most relevant candidates, sorted by Cross-Encoder score
        (highest first).
    """
    if not candidates:
        return []

    cross_encoder = get_cross_encoder_reranker()

    # Build (query, document) pairs for the cross-encoder
    pairs = [(query, c.get("document") or "") for c in candidates]
    scores = cross_encoder.predict(pairs)

    # Attach scores and sort descending
    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Stage 1 – Retrieve a broad set of candidates (Top-K = 50)
# Stage 2 – Rerank with Cross-Encoder and keep Top-5
# ---------------------------------------------------------------------------

INITIAL_CANDIDATES = 50   # broad retrieval from ChromaDB
RERANK_TOP_K = 5           # items passed to the LLM after reranking


@dbg_print
def get_relevant_products_from_query(query: str, filters: list[dict] | None = None):
    """Retrieve products that are most relevant to a given query using ChromaDB.

    This function implements a two-stage **Filter-then-Rerank** pipeline:

      1. **Metadata Pre-Filtering** – Infers hard filters from the query
         (e.g. category, colour, price range) and applies them as a Chroma
         ``where`` clause to shrink the search space *before* any vector math.
      2. **Hybrid Retrieval (Top-K=50)** – Runs a semantic similarity search
         over the (possibly filtered) collection to fetch a broad set of
         initial candidates.
      3. **Cross-Encoder Reranking** – A lightweight Cross-Encoder
         (``cross-encoder/ms-marco-MiniLM-L-6-v2``) compares the raw query
         against each candidate's document text and re-scores them.
      4. **Context Compression** – Only the top 5 highest-scoring items are
         returned for inclusion in the LLM prompt, drastically reducing
         token usage.

    Args:
        query: The user's natural-language search query.
        filters: Pre-computed serializable metadata filters. If ``None``,
                 the function will generate them internally (LLM call).

    Returns:
      A list of the top-5 most relevant product dicts from ChromaDB.
    """
    # Generate structured filters if not pre-computed by the caller
    if filters is None:
        filters = generate_serializeable_metadata_filters_from_query(query)

    # Build base where clause
    base_where = _build_chroma_where_from_filters(filters)

    # Prepare query embedding — convert to plain list to avoid numpy
    # serialization overhead inside Chroma's transport layer.
    query_embedding = get_products_embedder().encode([query], convert_to_numpy=True).tolist()

    def _run_chroma_query(where_clause: dict | None, limit: int = INITIAL_CANDIDATES) -> List[dict]:
        """Helper to run a Chroma query and normalize results into a list of product dicts."""
        # Chroma expects `where` to be either omitted or contain a single
        # top-level operator. An empty dict `{}` is considered invalid and
        # will raise `Expected where to have exactly one operator`.
        if not where_clause:
            where_clause = None

        # Explicitly request only the fields we need — documents (for
        # reranking), metadatas (for context), and distances (for scoring).
        # Omitting "embeddings" avoids transferring the largest payload.
        results = get_products_collection().query(
            query_embeddings=query_embedding,
            n_results=limit,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        #print(results)

        # Chroma returns dict-of-lists; we normalize to list of dicts for convenience.
        ids_list = results.get("ids", [[]])[0] or []
        docs_list = results.get("documents", [[]])[0] or []
        metas_list = results.get("metadatas", [[]])[0] or []
        scores_raw = np.array(results.get("distances", [[]])[0] or [], dtype=float)
        scores = 1 / (1 + scores_raw) if len(scores_raw) else np.zeros_like(scores_raw)

        normalized: List[dict] = []
        for i, pid in enumerate(ids_list):
            meta = metas_list[i] if i < len(metas_list) else {}
            doc = docs_list[i] if i < len(docs_list) else None
            score = float(scores[i]) if i < len(scores) else None
            normalized.append({
                "id": pid,
                "document": doc,
                "metadata": meta,
                "score": score,
            })
        return normalized

    # If no filters, just do a pure semantic search
    if not base_where:
        candidates = _run_chroma_query(where_clause=None)
        return _rerank_with_cross_encoder(query, candidates, top_k=RERANK_TOP_K)

    # First attempt with full filter set
    res = _run_chroma_query(where_clause=base_where)

    # If the result set is already large enough, go straight to reranking.
    if len(res) >= RERANK_TOP_K:
        return _rerank_with_cross_encoder(query, res, top_k=RERANK_TOP_K)

    # Gradually relax filters using importance order to broaden the search.
    importance_order = _cached_read_file_as_list(get_client_filter_on_list_file())

    def _filters_without_low_importance(current_filters: list[dict], drop_after_index: int) -> list[dict]:
        if drop_after_index + 1 >= len(importance_order):
            return current_filters
        to_drop = set(importance_order[drop_after_index + 1:])
        return [f for f in current_filters if f.get('field') not in to_drop]

    if filters:
        for i in range(len(importance_order)):
            reduced_filters = _filters_without_low_importance(filters, i)
            where_reduced = _build_chroma_where_from_filters(reduced_filters)
            # Skip if the reduced where clause is identical to the one we already tried
            if where_reduced == base_where:
                continue
            res = _run_chroma_query(where_clause=where_reduced)
            if len(res) >= RERANK_TOP_K:
                break

    # Final fallback: semantic search with no filters if still too few
    if len(res) < RERANK_TOP_K:
        res = _run_chroma_query(where_clause=None)

    # Stage 2: Cross-Encoder reranking – keep only the top RERANK_TOP_K items
    return _rerank_with_cross_encoder(query, res, top_k=RERANK_TOP_K)


@dbg_print
def generate_items_context(relevant_products: list) -> str:
    """Build a **compact** context string from the top reranked products.

    Only essential metadata fields are included so the LLM prompt stays
    short and focused.  Each item is rendered on its own line for clarity.

    The fields emitted are read from the client's ``metadata_fields_list.txt``
    configuration file, which lists every field that should appear in the
    context sent to the LLM.

    Parameters:
        relevant_products: A list of product dicts, each containing at least
            a ``metadata`` sub-dict with product attributes.

    Returns:
        A multi-line string where each line is a concise description of one
        product, suitable for inclusion in an LLM prompt.
    """
    metadata_fields = _cached_read_file_as_tuple(get_client_metadata_fields_list())
    lines: list[str] = []

    for item in relevant_products:
        meta = item.get("metadata", {})
        parts: list[str] = []
        for f in metadata_fields:
            val = meta.get(f, "N/A")
            if val and str(val).strip():
                parts.append(f"{f}: {val}")
        lines.append(". ".join(parts))

    return "\n".join(lines)


@dbg_print
def query_products(query: str) -> dict:
    """
    Execute a product query process to generate a response based on the nature of the query.

    This function analyzes the type of query — whether it is technical or creative — and retrieves
    relevant product information accordingly. It constructs a prompt that includes product details
    and the original query, and then generates parameters for querying an LLM.
    Finally, it generates a response based on the prompt and returns the content of the response.

    Parameters:
    query (str): The input query string that needs to be analyzed and answered using product data.

    Returns:
    dict: A dictionary of keyword arguments (`kwargs`) containing the prompt and additional settings
          for creating a response, suitable for input to an LLM or other processing system.

    Outputs:
    dict: A dictionary with the parameters to call an LLM
    """

    # --- Parallel LLM pre-processing -------------------------------------------
    # `technical_or_creative` and `generate_serializeable_metadata_filters_from_query`
    # are independent LLM calls that together dominate latency.  Running them
    # concurrently roughly halves the wall-clock time of this phase.
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_label = executor.submit(technical_or_creative, query)
        future_filters = executor.submit(
            generate_serializeable_metadata_filters_from_query, query
        )

        query_label = future_label.result()
        filters = future_filters.result()

    # Obtain necessary parameters based on the query type
    parameters_dict = get_params_for_task(query_label)

    # Retrieve products that are relevant to the query (pass pre-computed filters)
    relevant_products = get_relevant_products_from_query(query, filters=filters)

    # Create a context string from the relevant products
    context = generate_items_context(relevant_products)
    prompt_path = os.path.join(get_client_system_prompts_path(), "query_products.txt")
    prompt = _cached_read_from_text_file(prompt_path)
    prompt = prompt.format(context=context, query=query)
    kwargs = generate_params_dict(prompt, role='assistant', **parameters_dict)
    result = generate_with_single_input(**kwargs)

    return result


if __name__ == '__main__':
    # Simple manual test for filter generation; Chroma query requires runtime wiring.
    #query = "Give me three T-shirts to use in sunny days"
    #filters = generate_filters_from_query(query)
    #results = get_relevant_products_from_query(query)
    #dump_to_json_file("./results.json", results, 2)

    #t = generate_items_context(results)
    #print("Context for items:\n", t[:1000]) # Print the first 1000 characters of the context for brevity
    # query = "Show me some men's suits"
    #query = "Do you have any women's dresses?"
    #query = "Build me a look for a job interview for a sales position for a man"
    #query = "What's a good look for a woman to wear to a park on a Sunday afternoon?"
    query = "Do you have any grey sarees in your catalog?"

    result = query_products(query)
    print(result)



