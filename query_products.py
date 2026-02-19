from llm_utils import (
    get_products_embedder,
    get_products_collection,
    get_client_filter_on_list_file,
    get_client_metadata_fields_list,
    get_client_system_prompts_path,
    get_params_for_task,
    generate_params_dict,
    generate_with_single_input,
)
from utils import dbg_print, read_file_as_list, read_file_as_tuple, read_from_text_file
import numpy as np
from metadata_filters import generate_serializeable_metadata_filters_from_query
from technical_or_creative import technical_or_creative
from typing import List
import os

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
def get_relevant_products_from_query(query: str):
    """Retrieve products that are most relevant to a given query using ChromaDB.

    This function:
      1. Infers metadata filters from the natural-language query.
      2. Converts them to a Chroma-compatible `where` clause.
      3. Runs a vector similarity search over the products collection.
      4. If the result set is small, gradually relaxes the filters following
         an importance order to broaden the search.

    Returns:
      A list of product result dicts from ChromaDB.
    """
    # Generate structured filters based on query text
    filters = generate_serializeable_metadata_filters_from_query(query)

    # Build base where clause
    base_where = _build_chroma_where_from_filters(filters)

    # Prepare query embedding using the embedder getter
    query_embedding = get_products_embedder().encode([query], convert_to_numpy=True)

    def _run_chroma_query(where_clause: dict | None, limit: int = 20) -> List[dict]:
        """Helper to run a Chroma query and normalize results into a list of product dicts."""
        # Chroma expects `where` to be either omitted or contain a single
        # top-level operator. An empty dict `{}` is considered invalid and
        # will raise `Expected where to have exactly one operator`.
        if not where_clause:
            where_clause = None

        results = get_products_collection().query(
            query_embeddings=query_embedding,
            n_results=limit,
            where=where_clause,
        )
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
        return _run_chroma_query(where_clause=None, limit=20)

    # First attempt with full filter set
    res = _run_chroma_query(where_clause=base_where, limit=20)

    # If the result set is small, gradually relax filters using importance order
    # importance_order = ['baseColor', 'masterCategory', 'usage', 'masterCategory', 'season', 'gender']
    importance_order = read_file_as_list(get_client_filter_on_list_file())

    def _filters_without_low_importance(current_filters: list[dict], drop_after_index: int) -> list[dict]:
        if drop_after_index + 1 >= len(importance_order):
            return current_filters
        to_drop = set(importance_order[drop_after_index + 1:])
        return [f for f in current_filters if f.get('field') not in to_drop]

    if len(res) < 10 and filters:
        for i in range(len(importance_order)):
            reduced_filters = _filters_without_low_importance(filters, i)
            where_reduced = _build_chroma_where_from_filters(reduced_filters)
            res = _run_chroma_query(where_clause=where_reduced, limit=20)
            if len(res) >= 5:
                return res

        # Final fallback: semantic search with no filters if still too few
        if len(res) < 5:
            res = _run_chroma_query(where_clause=None, limit=20)

    return res


@dbg_print
def generate_items_context(relevant_products: list) -> str:
    """
    Compile detailed product information from a list of result objects into a formatted string.

    This function takes a list of results, each containing various product attributes, and constructs
    a human-readable summary for each product. Each product's details, including ID, name, category,
    usage, gender, type, and other characteristics, are concatenated into a string that describes
    all products in the list.

    Parameters:
    results (list): A list of result objects, each having a `properties` attribute that is a dictionary
                    containing product attributes such as 'product_id', 'productDisplayName',
                    'masterCategory', 'usage', 'gender', 'articleType', 'subCategory',
                    'baseColour', 'season', and 'year'.

    Returns:
    str: A multi-line string where each line contains the formatted details of a single product.
         Each product detail includes the product ID, name, category, usage, gender, type, color,
         season, and year.
    """
    metadata_fields = read_file_as_tuple(get_client_metadata_fields_list())
    t = ""  # Initialize an empty string to accumulate product information

    for item in relevant_products:  # Iterate through each item in the results list
        item = item['metadata']

        # Append formatted product details to the output string
        for f in metadata_fields:
            t += f"{f}: {item.get(f, 'N/A')}. "

    return t  # Return the complete formatted string with product details


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

    # Determine if the query is technical or creative in nature
    query_label = technical_or_creative(query)

    # Obtain necessary parameters based on the query type
    parameters_dict = get_params_for_task(query_label)

    # Retrieve products that are relevant to the query
    relevant_products = get_relevant_products_from_query(query)

    # Create a context string from the relevant products
    context = generate_items_context(relevant_products)

    prompt_path = os.path.join(get_client_system_prompts_path(), "query_products.txt")
    prompt = read_from_text_file(prompt_path)
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


    query = "Make a wonderful look for a man attending a wedding party happening during night."
    #query = "What's a good look for a woman to wear to a park on a Sunday afternoon?"

    result = query_products(query)
    print(result)



