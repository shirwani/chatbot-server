from utils import dbg_print, read_from_text_file
from llm_utils import (
    get_client_system_prompts_path,
    generate_params_dict,
    generate_with_single_input,
)
from query_type import get_query_type
from query_products import query_products
from query_faq_chroma import query_faq_chroma
from spell_corrector import SpellCorrector
import os

# Build a shared spell corrector backed by a general English word list if available.
# If the optional `wordfreq` dependency is missing, fail soft and continue without
# auto-correction.
try:
    _SPELL_CORRECTOR = SpellCorrector.from_english_dictionary(
        min_length=2,
        max_words=50000,
    )
except Exception:
    _SPELL_CORRECTOR = None


@dbg_print
def do_execute_prompt(query: str) -> dict | None:
    """
    Determines the type of a given query (FAQ or Product) and executes the appropriate workflow.

    Parameters:
    - query (str): The user's query string.

    Returns:
    - dict: A dictionary of keyword arguments to be used for further processing.
      If the query is neither FAQ nor Product-related, returns a default response dictionary
      instructing the assistant to answer based on existing context.
    """
    # Normalise and optionally spell-correct the user's query once, so all
    # downstream components (FAQ, product search, LLM fallbacks) see the
    # same cleaned version.
    raw_query = (query or "").strip()
    if not raw_query:
        return None

    if _SPELL_CORRECTOR is not None:
        query = _SPELL_CORRECTOR.fix_string(raw_query)
    else:
        query = raw_query

    # First try fast Chroma-based FAQ lookup
    try:
        response = query_faq_chroma(query)
        if response and response != '':
            return response
    except:
        prompt = f"User provided a question that broke the querying system. Instruct them to rephrase it." \
                 f"Answer it based on the context you already have so far. Query provided by the user: {query}"
        kwargs = generate_params_dict(prompt=prompt, temperature=1.0)
        response = generate_with_single_input(**kwargs)
        return response

    # Is it product-related? If so, execute the product query workflow
    label = get_query_type(query)

    if label == 'PRODUCT':
        try:
            response = query_products(query)
            return response
        except:
            prompt = f"User provided a question that broke the querying system. Instruct them to rephrase it." \
                     f"Answer it based on the context you already have so far. Query provided by the user: {query}"
            kwargs = generate_params_dict(prompt=prompt, temperature=1.0)
            response = generate_with_single_input(**kwargs)
            return response

    # Default response for queries that do not fit FAQ or Product categories
    prompt_path = os.path.join(get_client_system_prompts_path(), "fall_through_query_type.txt")
    prompt = read_from_text_file(prompt_path)
    prompt = prompt.format(query=query)
    kwargs = generate_params_dict(prompt=prompt, temperature=0.5)
    response = generate_with_single_input(**kwargs)
    return response


if __name__ == '__main__':
    test_queries = [
        #"Do you offer discounts?",
        "How can I contact customer support?",
        #"Do you carry any summer dresses?",
        #"What is your return policy?",
        #"Make a wonderful look for a man attending a wedding party happening during night.",
        #"How do I track my order?",
        #"Do you have a waterproof jacket under $400?",
        #"What's a good look for a woman to wear to a park on a Sunday afternoon?",
        #"Can I cancel my order after placing it?"
        #"Are you hiring?",
        #"Are you open on weekends?",
    ]
    for query in test_queries:
        response = do_execute_prompt(query)
        print(f"Query: '{query}' \nResult: '{response}'\n\n")
