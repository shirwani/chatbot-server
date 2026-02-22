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


def _trim_conversation_context(conversation_context: str | None, max_pairs: int = 6) -> str | None:
    """Return a shortened conversation transcript limited to the last N user/assistant turns.

    The widget serializes messages as lines like:
        "User: ..." or "Assistant: ...".

    This helper keeps the tail of that log so we don't blow up the model's
    context window with a very long chat history.
    """
    if not conversation_context:
        return None

    # Split into non-empty lines and keep only the last 2 * max_pairs lines,
    # which should roughly correspond to `max_pairs` user/assistant exchanges.
    lines = [ln for ln in conversation_context.splitlines() if ln.strip()]
    if not lines:
        return None

    keep = 2 * max_pairs
    trimmed_lines = lines[-keep:]
    return "\n".join(trimmed_lines)


@dbg_print
def do_execute_prompt(query: str, conversation_context: str | None = None) -> dict | None:
    """Determine the type of a user query and execute the appropriate workflow.

    Parameters
    ----------
    query:
        The latest user utterance.
    conversation_context:
        Optional serialized conversation history (e.g. "User: ...\nAssistant: ...").
        When provided, it will be passed through to LLM calls so they can answer
        with full conversational awareness instead of treating each query in
        isolation.

    Returns
    -------
    dict | None
        A dictionary of keyword arguments / model response payload used by the
        client, or ``None`` if the query is empty.
    """
    # Apply a server-side cap on conversation history so the LLM does not see
    # an unbounded transcript. We keep only the most recent ~6 user/assistant
    # exchanges regardless of what the client sends.
    conversation_context = _trim_conversation_context(conversation_context, max_pairs=6)

    # Normalise and optionally spell-correct the *latest* user query.  This is the
    # only text that should ever be used for:
    #   - FAQ / Chroma lookups
    #   - query-type classification (PRODUCT vs other)
    # The broader conversation_context is only for shaping the LLM's final
    # response, and must NOT be concatenated into the query used for search or
    # routing decisions.
    raw_query = (query or "").strip()
    if not raw_query:
        return None

    if _SPELL_CORRECTOR is not None:
        latest_query = _SPELL_CORRECTOR.fix_string(raw_query)
    else:
        latest_query = raw_query

    print(f"\n*************\nNormalized query: '{latest_query}'\n**************\n")

    # First try fast Chroma-based FAQ lookup using ONLY the latest user message.
    try:
        response = query_faq_chroma(latest_query)

        print(f"\n*************\nquery_faq_chroma() response: '{response}'\n**************\n")

        if response and response != "N/A":
            return response
    except Exception:
        # If FAQ lookup itself fails, fall back to the LLM using whatever
        # conversation context we have.
        base_prompt = (
            "User provided a question that broke the querying system. "
            "Instruct them to rephrase it. Answer it based on the context "
            "you already have so far. Query provided by the user: {query}"
        )
        prompt = base_prompt.format(query=latest_query)
        # If we have a prior conversation transcript, prepend it so the LLM
        # can see what "it" / "ones" etc. refer to.
        if conversation_context:
            combined = (
                "Conversation so far:\n" f"{conversation_context.strip()}\n\n"
                f"Latest query: {latest_query}"
            )
        else:
            combined = prompt
        kwargs = generate_params_dict(prompt=combined, temperature=1.0)
        response = generate_with_single_input(**kwargs)
        return response

    # Is it product-related? If so, execute the product query workflow.
    # Classification is based solely on the latest user query, not the
    # accumulated conversation context.
    label = get_query_type(latest_query)

    if label == "PRODUCT":
        try:
            response = query_products(latest_query)
            return response
        except Exception:
            base_prompt = (
                "User provided a question that broke the querying system. "
                "Instruct them to rephrase it. Answer it based on the context "
                "you already have so far. Query provided by the user: {query}"
            )
            prompt = base_prompt.format(query=latest_query)
            if conversation_context:
                combined = (
                    "Conversation so far:\n" f"{conversation_context.strip()}\n\n"
                    f"Latest query: {latest_query}"
                )
            else:
                combined = prompt
            kwargs = generate_params_dict(prompt=combined, temperature=1.0)
            response = generate_with_single_input(**kwargs)
            return response

    # Default response for queries that do not fit FAQ or Product categories
    prompt_path = os.path.join(
        get_client_system_prompts_path(), "fall_through_query_type.txt"
    )
    base_prompt = read_from_text_file(prompt_path)

    # Inject conversation context, if available, ahead of the templated prompt.
    # The underlying fall-through prompt typically expects something like
    #   "{query}" to be interpolated; we keep that behaviour while also
    # giving the LLM a compact transcript of the prior turns.
    filled_prompt = base_prompt.format(query=latest_query)
    if conversation_context:
        prompt = (
            "You are continuing a multi-turn conversation with a user. "
            "Use the dialogue so far plus the latest question to answer.\n\n"
            f"Conversation so far:\n{conversation_context.strip()}\n\n"
            f"Latest user message: {latest_query}\n\n"
            f"Instructions and additional guidance:\n{filled_prompt}"
        )
    else:
        prompt = filled_prompt

    kwargs = generate_params_dict(prompt=prompt, temperature=0.5)
    response = generate_with_single_input(**kwargs)
    return response


if __name__ == "__main__":
    test_queries = [
        # "Do you offer discounts?",
        # "Build me a look for a formal job interview for a sales position"
        "What's a good look for a woman to wear to a park on a Sunday afternoon?",
        # "Do you carry any summer dresses?",
        # "What is your return policy?",
        # "Make a wonderful look for a man attending a wedding party happening during night.",
        # "How do I track my order?",
        # "Do you have a waterproof jacket under $400?",
        # "What's a good look for a woman to wear to a park on an afternoon?",
        # "Can I cancel my order after placing it?"
        # "Are you hiring?",
        # "Are you open on weekends?",
    ]
    for query in test_queries:
        response = do_execute_prompt(query)
        print(f"Query: '{query}' \nResult: '{response}'\n\n")
