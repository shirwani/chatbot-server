from llm_utils import (
    get_client_system_prompts_path,
    generate_params_dict,
    generate_with_single_input,
)
from utils import dbg_print, read_from_text_file
import os


@dbg_print
def technical_or_creative(query: str) -> str:
    """
    Determines whether a given instruction prompt is related to a frequently asked question (FAQ) or a product inquiry.

    Parameters:
    - query (str): The instruction or query to be labeled as either FAQ or product-related.

    Returns:
    - str: The label 'creative' if the prompt is classified as a creative query, 'technical' if it relates to technical information, or
      None if the label is inconclusive.
    """

    prompt_path = os.path.join(get_client_system_prompts_path(), "technical_or_creative.txt")
    prompt = read_from_text_file(prompt_path)
    prompt = prompt.format(query=query)
    kwargs = generate_params_dict(prompt=prompt, temperature=0, max_tokens=2048)
    response = generate_with_single_input(**kwargs)
    print(f"    - technical_or_creative() -> response: {response}")
    return response


if __name__ == "__main__":
    """
    Main method for testing check_if_faq_or_product function.
    """
    test_queries = [
        "Give me two sneakers with vibrant colors.",
        #"What are the most expensive clothes you have in your catalogue?",
        #"I have a green dress and I like a suggestion on an accessory to match with it.",
        #"Give me three trousers with vibrant colors you have in your catalogue.",
        #"Create a look for a woman walking in a park on a sunny day. It must be fresh due to hot weather.",
        #"Make a wonderful look for a man attending a wedding party happening during night.",
    ]
    for query in test_queries:
        result = technical_or_creative(query)
        print(f"Query: '{query}' => Label: {result}")
