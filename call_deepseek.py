from llm_utils import *
from openai import OpenAI  # Requires `openai` Python package >= 1.0.0

@dbg_print
def ask_deepseek_r1(payload: dict) -> str:
    """Call the DeepSeek R1 (OpenAI-compatible API) chat completion endpoint.

    Args:
        payload: Dict with at least a "messages" list in OpenAI chat format.
            Optional keys: temperature, top_p, max_tokens.

    Env vars supported:
        - DEEPSEEK_API_KEY (preferred) or DEEPSEEK_API_TOKEN
        - DEEPSEEK_BASE_URL (optional, default: https://api.deepseek.com)
        - DEEPSEEK_MODEL (optional, default: deepseek-reasoner)
    """

    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_TOKEN")
    if not deepseek_key:
        raise RuntimeError(
            "Missing DeepSeek API key. Set DEEPSEEK_API_KEY (or DEEPSEEK_API_TOKEN) in your environment."
        )

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    print(f"\nAsking ({model}) via {base_url}...")

    client = OpenAI(api_key=deepseek_key, base_url=base_url)

    # Safely pull generation params from payload, allowing for None
    raw_temperature = payload.get("temperature", None)
    raw_top_p = payload.get("top_p", None)
    raw_max_tokens = payload.get("max_tokens", None)

    temperature = float(raw_temperature) if raw_temperature is not None else 0.7
    top_p = float(raw_top_p) if raw_top_p is not None else 1.0
    max_tokens = int(raw_max_tokens) if raw_max_tokens is not None else 2048

    response = client.chat.completions.create(
        model=model,
        messages=payload.get("messages", []),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    # For DeepSeek R1, reasoning may be in `response.choices[0].message.reasoning_content`
    # We return the final user-visible content here.
    message = response.choices[0].message


    # The OpenAI client exposes `content` as the text field.
    return getattr(message, "content", message)


if __name__ == "__main__":
    payload = {
        "messages": [
            {
            "role": "user",
            "content": "Decide if the following query is a query that is creative or technical.\n\nThings that are creative are related to creating, composing, making new things, feeling, etc.\nThings that are technical are related to specific information about products in the inventory, such as: prices, quantity, availability, etc.\n\nLabel it as CREATIVE or TECHNICAL.\n\n\nEXAMPLES:\n    Give me suggestions on a nice look for a nightclub. Label: CREATIVE\n    What are the blue dresses you have available? Label: TECHNICAL\n    Give me three T-shirts for summer. Label: TECHNICAL\n    Give me a look for attending a wedding party. Label: CREATIVE\n\n\nQuery to be analyzed: Give me two sneakers with vibrant colors..\n\n\nOUTPUT:\n\nOnly output one token: the label, which should be \"TECHNICAL\" or \"CREATIVE\", in UPPERCASE.\nNo other characters other than \"TECHNICAL\" or \"CREATIVE\" should be generated in your response."
            }
        ],
        "top_p": None,
        "temperature": 0,
        "max_tokens": 2048,
        "model": "deepseek"
        }

    result = ask_deepseek_r1(payload)
    print(f"Response: {result}\n")
