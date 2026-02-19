import ollama
from utils import *

@dbg_print
def ask_local_ollama_llama3(payload: dict = None):
    """Call a local Ollama model using the fields in `payload`.

    Expected payload keys:
      - model: str
      - prompt: str (for single-shot generation) OR
      - messages: list[dict] (for chat-style, if you later switch to `ollama.chat`)
      - temperature, top_p, max_tokens, etc. as optional controls (currently ignored if unsupported).
    """

    if payload is None:
        raise ValueError("payload must be provided to ask_local_ollama_llama3")

    # If `messages` is present and you later want chat-style, you could switch to ollama.chat.
    # For now, we flatten to a prompt string if 'prompt' isn't already present.
    if "prompt" not in payload:
        if "messages" in payload:
            # Simple flatten: join all message contents; adjust as needed.
            prompt_text = "\n".join(m.get("content", "") for m in payload["messages"])
            payload = {**payload, "prompt": prompt_text}
        else:
            raise ValueError("payload must contain either 'prompt' or 'messages'")

    # Minimal supported args for current ollama.generate client: model and prompt
    generate_args = {
        "model": payload.get("model", "llama3"),
        "prompt": payload["prompt"],
    }

    response = ollama.generate(**generate_args)
    # Ollama's Python client returns a dict with a 'response' field containing the text
    return response["response"]


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
        "max_tokens": 1,
        "model": "llama3"
        }

    result = ask_local_ollama_llama3(payload)
    print(f"Response: {result}\n")
