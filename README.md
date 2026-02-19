# RAG Chatbot Code

This directory contains the backend code for a Retrieval-Augmented Generation (RAG) chatbot that can be customized per client site (e.g. `bolangaro.com`). The backend exposes a simple HTTP API that your web widget(s) can call to get answers.

## Overview

Main pieces in this folder:

- `chat_server.py` – Flask HTTP server that exposes the chatbot API.
- `execute_prompt.py` – Orchestrates the end-to-end flow for a user query.
- `query_faq_chroma.py` – Uses ChromaDB to answer FAQ-style questions from indexed documents.
- `query_products.py` – Searches product / inventory data and uses the LLM to craft responses.
- `query_type.py` – Classifies a query (e.g. FAQ vs PRODUCT / other) using an LLM.
- `llm_utils.py` – Common helpers for calling different LLMs and building prompts.
- `call_ollama.py` – Talks to a local Ollama instance (e.g. `llama3`).
- `call_deepseek.py` – Talks to the DeepSeek API (OpenAI-compatible).
- `spell_corrector.py` – Optional spelling correction for user queries.
- `metadata_filters.py`, `utils.py`, `technical_or_creative.py`, `setup_new_client.py` – Supporting utilities and setup tools.

Client-specific configuration and data lives under `../chatbot-client-sites/<client-domain>/` and includes:

- `chroma_db/` – Vector database for FAQ / document retrieval.
- `faq/`, `documents/`, etc. – Plain-text content that can be indexed.
- `product_metadata/`, `inventory/` – Product and inventory metadata.
- `system_prompts/` – System prompt templates that steer the LLM.

The active client is controlled via the `.env` file in this folder.

---

## 1. Environment and configuration

### 1.1. Python version

Use Python 3.11+ (3.13 is currently used in development).

### 1.2. Install dependencies

From the `code/` directory:

```bash
cd /Users/zshirwan/projects/RAG_Chatbot/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.3. Environment variables (`.env`)

The backend is configured with a `.env` file in this directory (already present). Typical contents:

```dotenv
OPENAI_API_KEY="..."          # If you use OpenAI or compatible APIs
HF_TOKEN="..."                # Optional Hugging Face token
DEEPSEEK_API_KEY="..."       # DeepSeek API key (if using DeepSeek)

DEFAULT_LLM="llama3"          # Default model name (e.g. for Ollama)

CLIENT_SITES_LOCATION="../chatbot-client-sites"
CLIENT_SITE="bolangaro.com"   # Active client folder name

DEBUG=True
```

Notes:

- Keys in this example are placeholders; never commit real API keys.
- `CLIENT_SITES_LOCATION` is a path to the parent directory that contains all client-specific folders.
- `CLIENT_SITE` must match a folder name inside `CLIENT_SITES_LOCATION`.

The app loads this `.env` via `python-dotenv` (see `llm_utils.py` / `utils.py`). If you update `.env`, restart the server.

---

## 2. Running the chatbot server locally

The HTTP server is implemented in `chat_server.py` using Flask.

### 2.1. Start the server

From the `code/` directory with your virtualenv activated:

```bash
python chat_server.py
```

By default this starts a server on `http://0.0.0.0:8001` with debug mode enabled.

### 2.2. Health / smoke test

The root route (`/`) serves as the main chat endpoint. You can test it directly:

```bash
curl "http://localhost:8001/?prompt=What+are+your+store+hours%3F"
```

Expected JSON shape:

```json
{
  "answer": "...model-generated answer..."
}
```

If you see a response, your stack (LLM + Chroma + prompts) is wired correctly.

---

## 3. HTTP API

### 3.1. Endpoint

- **URL:** `/`
- **Methods:** `GET`, `POST`, `OPTIONS`
- **CORS:** CORS is enabled for `http://localhost:8001` and `http://127.0.0.1:8001` so that the JS widget can call it from another origin.

### 3.2. Request format

You can call the endpoint in two ways:

1. **GET with query param**

   - `prompt`: The user’s question.

   Example:
   ```bash
   curl "http://localhost:8001/?prompt=Do+you+have+waterproof+jackets%3F"
   ```

2. **POST (JSON or form-encoded)**

   - JSON body: `{ "prompt": "Do you have waterproof jackets?" }`
   - Or form-encoded body with `prompt` field.

   Example:
   ```bash
   curl -X POST "http://localhost:8001/" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Do you have waterproof jackets?"}'
   ```

### 3.3. Response format

- Always JSON: `{ "answer": <string or structured result> }`
- The exact shape of `answer` depends on the downstream workflow (`query_products`, FAQ, etc.). In many cases it will be a plain string; some paths may return richer objects.

---

## 4. Query processing pipeline

Here is what happens when `chat_server.py` receives a prompt:

1. **HTTP layer** (`chat_server.py`)
   - Extracts `prompt` from GET/POST.
   - Logs the received prompt.
   - Calls `do_execute_prompt(prompt)` from `execute_prompt.py`.

2. **Input cleaning & spell correction** (`execute_prompt.py`)
   - Trims whitespace and validates the string.
   - Optionally applies spelling correction via `SpellCorrector`.

3. **FAQ retrieval via Chroma** (`query_faq_chroma.py`)
   - Attempts to answer immediately from the Chroma-based FAQ collection.
   - If this returns a non-empty result, that is sent back to the user.
   - If Chroma raises an error, the system falls back to a generic LLM prompt asking the user to rephrase and answering based on existing context.

4. **Query type classification** (`query_type.py`)
   - If FAQ retrieval didn’t answer, the query is classified (e.g. `PRODUCT` vs other) using an LLM.

5. **Product search flow** (`query_products.py`)
   - If classified as `PRODUCT`, the system queries product and inventory data.
   - On error, it falls back to the same generic “rephrase / answer from context” LLM call.

6. **Fallback system prompt** (`system_prompts/fall_through_query_type.txt`)
   - For non-FAQ, non-product queries, the code loads a fall-through system prompt from the active client’s `system_prompts` directory and calls the LLM to generate an answer.

All LLM calls are funneled through helpers in `llm_utils.py`, which build parameter dictionaries (`generate_params_dict`) and perform the actual call (`generate_with_single_input`).

---

## 5. LLM backends

This project can route to several backends:

- **Local Ollama** via `call_ollama.py` (e.g. `llama3`).
  - Ensure the Ollama daemon is running locally and that the `llama3` model is pulled.
  - `ask_local_ollama_llama3(payload)` expects a dict with either a `prompt` string or a `messages` list.

- **DeepSeek** via `call_deepseek.py`.
  - Requires `DEEPSEEK_API_KEY` and optionally `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`.
  - `ask_deepseek_r1(payload)` expects an OpenAI-style `messages` list and optional `temperature`, `top_p`, `max_tokens`.

- **Other OpenAI-compatible providers** may be configured in `llm_utils.py` using `OPENAI_API_KEY` and `DEFAULT_LLM`.

Which backend is used for a given call is configured in `llm_utils.py` and via environment variables.

---

## 6. Client sites and data

The `CLIENT_SITES_LOCATION` and `CLIENT_SITE` env vars select the active client. For example, with:

```dotenv
CLIENT_SITES_LOCATION="../chatbot-client-sites"
CLIENT_SITE="bolangaro.com"
```

the code will look for data in:

```text
../chatbot-client-sites/bolangaro.com/
  chroma_db/
  faq/
  inventory/
  product_metadata/
  system_prompts/
```

### 6.1. Changing the active client

1. Create a new folder under `../chatbot-client-sites/<your-domain>/` with the same structure as an existing client.
2. Populate `faq`, `documents`, `inventory`, and `system_prompts` as needed.
3. Point `.env` to the new client:

   ```dotenv
   CLIENT_SITE="your-domain.com"
   ```

4. Rebuild any Chroma collections for the new client using the indexing scripts (see below).

---

## 7. Indexing / maintenance scripts

Several helper scripts live in this directory to build or maintain the vector indexes:

- `create_chromadb_faq_collection.py` – Build/refresh the FAQ Chroma collection from client text files.
- `create_chromadb_products_collection.py` – Build/refresh the product collection.
- `drop_faq_collection.py` – Drop an existing FAQ collection.
- `setup_new_client.py` – Bootstrap a new client directory structure and prompts.

Typical usage from the `code/` directory (after activating your venv):

```bash
python create_chromadb_faq_collection.py
python create_chromadb_products_collection.py
```

These scripts read configuration from `.env` and client folders, then populate `chroma_db/` under the client site.

---

## 8. Local testing utilities

Most modules include a small `if __name__ == "__main__":` block that you can run directly for quick manual tests. Examples:

```bash
python execute_prompt.py          # Runs a few hard-coded test queries
python call_ollama.py             # Tests a classification prompt via local Ollama
python call_deepseek.py           # Tests the same via DeepSeek
```

---

## 9. Development tips

- Keep `.env` out of version control or scrub secrets before committing.
- When changing LLM providers or model names, update `DEFAULT_LLM` and any related variables in `.env`, and adjust `llm_utils.py` if needed.
- If you add new client sites, consider creating a small script or README under `chatbot-client-sites/<client>/` documenting their specific prompts and data sources.

---

## 10. Troubleshooting

- **Server starts but responses are empty or generic**
  - Check that Chroma collections exist under the active client and are populated.
  - Verify that your LLM backend is reachable and API keys are valid.

- **CORS / browser errors (`TypeError: Failed to fetch`)**
  - Ensure your frontend is loaded from an origin that’s included in `allowed_origins` inside `chat_server.py`.
  - Adjust `allowed_origins` or add `flask-cors` if needed.

- **Import errors (`ModuleNotFoundError` for local modules)**
  - Make sure you’re running commands from the `code/` directory so that relative imports and `sys.path` tweaks in `chat_server.py` behave correctly.

This README should give you enough to run, test, and extend the RAG chatbot backend in this `code/` directory.
