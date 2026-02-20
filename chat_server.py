from __future__ import annotations
import sys
from flask import Flask, jsonify, request
sys.path.append('')  # Add the code directory to the Python path for imports
from llm_utils import (
    set_client_name,
    get_client_name,
    get_client_chroma_db_path,
)
from execute_prompt import do_execute_prompt

app = Flask(__name__)

# Enable cross-origin calls from the local http server / embedded sites.
# This avoids browser "TypeError: Failed to fetch" caused by CORS blocking.
# Change this to only allow specific url's if you want to be more restrictive.
allowed_origins = {"*"}

try:
    from flask_cors import CORS  # type: ignore

    CORS(
        app,
        origins="*" if "*" in allowed_origins else list(allowed_origins),
        supports_credentials=False,
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        max_age=86400,
        automatic_options=True,
    )
except Exception:
    # If flask-cors isn't installed yet, we'll still add headers manually below.
    pass


@app.after_request
def add_cors_headers(response):
    """Ensure all responses include suitable CORS headers.

    This covers cases where flask-cors is not installed or not applied.
    """
    origin = request.headers.get("Origin", "")

    # If '*' is configured, allow any origin. Otherwise, only allow explicit ones.
    if "*" in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Max-Age"] = "86400"
    elif origin and origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Max-Age"] = "86400"

    return response


# RAG server


@app.route("/", methods=["GET", "POST", "OPTIONS"])
def home():
    # Handle CORS preflight explicitly just in case
    if request.method == "OPTIONS":
        resp = jsonify({"ok": True})
        return resp, 200

    # For GET requests from the widget, the question usually comes in as a
    # query parameter. For POST, it could be JSON or form-encoded; handle both.
    if request.method == "GET":
        client_site = (request.args.get("client_site", "") or "").strip().lower()
        prompt = request.args.get("prompt", "How can I contact customer service?")
        conversation_context = request.args.get("conversation_context", "") or None
    else:
        json_data = request.get_json(silent=True) or {}
        client_site = (
                json_data.get("client_site")
                or request.form.get("client_site")
                or request.args.get("client_site", "")
        )
        prompt = (
            json_data.get("prompt")
            or request.form.get("prompt")
            or request.args.get("prompt", "What are your hours?")
        )
        conversation_context = (
            json_data.get("conversation_context")
            or request.form.get("conversation_context")
            or request.args.get("conversation_context", "")
        ) or None

    print("Client site:", client_site)
    print("Received prompt:", prompt)
    if conversation_context:
        print("Conversation context (truncated):", conversation_context[:200], "...")

    set_client_name(client_site)

    print(f"Set client name to '{get_client_name()}' for this request.")
    print(f"chromadb_path is '{get_client_chroma_db_path()}' for this request.")

    answer = do_execute_prompt(prompt, conversation_context=conversation_context)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    # Bind on all interfaces so the client (running on a different port) can reach it
    app.run(host="0.0.0.0", port=8001, debug=True)
