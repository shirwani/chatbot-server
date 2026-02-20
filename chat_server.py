from __future__ import annotations
import os
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

# FLASK_MANAGE_CORS controls whether Flask adds CORS headers itself.
#
# Set FLASK_MANAGE_CORS=True  in your local .env so that Flask adds the headers
# when running without a reverse proxy (e.g. python chat_server.py on port 8001).
#
# In production the nginx reverse proxy already adds Access-Control-Allow-Origin
# to every response, so Flask must NOT add them too — duplicate headers cause
# browsers to reject the response with a CORS error even when both copies say "*".
# Leave FLASK_MANAGE_CORS unset (or set it to False) on the production server.
_flask_manage_cors = os.environ.get("FLASK_MANAGE_CORS", "false").strip().lower() in ("1", "true", "yes")

if _flask_manage_cors:
    # Allow any origin. Restrict this if you want tighter security.
    _allowed_origins = {"http://localhost:8002"}

    # Try to configure flask-cors if it's installed. If not, we'll fall back to
    # the manual @after_request CORS handler below.
    try:
        from flask_cors import CORS  # type: ignore

        CORS(
            app,
            origins="*" if "*" in _allowed_origins else list(_allowed_origins),
            supports_credentials=False,
            methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization"],
            max_age=86400,
            automatic_options=True,
        )
        print("[chat_server] flask_cors applied (FLASK_MANAGE_CORS=True).", file=sys.stderr)
    except Exception as e:
        print(f"[chat_server] flask_cors not applied ({type(e).__name__}: {e}). Falling back to manual CORS.", file=sys.stderr)

        @app.after_request
        def add_cors_headers(response):
            """Manually add CORS headers when flask-cors is unavailable."""
            origin = request.headers.get("Origin", "")
            if "*" in _allowed_origins:
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Vary"] = "Origin"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
                response.headers["Access-Control-Max-Age"] = "86400"
            elif origin and origin in _allowed_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Vary"] = "Origin"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
                response.headers["Access-Control-Max-Age"] = "86400"
            return response
else:
    print("[chat_server] FLASK_MANAGE_CORS is not set — CORS headers will NOT be added by Flask "
          "(expected in production where nginx handles CORS).", file=sys.stderr)


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
