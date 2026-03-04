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
from session_manager import SessionManager

app = Flask(__name__)

session_manager = SessionManager()

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
    _allowed_origins = {"*"}

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


def _first_non_empty(*values):
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _coerce_positive_int(value):
    try:
        as_int = int(value)
        if as_int > 0:
            return as_int
    except Exception:
        pass
    return None


def _stringify_answer(answer):
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        nested = answer.get("answer")
        if isinstance(nested, str):
            return nested
    return "" if answer is None else str(answer)


@app.route("/session", methods=["POST", "OPTIONS"])
def create_session():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    json_data = request.get_json(silent=True) or {}
    client_site = _first_non_empty(
        json_data.get("client_site"),
        request.form.get("client_site"),
        request.args.get("client_site"),
        "goai.com",
    )
    max_context_chars = _coerce_positive_int(
        _first_non_empty(
            json_data.get("max_context_chars"),
            request.form.get("max_context_chars"),
            request.args.get("max_context_chars"),
        )
    )

    session_id = session_manager.create_session(client_site=client_site, max_context_chars=max_context_chars)
    print(f"[chat_server] Created session {session_id} for client_site='{client_site}'.")
    return jsonify({"session_id": session_id, "client_site": client_site})


@app.route("/session/<session_id>", methods=["DELETE", "OPTIONS"])
def delete_session(session_id: str):
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    removed = session_manager.end_session(session_id)
    if not removed:
        return jsonify({"error": "session not found"}), 404

    print(f"[chat_server] Deleted session {session_id}.")
    return jsonify({"ok": True})


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
        client_site = (request.args.get("client_site", "goai.com") or "goai.com").strip().lower()
        prompt = request.args.get("prompt", "How can I contact customer service?")
        conversation_context = request.args.get("conversation_context", "") or None
        session_id = request.args.get("session_id")
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
        session_id = _first_non_empty(
            json_data.get("session_id"),
            request.form.get("session_id"),
            request.args.get("session_id"),
        )

    active_session = None
    if session_id:
        active_session = session_manager.get_session(session_id)
        if not active_session:
            return jsonify({"error": "Invalid or expired session_id"}), 400
        conversation_context = session_manager.get_context(session_id)
        client_site = active_session.client_site or client_site

    client_site = (client_site or "goai.com").strip().lower()

    print("Client site:", client_site)
    print("Session id:", session_id or "<none>")
    print("Received prompt:", prompt)
    if conversation_context:
        print("Conversation context (truncated):", conversation_context[:200], "...")

    set_client_name(client_site)

    print(f"Set client name to '{get_client_name()}' for this request.")
    print(f"chromadb_path is '{get_client_chroma_db_path()}' for this request.")

    answer = do_execute_prompt(prompt, conversation_context=conversation_context)

    if active_session:
        session_manager.append_turn(session_id, prompt, _stringify_answer(answer))

    return jsonify({"answer": answer})


if __name__ == "__main__":
    # Bind on all interfaces so the client (running on a different port) can reach it
    app.run(host="0.0.0.0", port=8001, debug=True)
