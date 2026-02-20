"""Tests that verify CORS header behavior for chat_server.

Run with:
    cd /Users/zshirwan/projects/RAG_Chatbot/code
    python -m pytest unit_tests/test_cors.py -v
"""
from __future__ import annotations
import os
import sys

# Ensure the code directory is on the path before importing anything.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import importlib
import types
import unittest


def _make_app(flask_manage_cors: bool):
    """Re-import chat_server with a specific FLASK_MANAGE_CORS value.

    Because chat_server runs module-level code at import time we need to
    reload it (or import it fresh) for each test scenario.
    """
    os.environ["FLASK_MANAGE_CORS"] = "true" if flask_manage_cors else "false"

    # Remove cached module so the re-import re-runs module-level code.
    for key in list(sys.modules):
        if key == "chat_server":
            del sys.modules[key]

    # Stub heavy dependencies so we can import chat_server without the full
    # ML stack installed in this test environment.
    _stub_heavy_deps()

    import chat_server  # noqa: PLC0415
    return chat_server.app


def _stub_heavy_deps():
    """Insert lightweight stubs for modules that aren't needed for CORS tests."""
    stubs = [
        "llm_utils",
        "execute_prompt",
        "utils",
        "pandas",
        "chromadb",
        "sentence_transformers",
        "openai",
        "dotenv",
    ]
    for name in stubs:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            # Provide the symbols chat_server.py actually imports from these modules
            stub.set_client_name = lambda *a, **kw: None          # type: ignore[attr-defined]
            stub.get_client_name = lambda: "demo"                  # type: ignore[attr-defined]
            stub.get_client_chroma_db_path = lambda: "/tmp/db"     # type: ignore[attr-defined]
            stub.do_execute_prompt = lambda *a, **kw: "stub answer"# type: ignore[attr-defined]
            sys.modules[name] = stub


class TestCorsHeadersManaged(unittest.TestCase):
    """When FLASK_MANAGE_CORS=True Flask should add CORS headers."""

    def setUp(self):
        self.app = _make_app(flask_manage_cors=True)
        self.client = self.app.test_client()

    def test_get_has_cors_header(self):
        resp = self.client.get(
            "/?client_site=demo&prompt=hello",
            headers={"Origin": "http://localhost:8002"},
        )
        acao = resp.headers.getlist("Access-Control-Allow-Origin")
        self.assertTrue(
            len(acao) > 0,
            "Expected at least one Access-Control-Allow-Origin header",
        )
        # There must be exactly ONE such header — duplicates break browsers.
        self.assertEqual(
            len(acao),
            1,
            f"Duplicate Access-Control-Allow-Origin headers found: {acao}",
        )
        self.assertEqual(acao[0], "*")

    def test_options_preflight_has_cors_header(self):
        resp = self.client.options(
            "/",
            headers={
                "Origin": "http://localhost:8002",
                "Access-Control-Request-Method": "GET",
            },
        )
        acao = resp.headers.getlist("Access-Control-Allow-Origin")
        self.assertTrue(len(acao) > 0, "OPTIONS response missing Access-Control-Allow-Origin")
        self.assertEqual(len(acao), 1, f"Duplicate ACAO on OPTIONS: {acao}")


class TestCorsHeadersNotManaged(unittest.TestCase):
    """When FLASK_MANAGE_CORS=False Flask must NOT add any CORS headers
    (nginx handles it; duplicates break the browser CORS check)."""

    def setUp(self):
        self.app = _make_app(flask_manage_cors=False)
        self.client = self.app.test_client()

    def test_get_has_no_cors_header(self):
        resp = self.client.get(
            "/?client_site=demo&prompt=hello",
            headers={"Origin": "http://localhost:8002"},
        )
        acao = resp.headers.getlist("Access-Control-Allow-Origin")
        self.assertEqual(
            len(acao),
            0,
            f"Flask should NOT add CORS headers in production mode, but got: {acao}",
        )


class TestRemoteServerNoDuplicateCors(unittest.TestCase):
    """Integration test — hit the live remote server and check for duplicate headers.

    This test is skipped automatically when there is no network access.
    """

    REMOTE_URL = "https://chatbot.zakishirwani.com/"

    def test_no_duplicate_acao_header(self):
        import urllib.request
        import urllib.error
        import ssl

        # Create an SSL context that skips certificate verification so the test
        # works in environments that lack the root CA bundle (e.g. macOS without
        # the certifi certificates installed for Python).
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            self.REMOTE_URL + "?client_site=bolangaro.com&prompt=hello",
            headers={
                "Origin": "http://localhost:8002",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                # In Python's http.client, headers with the same name are joined
                # with ", ". If there are duplicates the joined value will contain
                # a comma between the two occurrences (e.g. "*, *").
                acao = resp.headers.get("Access-Control-Allow-Origin", "")
                self.assertNotIn(
                    ",",
                    acao,
                    f"Duplicate Access-Control-Allow-Origin detected on remote server: '{acao}'. "
                    "Make sure FLASK_MANAGE_CORS is NOT set on the production server.",
                )
        except urllib.error.URLError as exc:
            self.skipTest(f"Remote server unreachable: {exc}")


if __name__ == "__main__":
    unittest.main()



