"""pipelines/test_mcp_config.py — the server settings every pipeline reads (Ticket 13).

Run: uv run pytest pipelines/test_mcp_config.py -q
"""
from __future__ import annotations

from pipelines import mcp_config


def test_defaults_reproduce_the_local_mirror(monkeypatch):
    for var in ("MCP_SERVER_URL", "MCP_SERVER_AUTH_TOKEN", "MCP_SEARCH_TOOL"):
        monkeypatch.delenv(var, raising=False)
    assert mcp_config.mcp_url() == "http://127.0.0.1:8000/mcp"
    assert mcp_config.mcp_headers() is None
    assert mcp_config.search_tool() == "search_fastapi_knowledge_sources"


def test_kapa_style_settings(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_URL", "https://acme.mcp.kapa.ai")
    monkeypatch.setenv("MCP_SERVER_AUTH_TOKEN", "secret-key")
    monkeypatch.setenv("MCP_SEARCH_TOOL", "search_acme_knowledge_sources")
    assert mcp_config.mcp_url() == "https://acme.mcp.kapa.ai"
    assert mcp_config.mcp_headers() == {"Authorization": "Bearer secret-key"}
    assert mcp_config.search_tool() == "search_acme_knowledge_sources"


def test_blank_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_URL", "  ")
    monkeypatch.setenv("MCP_SERVER_AUTH_TOKEN", "")
    assert mcp_config.mcp_url() == "http://127.0.0.1:8000/mcp"
    assert mcp_config.mcp_headers() is None


def test_describe_never_leaks_the_token(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_AUTH_TOKEN", "super-secret-value")
    assert "super-secret-value" not in mcp_config.describe()
