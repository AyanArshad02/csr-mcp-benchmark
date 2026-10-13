"""pipelines/mcp_config.py — Where every pipeline finds the MCP search server (Ticket 13).

All three pipelines read their connection settings from here, so pointing the benchmark at a
different server, such as a Kapa hosted MCP server, is a settings change in `.env`:

    MCP_SERVER_URL=https://<subdomain>.mcp.kapa.ai
    MCP_SERVER_AUTH_TOKEN=<your Kapa API key>          # sent as "Authorization: Bearer <key>"
    MCP_SEARCH_TOOL=search_<product>_knowledge_sources

Kapa's hosted MCP server uses exactly that header and tool-name pattern
(docs.kapa.ai/retrieval/hosted-mcp-server, checked 2026-10-13). With none of these set, the
defaults reproduce every published run against the local mirror (`uv run python -m server.main`).
"""
from __future__ import annotations

import os

DEFAULT_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_TOOL = "search_fastapi_knowledge_sources"


def mcp_url() -> str:
    return os.environ.get("MCP_SERVER_URL", "").strip() or DEFAULT_URL


def mcp_headers() -> dict[str, str] | None:
    """Bearer auth header when MCP_SERVER_AUTH_TOKEN is set; None for the unauthenticated mirror."""
    token = os.environ.get("MCP_SERVER_AUTH_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else None


def search_tool() -> str:
    """Tool the single-shot baseline calls by name (the agents discover tools themselves)."""
    return os.environ.get("MCP_SEARCH_TOOL", "").strip() or DEFAULT_TOOL


def describe() -> str:
    """One-line summary for logs; never prints the token."""
    auth = "bearer token set" if mcp_headers() else "no auth"
    return f"MCP server {mcp_url()} ({auth}), search tool {search_tool()}"
