"""Download raw FastAPI docs Markdown from GitHub and map each file to its canonical URL.

Usage: uv run python -m ingest.download
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO = "tiangolo/fastapi"
DOCS_PREFIX = "docs/en/docs/"
RAW_DIR = Path("data/raw_docs")
SITE_BASE = "https://fastapi.tiangolo.com"

# Internal tooling docs, not developer-facing FastAPI content — excluded from the corpus.
EXCLUDED_FILES = {"_llm-test.md"}

# FastAPI's docs use a custom MkDocs macro, `{* path hl[...] title["..."] *}`, that the
# real site-build pipeline inlines into actual example code from a separate source file.
# Raw GitHub Markdown never runs that build step, so without expanding these ourselves,
# ~20% of chunks end up with a literal "{* ... *}" placeholder instead of real code.
_MACRO_RE = re.compile(r"\{\*\s*(\S+)[^*]*\*\}")
_LEADING_DOTDOT_RE = re.compile(r"^(\.\./)+")
_LANG_BY_EXT = {".py": "python", ".js": "javascript", ".json": "json", ".yml": "yaml", ".yaml": "yaml"}


@dataclass(frozen=True)
class DocFile:
    relative_path: str  # path under docs/en/docs/, e.g. "tutorial/dependencies.md"
    source_url: str  # canonical published URL


def _default_branch(client: httpx.Client) -> str:
    resp = client.get(f"https://api.github.com/repos/{REPO}")
    resp.raise_for_status()
    return resp.json()["default_branch"]


def _list_doc_paths(client: httpx.Client, branch: str) -> list[str]:
    resp = client.get(
        f"https://api.github.com/repos/{REPO}/git/trees/{branch}",
        params={"recursive": "1"},
    )
    resp.raise_for_status()
    tree = resp.json()["tree"]
    return [
        entry["path"]
        for entry in tree
        if entry["type"] == "blob"
        and entry["path"].startswith(DOCS_PREFIX)
        and entry["path"].endswith(".md")
        and Path(entry["path"]).name not in EXCLUDED_FILES
    ]


def path_to_url(relative_path: str) -> str:
    """Map a docs/en/docs/-relative Markdown path to its canonical fastapi.tiangolo.com URL.

    FastAPI's docs site is built with MkDocs Material's clean-URL convention:
    index.md maps to its directory's URL; other files drop the .md suffix and
    get a trailing slash. Spot-check a sample of mapped URLs before trusting at scale.
    """
    p = Path(relative_path)
    if p.name == "index.md":
        parts = p.parent.parts
    else:
        parts = p.with_suffix("").parts
    if not parts:
        return f"{SITE_BASE}/"
    return f"{SITE_BASE}/" + "/".join(parts) + "/"


def _resolve_repo_path(macro_path: str) -> str:
    """Macro paths are relative to docs/en/docs/ (e.g. '../../docs_src/foo.py'), but
    every observed case resolves cleanly by just stripping the leading '../' segments
    and treating the remainder as relative to the repo root."""
    return _LEADING_DOTDOT_RE.sub("", macro_path)


def _fetch_source_file(client: httpx.Client, branch: str, repo_path: str) -> str | None:
    raw_url = f"https://raw.githubusercontent.com/{REPO}/{branch}/{repo_path}"
    resp = client.get(raw_url)
    if resp.status_code != 200:
        logger.warning(f"Could not fetch macro source {repo_path} ({resp.status_code})")
        return None
    return resp.text


def _expand_macros(client: httpx.Client, branch: str, content: str, source_cache: dict[str, str | None]) -> str:
    def replace(match: re.Match[str]) -> str:
        macro_path = match.group(1)
        repo_path = _resolve_repo_path(macro_path)
        if repo_path not in source_cache:
            source_cache[repo_path] = _fetch_source_file(client, branch, repo_path)
        source = source_cache[repo_path]
        if source is None:
            return match.group(0)  # leave the macro placeholder if the fetch failed
        lang = _LANG_BY_EXT.get(Path(repo_path).suffix, "")
        return f"```{lang}\n{source.rstrip()}\n```"

    return _MACRO_RE.sub(replace, content)


def download_fastapi_docs(dest_dir: Path = RAW_DIR) -> list[DocFile]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    doc_files: list[DocFile] = []
    source_cache: dict[str, str | None] = {}

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        branch = _default_branch(client)
        logger.info(f"Default branch: {branch}")

        paths = _list_doc_paths(client, branch)
        logger.info(f"Found {len(paths)} Markdown files under {DOCS_PREFIX}")

        for path in paths:
            relative_path = path[len(DOCS_PREFIX) :]
            raw_url = f"https://raw.githubusercontent.com/{REPO}/{branch}/{path}"
            resp = client.get(raw_url)
            resp.raise_for_status()

            content = _expand_macros(client, branch, resp.text, source_cache)

            local_path = dest_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(content, encoding="utf-8")

            doc_files.append(
                DocFile(relative_path=relative_path, source_url=path_to_url(relative_path))
            )

    failed = sum(1 for v in source_cache.values() if v is None)
    logger.info(
        f"Downloaded {len(doc_files)} files to {dest_dir}/; "
        f"expanded {len(source_cache) - failed}/{len(source_cache)} unique macro source files"
        f"{f' ({failed} failed)' if failed else ''}"
    )
    return doc_files


if __name__ == "__main__":
    files = download_fastapi_docs()
    print(f"\n{len(files)} files downloaded. Sample URL mappings:")
    for f in files[:10]:
        print(f"  {f.relative_path:50s} -> {f.source_url}")
