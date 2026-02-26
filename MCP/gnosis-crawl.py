#!/usr/bin/env python3
"""
Gnosis Crawl MCP Bridge
=======================

The world's only agentic web crawler — MCP interface.

Exposes both traditional crawl tools (Mode A) and an autonomous agent
loop (Mode B) to any MCP-compatible host. Built using the brain of a
human that knows about distributed crawling architectures.

Defaults to LOCAL crawling (gnosis-crawl:8080) with NO AUTH required.
Automatically fixes localhost/127.0.0.1 references to gnosis-crawl:8080.

Tools:
  - crawl_url: fetch markdown from a single URL (supports JS injection)
  - crawl_batch: process multiple URLs (supports JS injection, async/collated)
  - raw_html: fetch raw HTML without markdown conversion
  - download_file: download files (PDFs, etc.) through the crawler
  - agent_run: submit a multi-step task to the autonomous agent (Mode B)
  - agent_status: check status of a running agent task
  - ghost_extract: Ghost Protocol — screenshot + vision AI extraction (anti-bot bypass)
  - set_auth_token: save Wraith API token to .wraithenv (optional)
  - crawl_status: report configuration (base URL, token presence)
  - crawl_validate: validate whether crawled text is usable
  - crawl_search: fuzzy search across cached crawl files
  - crawl_cache_list: list cached crawl files and metadata
  - crawl_remote_search: fuzzy search against crawler service cache
  - crawl_remote_cache_list: list crawler service cache entries
  - crawl_remote_cache_doc: fetch one cached document by id from service
  - mesh_peers: list mesh peers and their health/load status
  - mesh_status: get this node's mesh status and load metrics

JavaScript Injection for Markdown Extraction:
  - For crawl_url and crawl_batch: pass javascript_payload to inject code
  - The JavaScript will execute FIRST on the page, modifying the DOM
  - Then markdown extraction will run on the modified content
  - Useful for expanding hidden content, interacting with JS, etc.

Env/config:
  - WRAITH_AUTH_TOKEN        (optional, preferred if present)
  - GNOSIS_CRAWL_BASE_URL    (overrides default gnosis-crawl:8080)
  - .wraithenv file in repo root with line: WRAITH_AUTH_TOKEN=...

Defaults:
  - Local: "http://gnosis-crawl:8080" (default, always used unless overridden)
  - Auth: None required
"""

import difflib
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from mcp.server.fastmcp import FastMCP, Context
from urllib.parse import urlparse, unquote, quote

mcp = FastMCP("gnosis-crawl")

# Default to local gnosis-crawl:8080 - always the primary server
LOCAL_SERVER_URL = "http://gnosis-crawl:8080"
WRAITH_ENV_FILE = os.path.join(os.getcwd(), ".wraithenv")


def _normalize_server_url(url: str) -> str:
    """
    Normalize server URLs, converting localhost/127.0.0.1 to gnosis-crawl:8080.
    
    If someone passes localhost or 127.0.0.1 with any port, convert it to the
    standard gnosis-crawl:8080 local server.
    
    Args:
        url: Server URL to normalize
    
    Returns:
        str: Normalized URL (gnosis-crawl:8080 if localhost detected, otherwise original)
    """
    if not url:
        return LOCAL_SERVER_URL
    
    try:
        parsed = urlparse(url)
        # Fix localhost/127.0.0.1 references to use gnosis-crawl:8080
        if parsed.hostname in ("localhost", "127.0.0.1"):
            return LOCAL_SERVER_URL
    except Exception:
        pass
    
    return url


def _extract_domain(url: str) -> str:
    """
    Extract the domain name from a URL for storage organization.
    
    Args:
        url: Full URL to parse (e.g., "https://example.com/path")
    
    Returns:
        str: Lowercase domain name (e.g., "example.com") or "unknown" if parsing fails
    """
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return "unknown"

def _filename_from_content_disposition(value: str) -> Optional[str]:
    if not value:
        return None
    match = re.search(r"filename\\*=([^']*)''([^;]+)", value, flags=re.IGNORECASE)
    if match:
        return unquote(match.group(2))
    match = re.search(r'filename=\"?([^\";]+)\"?', value, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def _safe_filename(name: str) -> str:
    candidate = Path(name).name
    candidate = candidate.encode("ascii", "ignore").decode("ascii")
    if not candidate:
        return "download"
    safe = "".join(ch if ch.isalnum() or ch in " ._-()" else "_" for ch in candidate)
    return safe or "download"

def _is_google_host(url: str) -> bool:
    """Block direct Google crawling so users route through the serpapi-search MCP tool."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    host = host.lower()
    return host.endswith("google.com")

def _get_auth_token() -> Optional[str]:
    """
    Retrieve Wraith API authentication token from environment or .wraithenv file.
    
    Checks WRAITH_AUTH_TOKEN environment variable first, then falls back to
    reading from .wraithenv file in the current working directory.
    
    Returns None by default (no auth required for local gnosis-crawl:8080).
    
    Returns:
        Optional[str]: Authentication token if found, None otherwise
    """
    # Env wins
    tok = os.environ.get("WRAITH_AUTH_TOKEN")
    if tok:
        return tok.strip()
    # Fallback to .wraithenv
    try:
        if os.path.exists(WRAITH_ENV_FILE):
            with open(WRAITH_ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("WRAITH_AUTH_TOKEN="):
                        return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return None



CRAWL_CACHE_DIR = os.environ.get(
    "CRAWL_CACHE_DIR",
    os.path.join(os.getcwd(), "crawl_cache"),
)

# Content quality thresholds aligned to crawler-side classifier behavior.
_THIN_CHAR_THRESHOLD = 80
_THIN_WORD_THRESHOLD = 15
_MEDIUM_THIN_CHAR_THRESHOLD = 600
_MEDIUM_THIN_WORD_THRESHOLD = 120

# Patterns that indicate bot-block/challenge pages.
_BLOCK_PATTERNS = [
    re.compile(r"cloudflare", re.I),
    re.compile(r"just a moment", re.I),
    re.compile(r"please verify you are a human", re.I),
    re.compile(r"captcha", re.I),
]

# Known error-page signatures that should never be treated as sufficient.
_ERROR_PAGE_SIGNATURES = [
    "error code: 404",
    "you've arrived at an empty lot",
    "page not found",
    "doesn't look like there's anything at this address",
    "access denied",
]

# Patterns stripped before measuring substantive text length.
_NAV_NOISE_RE = re.compile(
    r"(skip to (?:main )?content|cookie|privacy policy|terms of service"
    r"|©|all rights reserved|toggle navigation|hamburger|navbar)",
    re.I,
)


def _strip_markdown_noise(text: str) -> str:
    """Remove markdown links, images, nav boilerplate — keep body text."""
    # Remove images.
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove links but keep anchor text.
    text = re.sub(r"\[([^\]]*)\]\(.*?\)", r"\1", text)
    # Remove heading markers.
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    # Remove nav-ish boilerplate.
    text = _NAV_NOISE_RE.sub("", text)
    return text.strip()


def _assess_content_quality(
    content: str,
    status_code: Optional[int] = None,
    blocked: bool = False,
) -> Dict[str, Any]:
    """
    Classify crawl content quality.

    Returns:
        {
            "quality": "empty" | "minimal" | "blocked" | "sufficient",
            "char_count": int,      # substantive chars after stripping noise
            "word_count": int,
            "blocked_reason": str | None,
            "status_code": int | None,
            "reason": str,
        }
    """
    normalized = content or ""
    lowered = normalized.lower()
    stripped = _strip_markdown_noise(normalized)
    char_count = len(stripped)
    word_count = len(stripped.split())
    code: Optional[int] = None
    if status_code is not None:
        try:
            code = int(status_code)
        except Exception:
            code = None

    # Decision order:
    # 1) blocked
    if blocked:
        return {
            "quality": "blocked",
            "char_count": char_count,
            "word_count": word_count,
            "blocked_reason": "blocked flag from crawler",
            "status_code": code,
            "reason": "blocked flag",
        }
    for pat in _BLOCK_PATTERNS:
        if pat.search(lowered):
            return {
                "quality": "blocked",
                "char_count": char_count,
                "word_count": word_count,
                "blocked_reason": pat.pattern,
                "status_code": code,
                "reason": f"blocked signature: {pat.pattern}",
            }

    # 2) status_code handling
    if code is not None:
        if code >= 500:
            return {
                "quality": "blocked",
                "char_count": char_count,
                "word_count": word_count,
                "blocked_reason": f"status_code={code}",
                "status_code": code,
                "reason": f"http_{code}",
            }
        if code >= 400:
            return {
                "quality": "minimal",
                "char_count": char_count,
                "word_count": word_count,
                "blocked_reason": None,
                "status_code": code,
                "reason": f"http_{code}",
            }

    # 3) known error-page signatures
    if any(sig in lowered for sig in _ERROR_PAGE_SIGNATURES):
        return {
            "quality": "minimal",
            "char_count": char_count,
            "word_count": word_count,
            "blocked_reason": None,
            "status_code": code,
            "reason": "error-page signature",
        }

    # 4) thin/medium-thin body thresholds
    if char_count < _THIN_CHAR_THRESHOLD or word_count < _THIN_WORD_THRESHOLD:
        return {
            "quality": "empty",
            "char_count": char_count,
            "word_count": word_count,
            "blocked_reason": None,
            "status_code": code,
            "reason": "thin body",
        }

    if char_count < _MEDIUM_THIN_CHAR_THRESHOLD or word_count < _MEDIUM_THIN_WORD_THRESHOLD:
        return {
            "quality": "minimal",
            "char_count": char_count,
            "word_count": word_count,
            "blocked_reason": None,
            "status_code": code,
            "reason": "medium-thin body",
        }

    return {
        "quality": "sufficient",
        "char_count": char_count,
        "word_count": word_count,
        "blocked_reason": None,
        "status_code": code,
        "reason": "sufficient body",
    }


def _slug_from_url(url: str) -> str:
    """Turn a URL into a short filesystem-safe slug."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    path = re.sub(r"[^a-zA-Z0-9_.-]", "_", path)
    return path[:80]


def _save_to_cache(url: str, markdown: str, quality: Dict[str, Any]) -> Optional[str]:
    """
    Write crawled markdown to crawl_cache/{domain}/{slug}_{ts_ms}_{hash}.md.

    Returns the file path or None on failure.
    """
    try:
        domain = _extract_domain(url) or "unknown"
        domain_dir = os.path.join(CRAWL_CACHE_DIR, domain)
        os.makedirs(domain_dir, exist_ok=True)

        slug = _slug_from_url(url)
        ts = int(time.time())
        ts_ms = int(time.time() * 1000)
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
        filename = f"{slug}_{ts_ms}_{digest}.md"
        filepath = os.path.join(domain_dir, filename)

        # Write header metadata + content.
        header = (
            f"<!-- crawl_url: {url} -->\n"
            f"<!-- crawl_ts: {ts} -->\n"
            f"<!-- quality: {quality['quality']} -->\n"
            f"<!-- char_count: {quality['char_count']} -->\n"
            f"<!-- word_count: {quality['word_count']} -->\n\n"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + (markdown or ""))

        return filepath
    except Exception:
        return None


def _find_fuzzy_in_text(
    query: str,
    text: str,
    threshold: float = 0.6,
    context_lines: int = 3,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Zero-index fuzzy search across lines of text using difflib.

    Returns matches with context, similarity score, and line numbers.
    """
    lines = text.splitlines()
    query_lower = query.lower().strip()
    query_tokens = query_lower.split()
    results: List[Dict[str, Any]] = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        line_lower = line_stripped.lower()

        # Exact substring match gets score 1.0.
        if query_lower in line_lower:
            sim = 1.0
        else:
            # SequenceMatcher on full line.
            sim = difflib.SequenceMatcher(None, query_lower, line_lower).ratio()

            # Boost if all query tokens appear in the line.
            if sim < threshold and all(t in line_lower for t in query_tokens):
                sim = max(sim, threshold)

        if sim >= threshold:
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            context = "\n".join(lines[start:end])
            results.append({
                "line_num": i + 1,
                "similarity": round(sim, 4),
                "matched_line": line_stripped,
                "context": context,
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:max_results]


def _extract_markdown_payload(result: Dict[str, Any]) -> str:
    """
    Extract best-available crawl text from response payload.

    Preference order:
    1) markdown
    2) markdown_plain
    3) content
    """
    if not isinstance(result, dict):
        return ""
    return (
        result.get("markdown")
        or result.get("markdown_plain")
        or result.get("content")
        or ""
    )


def _auth_headers() -> Dict[str, str]:
    """Build optional Authorization header from local token config."""
    headers: Dict[str, str] = {}
    tok = _get_auth_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def _resolve_base_url(server_url: Optional[str] = None) -> str:
    """
    Determine which Wraith server URL to use (defaults to gnosis-crawl:8080).
    
    Args:
        server_url: Optional explicit server URL. Automatically normalized
                   (localhost/127.0.0.1 converted to gnosis-crawl:8080)
    
    Returns:
        str: The resolved base URL for API calls (gnosis-crawl:8080 by default)
    """
    # If explicit server_url provided, normalize it
    if server_url:
        return _normalize_server_url(server_url)
    
    # Default to local gnosis-crawl:8080
    return LOCAL_SERVER_URL



@mcp.tool()
async def set_auth_token(token: str, ctx: Context = None) -> Dict[str, Any]:
    """
    Save Wraith API authentication token to .wraithenv file (optional).
    
    Stores the token persistently so it doesn't need to be passed with each request.
    The token is saved in .wraithenv in the current working directory.
    
    Note: auth is not required for local gnosis-crawl:8080.
    
    Args:
        token: Wraith API authentication token to save
        ctx: MCP context (optional)
    
    Returns:
        Dict[str, Any]: Success status and file path where token was saved
    """

    if not token:
        return {"success": False, "error": "No token provided"}
    try:
        with open(WRAITH_ENV_FILE, "w", encoding="utf-8") as f:
            f.write(f"WRAITH_AUTH_TOKEN={token}\n")
        return {"success": True, "message": "Saved token to .wraithenv", "file": WRAITH_ENV_FILE}
    except Exception as e:
        return {"success": False, "error": f"Failed to save token: {e}"}


@mcp.tool()
async def crawl_status(server_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Check Wraith crawler configuration and connection status.
    
    Reports the server URL being used (defaults to gnosis-crawl:8080) and 
    whether an auth token is configured. Auth is optional for local server.
    
    Args:
        server_url: Optional explicit server URL to check (defaults to gnosis-crawl:8080)
    
    Returns:
        Dict[str, Any]: Server URL being used and token availability status
    """

    base = _resolve_base_url(server_url)
    return {
        "success": True,
        "base_url": base,
        "token_present": _get_auth_token() is not None,
        "auth_required": False,  # Auth not required for local gnosis-crawl:8080
    }


@mcp.tool()
async def crawl_url(
    url: str,
    take_screenshot: bool = False,
    javascript_enabled: bool = False,
    javascript_payload: Optional[str] = None,
    markdown_extraction: str = "enhanced",
    dedupe_tables: bool = True,
    server_url: Optional[str] = None,
    timeout: int = 30,
    title: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Crawl a single URL and extract clean markdown content.
    
    Fetches a web page through the Wraith API on gnosis-crawl:8080 (local default),
    which handles JavaScript rendering, content extraction, and markdown conversion.
    Returns structured markdown optimized for AI consumption.
    
    JavaScript injection: If javascript_payload is provided, it will be executed
    FIRST on the page, then markdown extraction will run on the modified content.
    
    Defaults to LOCAL gnosis-crawl:8080 with NO AUTH required.
    
    Args:
        url: Target URL to crawl
        take_screenshot: If True, capture a full-page screenshot
        javascript_enabled: If True, execute JavaScript before extracting content
        javascript_payload: Optional JavaScript code to inject and execute BEFORE
                          markdown extraction. Runs first, then markdown processes
                          the modified page content.
        markdown_extraction: Extraction mode ("enhanced" applies content pruning)
        server_url: Optional explicit server URL (defaults to gnosis-crawl:8080)
        timeout: Request timeout in seconds (minimum 5)
        title: Optional title for the crawl report (defaults to domain name)
        ctx: MCP context (optional)
    
    Returns:
        Dict[str, Any]: Crawl results including markdown content, metadata, and any errors
    """

    if not url:
        return {"success": False, "error": "No URL provided"}
    if _is_google_host(url):
        return {
            "success": False,
            "error": "Direct Google crawling is disabled—use the serpapi-search MCP tools instead.",
        }

    base = _resolve_base_url(server_url)
    endpoint = f"{base}/api/markdown"

    if not title:
        title = f"Crawl: {_extract_domain(url)}"

    payload: Dict[str, Any] = {
        "url": url,
        "javascript_enabled": bool(javascript_enabled),
        "screenshot_mode": "full" if take_screenshot else None,
        "options": {
            "timeout": int(timeout),
            "dedupe_tables": bool(dedupe_tables),
        },
    }
    # Inject JavaScript BEFORE markdown extraction
    if javascript_payload:
        payload["javascript_payload"] = javascript_payload
    
    if markdown_extraction == "enhanced":
        payload["filter"] = "pruning"
        payload["filter_options"] = {"threshold": 0.48, "min_words": 2}

    headers: Dict[str, str] = {}
    tok = _get_auth_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(5, int(timeout)))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"{resp.status}: {await resp.text()}"}

                result = await resp.json()

                # --- Content validation & cache ---
                md = _extract_markdown_payload(result)
                quality = _assess_content_quality(
                    md,
                    status_code=result.get("status_code"),
                    blocked=bool(
                        result.get("blocked")
                        or result.get("captcha_detected")
                        or result.get("challenge")
                        or result.get("is_blocked")
                    ),
                )
                result["content_quality"] = quality

                if quality["quality"] != "sufficient":
                    result["warning"] = "Page is thin/error/blocked. Do NOT fabricate information from this result."

                cached_path = _save_to_cache(url, md, quality)
                if cached_path:
                    result["cached_file"] = cached_path

                return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def crawl_batch(
    urls: List[str],
    javascript_enabled: bool = False,
    javascript_payload: Optional[str] = None,
    take_screenshot: bool = False,
    async_mode: bool = True,
    collate: bool = False,
    collate_title: Optional[str] = None,
    dedupe_tables: bool = True,
    server_url: Optional[str] = None,
    timeout: int = 60,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Crawl multiple URLs in a single batch operation.
    
    Processes multiple URLs through Wraith on gnosis-crawl:8080 (local default),
    with options for asynchronous processing and automatic collation into a 
    single markdown document. Max 50 URLs per batch.
    
    JavaScript injection: If javascript_payload is provided, it will be executed
    FIRST on each page, then markdown extraction will run on the modified content.
    
    Defaults to LOCAL gnosis-crawl:8080 with NO AUTH required.
    
    Args:
        urls: List of URLs to crawl (max 50)
        javascript_enabled: If True, execute JavaScript on each page
        javascript_payload: Optional JavaScript code to inject and execute BEFORE
                          markdown extraction on each URL. Runs first, then markdown
                          processes the modified page content.
        take_screenshot: If True, capture screenshots for each URL
        async_mode: If True, process URLs asynchronously (faster)
        collate: If True, combine all results into a single markdown document
        collate_title: Title for collated document (auto-generated if not provided)
        server_url: Optional explicit server URL (defaults to gnosis-crawl:8080)
        timeout: Request timeout in seconds (minimum 10)
        ctx: MCP context (optional)
    
    Returns:
        Dict[str, Any]: Batch crawl results, either individual or collated markdown
    """

    if not urls:
        return {"success": False, "error": "No URLs provided"}
    if len(urls) > 50:
        return {"success": False, "error": "Maximum 50 URLs allowed per batch"}
    for target in urls:
        if _is_google_host(target):
            return {
                "success": False,
                "error": "Direct Google crawling is disabled—use the serpapi-search MCP tools instead.",
            }

    base = _resolve_base_url(server_url)
    endpoint = f"{base}/api/markdown"

    payload: Dict[str, Any] = {
        "urls": urls,
        "javascript_enabled": bool(javascript_enabled),
        "screenshot_mode": "full" if take_screenshot else None,
        "async": bool(async_mode),
        "collate": bool(collate),
        "options": {
            "timeout": int(timeout),
            "dedupe_tables": bool(dedupe_tables),
        },
    }
    # Inject JavaScript BEFORE markdown extraction
    if javascript_payload:
        payload["javascript_payload"] = javascript_payload
    
    if collate:
        payload["collate_options"] = {
            "title": collate_title or f"Batch Crawl Results ({len(urls)} URLs)",
            "add_toc": True,
            "add_source_headers": True,
        }

    headers: Dict[str, str] = {}
    tok = _get_auth_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(10, int(timeout)))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status not in (200, 202):
                    return {"success": False, "error": f"{resp.status}: {await resp.text()}"}

                result = await resp.json()

                # --- Validate & cache each result in the batch ---
                items = result.get("results", [])
                if isinstance(items, list):
                    cached_files = []
                    for item in items:
                        md = _extract_markdown_payload(item)
                        item_url = item.get("url") or ""
                        quality = _assess_content_quality(
                            md,
                            status_code=item.get("status_code"),
                            blocked=bool(
                                item.get("blocked")
                                or item.get("captcha_detected")
                                or item.get("challenge")
                                or item.get("is_blocked")
                            ),
                        )
                        item["content_quality"] = quality
                        if quality["quality"] != "sufficient":
                            item["warning"] = "Page is thin/error/blocked. Do NOT fabricate information from this result."
                        path = _save_to_cache(item_url, md, quality)
                        if path:
                            item["cached_file"] = path
                            cached_files.append(path)
                    result["cached_files"] = cached_files

                # Handle collated single-document response.
                if collate and ("markdown" in result or "markdown_plain" in result):
                    md = _extract_markdown_payload(result)
                    quality = _assess_content_quality(
                        md,
                        status_code=result.get("status_code"),
                        blocked=bool(
                            result.get("blocked")
                            or result.get("captcha_detected")
                            or result.get("challenge")
                            or result.get("is_blocked")
                        ),
                    )
                    result["content_quality"] = quality
                    if quality["quality"] != "sufficient":
                        result["warning"] = "Page is thin/error/blocked. Do NOT fabricate information from this result."
                    path = _save_to_cache(urls[0], md, quality)
                    if path:
                        result["cached_file"] = path

                return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def raw_html(
    url: str,
    javascript_enabled: bool = True,
    javascript_payload: Optional[str] = None,
    server_url: Optional[str] = None,
    timeout: int = 30,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Fetch raw HTML from a URL without markdown conversion.
    
    Returns the raw HTML source from a web page via gnosis-crawl:8080 (local default),
    optionally with JavaScript execution. Useful when you need the actual HTML 
    structure rather than cleaned markdown content.
    
    Defaults to LOCAL gnosis-crawl:8080 with NO AUTH required.
    
    Args:
        url: Target URL to fetch
        javascript_enabled: If True, execute JavaScript before capturing HTML
        javascript_payload: Optional JavaScript code to execute on the page
        server_url: Optional explicit server URL (defaults to gnosis-crawl:8080)
        timeout: Request timeout in seconds (minimum 5)
        ctx: MCP context (optional)
    
    Returns:
        Dict[str, Any]: Raw HTML content and metadata
    """

    if not url:
        return {"success": False, "error": "No URL provided"}
    if _is_google_host(url):
        return {
            "success": False,
            "error": "Direct Google crawling is disabled—use the serpapi-search MCP tools instead.",
        }

    base = _resolve_base_url(server_url)
    endpoint = f"{base}/api/raw"

    payload: Dict[str, Any] = {
        "url": url,
        "javascript_enabled": bool(javascript_enabled),
        "options": {
            "timeout": int(timeout),
        },
    }
    if javascript_payload:
        payload["javascript_payload"] = javascript_payload

    headers: Dict[str, str] = {}
    tok = _get_auth_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(5, int(timeout)))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"success": False, "error": f"{resp.status}: {await resp.text()}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def download_file(
    url: str,
    output_path: Optional[str] = None,
    use_browser: bool = False,
    javascript_enabled: bool = True,
    timeout: int = 30,
    server_url: Optional[str] = None,
    filename: Optional[str] = None,
    save_in_service: bool = False,
    session_id: Optional[str] = None,
    download: bool = False,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Download a file (e.g., PDF) through gnosis-crawl and save it locally.

    Args:
        url: File URL to download
        output_path: Local path to write file (optional; defaults to ./downloads/<name>)
        use_browser: If True, use Playwright in the service to fetch the file
        javascript_enabled: Enable JS in browser mode
        timeout: Request timeout in seconds
        server_url: Optional explicit server URL (defaults to gnosis-crawl:8080)
        filename: Optional filename hint for the service
        save_in_service: If True, store file in service storage
        session_id: Required when save_in_service=True
        download: If True, service returns Content-Disposition attachment header
        ctx: MCP context (optional)

    Returns:
        Dict with local path, size, and content metadata
    """
    if not url:
        return {"success": False, "error": "No URL provided"}
    if save_in_service and not session_id:
        return {"success": False, "error": "session_id required when save_in_service is True"}

    base = _resolve_base_url(server_url)
    endpoint = f"{base}/download"

    params = {
        "url": url,
        "use_browser": str(bool(use_browser)).lower(),
        "javascript": str(bool(javascript_enabled)).lower(),
        "timeout": str(int(timeout)),
        "download": str(bool(download)).lower(),
    }
    if filename:
        params["filename"] = filename
    if save_in_service:
        params["save"] = "true"
        params["session_id"] = session_id

    headers: Dict[str, str] = {}
    tok = _get_auth_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(5, int(timeout)))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.get(endpoint, params=params, headers=headers) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"{resp.status}: {await resp.text()}"}

                content = await resp.read()
                content_type = resp.headers.get("Content-Type", "application/octet-stream")
                disposition = resp.headers.get("Content-Disposition", "")

                if output_path:
                    target_path = output_path
                else:
                    name = filename or _filename_from_content_disposition(disposition)
                    if not name:
                        name = Path(urlparse(url).path).name or "download"
                    name = _safe_filename(name)
                    downloads_dir = os.path.join(os.getcwd(), "downloads")
                    os.makedirs(downloads_dir, exist_ok=True)
                    target_path = os.path.join(downloads_dir, name)

                with open(target_path, "wb") as f:
                    f.write(content)

                return {
                    "success": True,
                    "url": url,
                    "output_path": target_path,
                    "size_bytes": len(content),
                    "content_type": content_type,
                    "content_disposition": disposition,
                    "saved_in_service": bool(save_in_service),
                }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def crawl_validate(
    text: str,
    status_code: Optional[int] = None,
    blocked: bool = False,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Validate content quality of crawled text before using it for analysis.

    Run this on any crawl result before trusting its content. Returns a quality
    classification so the agent knows whether the data is real or likely an empty
    SPA page, Cloudflare block, or 404.

    CRITICAL: If quality is "empty", "minimal", or "blocked", do NOT extract
    facts, pricing, or product information from this content. Report that the
    page was inaccessible instead.

    Args:
        text: The markdown or text content to validate (from a crawl result).
        status_code: Optional HTTP status code from crawler response.
        blocked: Optional blocked/challenge boolean from crawler response.
        ctx: MCP context (optional).

    Returns:
        Dict with quality classification, char/word counts, and guidance.
    """
    quality = _assess_content_quality(text, status_code=status_code, blocked=blocked)
    usable = quality["quality"] == "sufficient"
    if usable:
        guidance = "Content appears sufficient for analysis."
    elif quality["quality"] == "blocked":
        guidance = (
            f"Page appears blocked ({quality['blocked_reason']}). "
            "Do NOT extract any facts from this content."
        )
    elif quality["quality"] == "minimal":
        guidance = (
            f"Only {quality['word_count']} words of body text. "
            "Treat with extreme caution — verify any claims against other sources."
        )
    else:
        guidance = "Page is empty. Do NOT fabricate information."

    result = {
        "success": True,
        "usable": usable,
        "guidance": guidance,
        **quality,
    }
    if not usable:
        result["warning"] = "Page is thin/error/blocked. Do NOT fabricate information from this result."
    return result


@mcp.tool()
async def crawl_search(
    query: str,
    domain: Optional[str] = None,
    similarity_threshold: float = 0.6,
    max_results: int = 10,
    context_lines: int = 3,
    cache_dir: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Fuzzy search across locally cached crawl results — no indexing required.

    Searches all .md files in crawl_cache/ (or a specific domain subdirectory)
    using difflib SequenceMatcher. Returns matching passages with context and
    similarity scores. Use this to "skim" previously crawled content for
    specific information without re-crawling.

    Args:
        query: Text to search for (supports partial/fuzzy matching).
        domain: Optional domain to restrict search (e.g. "example.com").
                If not provided, searches all cached domains.
        similarity_threshold: Minimum similarity score 0.0-1.0 (default: 0.6).
        max_results: Maximum matches to return across all files (default: 10).
        context_lines: Lines of context around each match (default: 3).
        cache_dir: Override crawl cache directory (defaults to CRAWL_CACHE_DIR).
        ctx: MCP context (optional).

    Returns:
        Dict with matches including file path, line number, similarity,
        matched line, and surrounding context.
    """
    if not query or not query.strip():
        return {"success": False, "error": "query is required and cannot be empty"}

    threshold = max(0.0, min(float(similarity_threshold), 1.0))

    search_dir = cache_dir or CRAWL_CACHE_DIR
    if domain:
        search_dir = os.path.join(search_dir, domain)

    if not os.path.isdir(search_dir):
        return {
            "success": True,
            "query": query,
            "count": 0,
            "matches": [],
            "message": f"No crawl cache found at {search_dir}",
        }

    all_matches: List[Dict[str, Any]] = []

    for root, _dirs, files in os.walk(search_dir):
        for fname in files:
            if not fname.endswith(".md"):
                continue
            filepath = os.path.join(root, fname)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue

            hits = _find_fuzzy_in_text(
                query, text,
                threshold=threshold,
                context_lines=context_lines,
                max_results=max_results,
            )
            for hit in hits:
                # Extract URL from the cache file header if present.
                url_match = re.search(r"<!-- crawl_url: (.+?) -->", text[:500])
                hit["file"] = filepath
                hit["source_url"] = url_match.group(1) if url_match else None
                all_matches.append(hit)

    # Sort by similarity descending, take top N.
    all_matches.sort(key=lambda x: x["similarity"], reverse=True)
    all_matches = all_matches[:max_results]

    return {
        "success": True,
        "query": query,
        "domain": domain,
        "similarity_threshold": threshold,
        "count": len(all_matches),
        "matches": all_matches,
    }


@mcp.tool()
async def crawl_cache_list(
    domain: Optional[str] = None,
    max_results: int = 50,
    cache_dir: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    List cached crawl files, optionally filtered by domain.

    Use this to see what URLs have been crawled and cached locally,
    before running crawl_search.

    Args:
        domain: Optional domain to filter (e.g. "example.com").
        max_results: Maximum files to return (default: 50).
        cache_dir: Override crawl cache directory.
        ctx: MCP context (optional).

    Returns:
        Dict with list of cached files, their URLs, quality, and timestamps.
    """
    base_dir = cache_dir or CRAWL_CACHE_DIR
    if domain:
        base_dir = os.path.join(base_dir, domain)

    if not os.path.isdir(base_dir):
        return {"success": True, "count": 0, "files": [], "message": "No cache found."}

    entries: List[Dict[str, Any]] = []
    for root, _dirs, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".md"):
                continue
            filepath = os.path.join(root, fname)
            try:
                stat = os.stat(filepath)
                # Read first 500 chars for metadata.
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    header = f.read(500)

                url_m = re.search(r"<!-- crawl_url: (.+?) -->", header)
                ts_m = re.search(r"<!-- crawl_ts: (\d+) -->", header)
                q_m = re.search(r"<!-- quality: (\w+) -->", header)

                entries.append({
                    "file": filepath,
                    "url": url_m.group(1) if url_m else None,
                    "crawl_ts": int(ts_m.group(1)) if ts_m else None,
                    "quality": q_m.group(1) if q_m else None,
                    "size_bytes": stat.st_size,
                    "domain": os.path.basename(os.path.dirname(filepath)),
                })
            except Exception:
                continue

    entries.sort(key=lambda x: x.get("crawl_ts") or 0, reverse=True)
    entries = entries[:max_results]

    return {
        "success": True,
        "count": len(entries),
        "files": entries,
    }


@mcp.tool()
async def crawl_remote_search(
    query: str,
    domain: Optional[str] = None,
    url_prefix: Optional[str] = None,
    min_similarity: float = 0.6,
    max_results: int = 10,
    quality_in: Optional[List[str]] = None,
    since_ts: Optional[int] = None,
    server_url: Optional[str] = None,
    timeout: int = 30,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Fuzzy search remote crawl cache on the crawler service.

    This lets agents scan existing remote cache first ("search-before-crawl"),
    reducing duplicate crawls and improving grounding consistency.

    Endpoint:
      POST {base}/api/cache/search

    Args:
        query: Search query text (required).
        domain: Optional domain filter (e.g. "homelight.com").
        url_prefix: Optional URL prefix filter.
        min_similarity: Similarity threshold 0.0-1.0 (default 0.6).
        max_results: Maximum matches (default 10).
        quality_in: Optional quality filter list (e.g. ["sufficient"]).
        since_ts: Optional unix timestamp lower-bound.
        server_url: Optional crawler base URL override.
        timeout: HTTP timeout seconds.
        ctx: MCP context (optional).

    Returns:
        Remote search result payload, or actionable error if endpoint is missing.
    """
    if not query or not query.strip():
        return {"success": False, "error": "query is required and cannot be empty"}

    base = _resolve_base_url(server_url)
    endpoint = f"{base}/api/cache/search"
    threshold = max(0.0, min(float(min_similarity), 1.0))

    payload: Dict[str, Any] = {
        "query": query.strip(),
        "min_similarity": threshold,
        "max_results": max(1, int(max_results)),
    }
    if domain:
        payload["domain"] = domain
    if url_prefix:
        payload["url_prefix"] = url_prefix
    if quality_in:
        payload["quality_in"] = quality_in
    if since_ts is not None:
        payload["since_ts"] = int(since_ts)

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(5, int(timeout)))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.post(endpoint, json=payload, headers=_auth_headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, dict):
                        data.setdefault("success", True)
                        data.setdefault("query", query)
                        data.setdefault("min_similarity", threshold)
                    return data
                text = await resp.text()
                if resp.status == 404:
                    return {
                        "success": False,
                        "error": "Remote cache search endpoint not available",
                        "status": resp.status,
                        "endpoint": endpoint,
                        "next_step": "Upgrade crawler API to expose POST /api/cache/search",
                        "details": text,
                    }
                return {"success": False, "error": f"{resp.status}: {text}", "status": resp.status, "endpoint": endpoint}
    except Exception as e:
        return {"success": False, "error": str(e), "endpoint": endpoint}


@mcp.tool()
async def crawl_remote_cache_list(
    domain: Optional[str] = None,
    quality: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    server_url: Optional[str] = None,
    timeout: int = 30,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    List remote crawl cache documents from crawler service.

    Endpoint:
      GET {base}/api/cache/list

    Args:
        domain: Optional domain filter.
        quality: Optional quality filter ("sufficient", etc.).
        limit: Max rows (default 50).
        offset: Pagination offset.
        server_url: Optional crawler base URL override.
        timeout: HTTP timeout seconds.
        ctx: MCP context (optional).

    Returns:
        Remote cache listing or actionable endpoint guidance.
    """
    base = _resolve_base_url(server_url)
    endpoint = f"{base}/api/cache/list"
    params: Dict[str, Any] = {
        "limit": max(1, int(limit)),
        "offset": max(0, int(offset)),
    }
    if domain:
        params["domain"] = domain
    if quality:
        params["quality"] = quality

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(5, int(timeout)))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.get(endpoint, params=params, headers=_auth_headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, dict):
                        data.setdefault("success", True)
                    return data
                text = await resp.text()
                if resp.status == 404:
                    return {
                        "success": False,
                        "error": "Remote cache list endpoint not available",
                        "status": resp.status,
                        "endpoint": endpoint,
                        "next_step": "Upgrade crawler API to expose GET /api/cache/list",
                        "details": text,
                    }
                return {"success": False, "error": f"{resp.status}: {text}", "status": resp.status, "endpoint": endpoint}
    except Exception as e:
        return {"success": False, "error": str(e), "endpoint": endpoint}


@mcp.tool()
async def crawl_remote_cache_doc(
    doc_id: str,
    server_url: Optional[str] = None,
    timeout: int = 30,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Fetch one remote cached crawl document by id.

    Endpoint:
      GET {base}/api/cache/doc/{doc_id}

    Args:
        doc_id: Remote cache document id.
        server_url: Optional crawler base URL override.
        timeout: HTTP timeout seconds.
        ctx: MCP context (optional).

    Returns:
        Cached document payload or endpoint guidance if unavailable.
    """
    if not doc_id or not doc_id.strip():
        return {"success": False, "error": "doc_id is required"}

    encoded_id = quote(doc_id.strip(), safe="")
    base = _resolve_base_url(server_url)
    endpoint = f"{base}/api/cache/doc/{encoded_id}"

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(5, int(timeout)))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.get(endpoint, headers=_auth_headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, dict):
                        data.setdefault("success", True)
                        data.setdefault("doc_id", doc_id)
                    return data
                text = await resp.text()
                if resp.status == 404:
                    return {
                        "success": False,
                        "error": "Remote cache doc endpoint unavailable or document not found",
                        "status": resp.status,
                        "endpoint": endpoint,
                        "next_step": "Verify doc_id exists and crawler API exposes GET /api/cache/doc/{doc_id}",
                        "details": text,
                    }
                return {"success": False, "error": f"{resp.status}: {text}", "status": resp.status, "endpoint": endpoint}
    except Exception as e:
        return {"success": False, "error": str(e), "endpoint": endpoint}


@mcp.tool()
async def agent_run(
    task: str,
    allowed_domains: Optional[List[str]] = None,
    max_steps: int = 12,
    timeout: int = 90,
    server_url: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Submit a multi-step task to the autonomous crawl agent (Mode B).

    The agent plans its own tool calls, executes them under policy gates,
    and returns a final answer with a full replayable trace. Requires
    AGENT_ENABLED=true on the server.

    Use this when a task requires visiting multiple pages, following links,
    or reasoning across crawled content — anything beyond a single fetch.

    Args:
        task: Natural language description of what to accomplish
              (e.g. "Find the pricing page on example.com and extract plan details")
        allowed_domains: Optional domain allowlist for this run
                        (e.g. ["example.com", "docs.example.com"])
        max_steps: Maximum agent loop iterations (default: 12, max: 50)
        timeout: Wall-clock timeout in seconds (default: 90, max: 300)
        server_url: Optional explicit server URL (defaults to gnosis-crawl:8080)
        ctx: MCP context (optional)

    Returns:
        Dict with agent response, trace, artifacts, and stop reason
    """
    if not task or not task.strip():
        return {"success": False, "error": "task is required"}

    base = _resolve_base_url(server_url)
    endpoint = f"{base}/api/agent/run"

    payload: Dict[str, Any] = {
        "task": task.strip(),
        "max_steps": min(max(1, int(max_steps)), 50),
        "max_wall_time_ms": min(max(5000, int(timeout) * 1000), 300_000),
    }
    if allowed_domains:
        payload["allowed_domains"] = allowed_domains

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(10, int(timeout) + 10))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.post(endpoint, json=payload, headers=_auth_headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, dict):
                        data.setdefault("success", True)
                    return data
                text = await resp.text()
                if resp.status == 404:
                    return {
                        "success": False,
                        "error": "Agent endpoint not available — ensure AGENT_ENABLED=true on the server",
                        "status": resp.status,
                        "endpoint": endpoint,
                        "details": text,
                    }
                if resp.status == 503:
                    return {
                        "success": False,
                        "error": "Agent is disabled on this server. Set AGENT_ENABLED=true to enable Mode B.",
                        "status": resp.status,
                    }
                return {"success": False, "error": f"{resp.status}: {text}", "status": resp.status}
    except Exception as e:
        return {"success": False, "error": str(e), "endpoint": endpoint}


@mcp.tool()
async def agent_status(
    run_id: str,
    server_url: Optional[str] = None,
    timeout: int = 15,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Check the status of a running or completed agent task.

    Args:
        run_id: The run_id returned by agent_run
        server_url: Optional explicit server URL (defaults to gnosis-crawl:8080)
        timeout: HTTP timeout seconds
        ctx: MCP context (optional)

    Returns:
        Dict with run status, progress, partial results, and trace
    """
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id is required"}

    base = _resolve_base_url(server_url)
    endpoint = f"{base}/api/agent/status/{quote(run_id.strip(), safe='')}"

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(5, int(timeout)))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.get(endpoint, headers=_auth_headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, dict):
                        data.setdefault("success", True)
                    return data
                text = await resp.text()
                if resp.status == 404:
                    return {
                        "success": False,
                        "error": f"Run '{run_id}' not found or agent endpoint unavailable",
                        "status": resp.status,
                    }
                return {"success": False, "error": f"{resp.status}: {text}", "status": resp.status}
    except Exception as e:
        return {"success": False, "error": str(e), "endpoint": endpoint}


@mcp.tool()
async def ghost_extract(
    url: str,
    server_url: Optional[str] = None,
    timeout: int = 60,
    prompt: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Ghost Protocol: screenshot a URL and extract content via vision AI.

    Bypasses DOM-based anti-bot detection (Cloudflare, CAPTCHAs, challenge pages)
    by capturing a screenshot of the rendered page and extracting text content
    from the pixels using Claude or GPT-4o vision.

    Use this when a normal crawl returns blocked/empty content.

    Args:
        url: The URL to ghost-extract
        server_url: Optional explicit server URL (defaults to gnosis-crawl:8080)
        timeout: HTTP timeout seconds (default 60 — ghost is slower than normal crawl)
        prompt: Optional custom vision extraction prompt
        ctx: MCP context (optional)

    Returns:
        Dict with extracted content, render_mode="ghost", timing metadata
    """
    if not url or not url.strip():
        return {"success": False, "error": "url is required"}

    base = _resolve_base_url(server_url)
    endpoint = f"{base}/api/agent/ghost"

    payload: Dict[str, Any] = {"url": url.strip(), "timeout": min(timeout, 120)}
    if prompt:
        payload["prompt"] = prompt

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(10, int(timeout) + 10))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.post(endpoint, json=payload, headers=_auth_headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, dict):
                        data.setdefault("success", True)
                    return data
                text = await resp.text()
                if resp.status == 503:
                    return {
                        "success": False,
                        "error": "Ghost Protocol is disabled on the server. Set AGENT_GHOST_ENABLED=true.",
                        "status": resp.status,
                    }
                return {"success": False, "error": f"{resp.status}: {text}", "status": resp.status}
    except Exception as e:
        return {"success": False, "error": str(e), "endpoint": endpoint}


@mcp.tool()
async def mesh_peers(
    server_url: Optional[str] = None,
    timeout: int = 15,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    List known mesh peers and their health status.

    Returns this node's identity plus all peers it knows about, including
    health status, load metrics, and capabilities. Use this to understand
    the mesh topology and which nodes are available.

    Requires MESH_ENABLED=true on the crawler.

    Endpoint:
      GET {base}/mesh/peers

    Args:
        server_url: Optional crawler base URL override.
        timeout: HTTP timeout seconds.
        ctx: MCP context (optional).

    Returns:
        Dict with node_id, node_name, peer_count, and peers list.
        Each peer includes node_id, node_name, advertise_url, tools,
        capabilities, healthy, missed_heartbeats, load metrics.
    """
    base = _resolve_base_url(server_url)
    endpoint = f"{base}/mesh/peers"

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(5, int(timeout)))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.get(endpoint, headers=_auth_headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    data.setdefault("success", True)
                    return data
                text = await resp.text()
                if resp.status == 503:
                    return {
                        "success": False,
                        "error": "Mesh is not enabled on the server. Set MESH_ENABLED=true.",
                        "status": resp.status,
                    }
                return {"success": False, "error": f"{resp.status}: {text}", "status": resp.status}
    except Exception as e:
        return {"success": False, "error": str(e), "endpoint": endpoint}


@mcp.tool()
async def mesh_status(
    server_url: Optional[str] = None,
    timeout: int = 15,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Get this node's mesh status including load metrics.

    Returns the node's identity, advertise URL, capabilities, current load
    (active crawls, agent runs, browser pool), and peer counts. Use this
    to check if a node is healthy and how busy it is.

    Requires MESH_ENABLED=true on the crawler.

    Endpoint:
      GET {base}/mesh/status

    Args:
        server_url: Optional crawler base URL override.
        timeout: HTTP timeout seconds.
        ctx: MCP context (optional).

    Returns:
        Dict with node_id, node_name, advertise_url, tools, capabilities,
        load (active_crawls, active_agent_runs, browser_pool_free,
        max_concurrent_crawls), total_peers, healthy_peers.
    """
    base = _resolve_base_url(server_url)
    endpoint = f"{base}/mesh/status"

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(5, int(timeout)))
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.get(endpoint, headers=_auth_headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    data.setdefault("success", True)
                    return data
                text = await resp.text()
                if resp.status == 503:
                    return {
                        "success": False,
                        "error": "Mesh is not enabled on the server. Set MESH_ENABLED=true.",
                        "status": resp.status,
                    }
                return {"success": False, "error": f"{resp.status}: {text}", "status": resp.status}
    except Exception as e:
        return {"success": False, "error": str(e), "endpoint": endpoint}


if __name__ == "__main__":
    mcp.run(transport="stdio")
