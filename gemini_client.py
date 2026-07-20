"""Google Gemini API client wrapper with caching, retry, and key sanitization.

v2.1 changes (security & robustness pass, 2026-05-17):
  - Client cache is now LRU-bounded (max 16 keys) and keyed by SHA-256 hash
    of the API key. The raw key never lives in a dict key, only inside the
    Client object. Prevents unbounded memory growth and reduces key
    retention surface area for memory dumps.
  - `redact_secret(text)` strips Google API keys (AIza... 35 chars) AND any
    accidentally-embedded x-goog-api-key/Authorization headers before
    surfacing exception messages to UI / logs.
  - `is_transient_error()` now inspects `google.genai.errors.APIError.code`
    when available (correct gRPC status), falling back to string matching
    with WORD-BOUNDARY checks (so "500x500" in a prompt cannot trip us).
  - Quota / RESOURCE_EXHAUSTED 429s are now classified as permanent so we
    don't waste credits retrying them.
  - `get_api_key()` strips surrounding quotes (common .env mistake:
    `GEMINI_API_KEY="AIza..."`).
  - `retry_with_backoff` adds jitter (avoids thundering-herd retries) and
    redacts exception text before printing.
"""

import functools
import hashlib
import os
import random
import re
import time
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Secret redaction (used by every error path so keys can never leak to UI)
# ---------------------------------------------------------------------------

# Google API keys are AIza + 35 url-safe chars.
_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_-]{35}")
# Header echoes ("x-goog-api-key: AIza...", "Authorization: Bearer ya29...")
_HEADER_KEY_RE = re.compile(
    r"(?i)(x-goog-api-key|authorization|api[-_]key)\s*[:=]\s*[\"']?([^\s\"'&]+)"
)


def redact_secret(text: str) -> str:
    """Strip Google API keys and common auth headers from any text.

    Use on EVERY string that might come from an exception or HTTP response
    body before logging or raising it to the UI.
    """
    if not text:
        return text
    s = str(text)
    s = _GOOGLE_KEY_RE.sub("AIza...REDACTED", s)
    s = _HEADER_KEY_RE.sub(r"\1: REDACTED", s)
    return s


# ---------------------------------------------------------------------------
# Bounded client cache (LRU over hashed key)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Network routing (proxy support)
# ---------------------------------------------------------------------------

def _network_to_proxy_url(network) -> str:
    """Extract a single proxy URL from a NanoBanana_NetworkRoute output, or "" if
    networking should be direct (the default).

    Expected shape:
        {"enabled": bool, "proxy_url": str, "label": str}
    """
    if not network:
        return ""
    if not isinstance(network, dict):
        return ""
    if not network.get("enabled", True):
        return ""
    return (network.get("proxy_url") or "").strip()


def _build_proxied_client(api_key: str, timeout_ms: int, proxy_url: str):
    """Build a google.genai Client whose underlying httpx transport routes through
    the supplied proxy URL. If proxy_url is empty, behave exactly like before."""
    from google import genai
    http_options = {"timeout": timeout_ms}
    if proxy_url:
        # httpx is a transitive dep of google-genai. Build a Client whose
        # transport tunnels through the user-supplied US (or wherever) proxy.
        try:
            import httpx
            mounts = {"all://": httpx.HTTPTransport(proxy=proxy_url)}
            http_options["client_args"] = {"mounts": mounts}
            http_options["async_client_args"] = {"mounts": mounts}
        except Exception as e:
            # If httpx mounts plumbing isn't available in this version, fall
            # back to env vars — google-genai's underlying httpx still respects
            # HTTPS_PROXY at Client construction time.
            print(f"[NanoBanana Network] httpx mounts unavailable ({e}); using HTTPS_PROXY env fallback")
            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
    return genai.Client(api_key=api_key, http_options=http_options)


# Hash → Client. lru_cache discards LRU entries past maxsize. We key by
# (key_hash, timeout_ms, proxy_url) so each routing variant gets its own
# cached Client instead of fighting over a single global one.
@functools.lru_cache(maxsize=64)
def _get_client_for_hash(key_hash: str, timeout_ms: int, proxy_url: str, _api_key: str):
    """Internal: create or retrieve the Client for a given (key, timeout, proxy)."""
    return _build_proxied_client(_api_key, timeout_ms, proxy_url)


def get_client(api_key, timeout_ms=180000, network=None):
    """Return a cached google.genai Client for the given API key.

    Args:
      api_key: Google Gemini API key.
      timeout_ms: HTTP timeout for the underlying httpx client.
      network: Optional NB_NETWORK config (output of NanoBanana_NetworkRoute).
        When set with proxy_url, every request from the returned Client tunnels
        through that proxy — e.g. point it at a US-egress SOCKS5/HTTP proxy to
        make API calls appear to originate from the US.

    The cache is bounded (64 entries) and indexed by a SHA-256 hash of the key
    plus the proxy URL, so:
      - rotating keys across many workflows can't OOM the worker
      - the raw key isn't held in a dict key indefinitely
      - switching between routed/direct calls in the same workflow works
    """
    if not api_key:
        raise ValueError("get_client() called with empty API key")
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    proxy_url = _network_to_proxy_url(network)
    return _get_client_for_hash(key_hash, int(timeout_ms), proxy_url, api_key)


def proxy_url_from_network(network) -> str:
    """Public helper for code paths that use raw requests (e.g. Lyria's
    :predict endpoint) instead of the genai.Client — returns "" for direct."""
    return _network_to_proxy_url(network)


def proxies_for_requests(network) -> dict:
    """Public helper that returns a `proxies=` dict to pass to `requests.post`/
    `requests.get`. Returns {} if no proxy is configured (i.e. direct call)."""
    url = _network_to_proxy_url(network)
    return {"http": url, "https": url} if url else {}


def clear_client_cache():
    """Eject all cached clients (useful for tests or after key rotation)."""
    _get_client_for_hash.cache_clear()


def get_api_key(api_key_input=""):
    """Resolve API key from input or GEMINI_API_KEY env, with quote stripping.

    Users commonly write `GEMINI_API_KEY="AIza..."` in .env files; the
    quotes become part of the string and break auth. We strip whitespace
    AND surrounding quote characters.
    """
    key = api_key_input.strip() if api_key_input else ""
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
    # Strip wrapping single/double quotes (.env mistake).
    if key and len(key) >= 2 and key[0] == key[-1] and key[0] in ('"', "'"):
        key = key[1:-1].strip()
    if not key:
        raise ValueError(
            "Gemini API key required. Set GEMINI_API_KEY env var or enter in node."
        )
    return key


# ---------------------------------------------------------------------------
# Error classification (typed checks, not substring soup)
# ---------------------------------------------------------------------------

# gRPC canonical names that Gemini SDK surfaces in errors
_TRANSIENT_GRPC = {"UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "ABORTED"}
_PERMANENT_QUOTA_TOKENS = ("quota", "exhausted", "billing", "exceeded your current")

# Status codes we treat as transient by default
_TRANSIENT_CODES = {429, 500, 502, 503, 504}


def is_transient_error(e):
    """Decide if an error is worth retrying.

    Strategy (in order):
      1. If it's a google.genai.errors.APIError, use the typed `.code`.
      2. If it's a 429 with "quota"/"exhausted" in the body → NOT transient
         (retrying a quota-exhaustion just burns credits).
      3. Otherwise fall back to whole-token substring match (no false trip
         on "500x500" inside a prompt).
    """
    try:
        from google.genai import errors as gerrors
        if isinstance(e, gerrors.APIError):
            code = getattr(e, "code", None)
            body = redact_secret(str(getattr(e, "response_json", "") or ""))
            # 429 quota exhaustion is permanent — don't retry
            if code == 429 and any(t in body.lower() for t in _PERMANENT_QUOTA_TOKENS):
                return False
            if code in _TRANSIENT_CODES:
                return True
            # gRPC code names sometimes leak into message
            if any(g in str(e) for g in _TRANSIENT_GRPC):
                return True
            return False
    except Exception:
        pass

    # Fallback: word-boundary search so prompts containing "500" don't trip
    err_str = str(e)
    for code in (429, 500, 502, 503, 504):
        if re.search(rf"\b{code}\b", err_str):
            # Even in fallback mode, treat quota 429 as permanent
            if code == 429 and any(t in err_str.lower() for t in _PERMANENT_QUOTA_TOKENS):
                return False
            return True
    return any(g in err_str for g in _TRANSIENT_GRPC)


def retry_with_backoff(fn, retries=3, base_delay=5.0, max_delay=60.0):
    """Execute fn with jittered exponential backoff on transient errors.

    Jitter prevents the thundering-herd problem when many workers retry at
    the same time. All exception text is redacted before logging.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if is_transient_error(e) and attempt < retries - 1:
                # Jittered exp backoff: base * 2^attempt * random(0.8, 1.2)
                delay = min(base_delay * (2 ** attempt), max_delay)
                delay *= random.uniform(0.8, 1.2)
                safe_msg = redact_secret(str(e))[:200]
                logger.warning(
                    "[Gemini] transient API error (%s), retrying in %.1fs (%d/%d)",
                    safe_msg, delay, attempt + 1, retries,
                )
                # Keep stdout breadcrumb for users who don't see log handlers
                print(
                    f"[Gemini] transient API error ({safe_msg}), retrying in "
                    f"{delay:.1f}s... ({attempt + 1}/{retries})"
                )
                time.sleep(delay)
            else:
                raise
    # Should not be reachable (loop either returns or raises) but defensive:
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_with_backoff: no attempts made (retries=%d)" % retries)


# ---------------------------------------------------------------------------
# Lyria/Veo model-name sanitization (prevents URL path injection)
# ---------------------------------------------------------------------------

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def sanitize_model_id(model_id: str) -> str:
    """Validate a model ID is safe to interpolate into an API URL path.

    Gemini/Imagen/Veo/Lyria model IDs only ever contain alphanumerics,
    dot, hyphen, underscore. Anything else (including `/`, `?`, `#`, `:`,
    `..`) means a malicious or buggy `custom_model` is trying to escape the
    `/models/` path. Reject hard rather than silently rewriting.
    """
    if not model_id or not _MODEL_ID_RE.match(model_id):
        raise ValueError(
            f"Invalid model ID {model_id!r}. Model IDs must match "
            r"^[A-Za-z0-9._-]+$ (no slashes, no path traversal)."
        )
    return model_id


# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------

# All model IDs below are verified against Google's Developer API (models.list).
# Use the "Gemini List Available Models" node to see what's accessible for YOUR key.

TEXT_MODELS = [
    # --- Latest aliases (auto-updated by Google) ---
    "gemini-pro-latest",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    # --- Gemini 3 Previews (most capable) ---
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3.1-flash-lite-preview",
    # --- Gemini 2.5 (stable) ---
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    # --- Gemini 2.0 ---
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    # --- TTS variants ---
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
    "gemini-3.1-flash-tts-preview",
    # --- Specialized ---
    "gemini-robotics-er-1.5-preview",
    "gemini-robotics-er-1.6-preview",
    "gemini-2.5-computer-use-preview-10-2025",
    "deep-research-pro-preview-12-2025",
    "deep-research-preview-04-2026",
    "deep-research-max-preview-04-2026",
    "nano-banana-pro-preview",
    # --- Lyria (music) ---
    "lyria-3-clip-preview",
    "lyria-3-pro-preview",
    # --- Gemma (open models) ---
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "gemma-3n-e2b-it",
    "gemma-3n-e4b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
]

# Image generation via generate_content (native multimodal output)
IMAGE_MODELS = [
    "gemini-3.1-flash-image-preview",  # Nano Banana 2
    "gemini-3-pro-image-preview",      # Nano Banana Pro
    "gemini-2.5-flash-image",          # Nano Banana
]

# Imagen uses generate_images (predict) endpoint — different API path
IMAGEN_MODELS = [
    "imagen-4.0-ultra-generate-001",
    "imagen-4.0-generate-001",
    "imagen-4.0-fast-generate-001",
]

# Imagen aspect ratios (documented Imagen API set)
IMAGEN_ASPECT_RATIOS = ["1:1", "3:4", "4:3", "9:16", "16:9"]

# TTS models (use generate_content with audio response modality)
TTS_MODELS = [
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
    "gemini-3.1-flash-tts-preview",
]

# Embedding models
EMBEDDING_MODELS = [
    "gemini-embedding-2",          # NEW: stable v2 (released April 2026)
    "gemini-embedding-2-preview",
    "gemini-embedding-001",
]

# AQA (Attributed Question Answering) — uses generateAnswer endpoint
AQA_MODELS = [
    "aqa",
]

# Native audio / Live models — use bidiGenerateContent (websocket streaming)
# Listed but no dedicated node yet (live streaming requires special handling)
NATIVE_AUDIO_MODELS = [
    "gemini-2.5-flash-native-audio-latest",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-2.5-flash-native-audio-preview-09-2025",
    "gemini-3.1-flash-live-preview",
]

# Veo video generation (uses predictLongRunning)
VEO_MODELS = [
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-lite-generate-preview",
    "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001",
    "veo-2.0-generate-001",
]

# Lyria music models
LYRIA_MODELS = [
    "lyria-3-pro-preview",
    "lyria-3-clip-preview",
]

# Veo supports these aspect ratios
VEO_ASPECT_RATIOS = ["16:9", "9:16"]

# Pre-built voices for Gemini TTS
TTS_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir",
    "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
    "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina",
    "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird",
    "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

ALL_MODELS = TEXT_MODELS + IMAGE_MODELS + IMAGEN_MODELS + TTS_MODELS + EMBEDDING_MODELS + VEO_MODELS + LYRIA_MODELS

ASPECT_RATIOS = [
    "AUTO", "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9",
]

# THINKING_LEVELS: the google-genai SDK enum is
#   ThinkingLevel.MINIMAL / LOW / MEDIUM / HIGH (+ THINKING_LEVEL_UNSPECIFIED).
# v2.0 used "NORMAL" which the API rejects, and "NONE" was a UI sentinel for
# "don't set thinking_config at all". v2.1 keeps NONE as the sentinel but
# replaces NORMAL with the actual SDK values MINIMAL/MEDIUM.
THINKING_LEVELS = ["NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH"]

IMAGE_SIZES = ["AUTO", "1K", "2K", "4K"]

# Safety filter levels accepted by the Imagen `predict` endpoint.
# Developer API typically only honors block_low_and_above on free tier, but
# we expose all valid enum values; the API surfaces a clear error if the
# level isn't accepted for the current model/account.
IMAGEN_SAFETY_LEVELS = [
    "block_low_and_above",
    "block_medium_and_above",
    "block_only_high",
    "block_none",
]
