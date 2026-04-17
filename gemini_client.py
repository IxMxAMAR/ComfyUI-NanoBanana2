"""Google Gemini API client wrapper with caching and retry."""

import os
import time
import logging

logger = logging.getLogger(__name__)

_client_cache = {}


def get_client(api_key, timeout_ms=180000):
    """Return a cached google.genai Client for the given API key."""
    from google import genai

    if api_key not in _client_cache:
        _client_cache[api_key] = genai.Client(
            api_key=api_key, http_options={"timeout": timeout_ms}
        )
    return _client_cache[api_key]


def get_api_key(api_key_input=""):
    """Resolve API key from input or GEMINI_API_KEY environment variable."""
    key = api_key_input.strip() if api_key_input else ""
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "Gemini API key required. Set GEMINI_API_KEY env var or enter in node."
        )
    return key


def is_transient_error(e):
    """Check if an error is transient and worth retrying."""
    err_str = str(e)
    return any(
        m in err_str
        for m in ("429", "500", "502", "503", "504", "DEADLINE_EXCEEDED")
    )


def retry_with_backoff(fn, retries=3, base_delay=5.0):
    """Execute fn with exponential backoff on transient errors."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if is_transient_error(e) and attempt < retries - 1:
                delay = base_delay * (2 ** attempt)
                print(
                    f"[Gemini] API error ({e}), retrying in {delay:.0f}s... "
                    f"({attempt + 1}/{retries})"
                )
                time.sleep(delay)
            else:
                raise


# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------

TEXT_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

IMAGE_MODELS = [
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
]

ALL_MODELS = TEXT_MODELS + IMAGE_MODELS

ASPECT_RATIOS = [
    "AUTO", "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9",
]

THINKING_LEVELS = ["NONE", "LOW", "NORMAL", "HIGH"]

IMAGE_SIZES = ["AUTO", "1K", "2K", "4K"]
