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
    "gemini-embedding-001",
    "gemini-embedding-2-preview",
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

THINKING_LEVELS = ["NONE", "LOW", "NORMAL", "HIGH"]

IMAGE_SIZES = ["AUTO", "1K", "2K", "4K"]
