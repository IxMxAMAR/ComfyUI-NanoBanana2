"""ComfyUI nodes for Google Gemini API.

v2.1 — 19 original nodes + 8 new feature nodes (see extra_nodes.py).
"""

import json
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared imports  (google-genai imported lazily inside methods)
# ---------------------------------------------------------------------------

try:
    from .shared.node_utils import AlwaysExecuteMixin, OptionalRerunMixin
    from .shared.auth import BaseAPIKeyNode
    from .shared.conversions import (
        tensor_to_jpeg_bytes, tensor_to_png_bytes,
        mask_to_jpeg_bytes, mask_to_png_bytes, resize_mask_to_image,
        bytes_to_tensor,
    )
    from .gemini_client import (
        get_client, get_api_key, retry_with_backoff,
        redact_secret, sanitize_model_id,
        TEXT_MODELS, IMAGE_MODELS, IMAGEN_MODELS, IMAGEN_ASPECT_RATIOS,
        IMAGEN_SAFETY_LEVELS, TTS_MODELS, TTS_VOICES, EMBEDDING_MODELS,
        VEO_MODELS, VEO_ASPECT_RATIOS, LYRIA_MODELS, ALL_MODELS,
        ASPECT_RATIOS, THINKING_LEVELS, IMAGE_SIZES,
    )
except ImportError:
    # Fallback for direct testing / flat layouts
    from shared.node_utils import AlwaysExecuteMixin, OptionalRerunMixin
    from shared.auth import BaseAPIKeyNode
    from shared.conversions import (
        tensor_to_jpeg_bytes, tensor_to_png_bytes,
        mask_to_jpeg_bytes, mask_to_png_bytes, resize_mask_to_image,
        bytes_to_tensor,
    )
    from gemini_client import (
        get_client, get_api_key, retry_with_backoff,
        redact_secret, sanitize_model_id,
        TEXT_MODELS, IMAGE_MODELS, IMAGEN_MODELS, IMAGEN_ASPECT_RATIOS,
        IMAGEN_SAFETY_LEVELS, TTS_MODELS, TTS_VOICES, EMBEDDING_MODELS,
        VEO_MODELS, VEO_ASPECT_RATIOS, LYRIA_MODELS, ALL_MODELS,
        ASPECT_RATIOS, THINKING_LEVELS, IMAGE_SIZES,
    )


# ===================================================================
# Helper functions
# ===================================================================

def _resolve_model(model, custom_model):
    """Return custom_model if non-empty, else model."""
    cm = custom_model.strip() if custom_model else ""
    return cm if cm else model


def _build_image_parts(ref_images, labels=True, lossless=False):
    """Convert a list of optional image tensors into genai Part objects.

    Args:
        ref_images: list of (tensor_or_None) image inputs.
        labels: whether to prepend text labels.
        lossless: use PNG (no quality loss) instead of JPEG. Recommended
            for vision/OCR tasks where small text matters.

    Returns:
        (parts_list, ref_count) - list of genai Part objects and count
        of images added (ref_count starts at 1 and is the next available
        index, so a return of 1 means zero images were added).
    """
    from google.genai import types

    parts = []
    ref_count = 1
    for tensor_batch in ref_images:
        if tensor_batch is None:
            continue
        for i in range(tensor_batch.shape[0]):
            single = tensor_batch[i : i + 1]
            try:
                if lossless:
                    img_bytes = tensor_to_png_bytes(single)
                    mime = "image/png"
                else:
                    img_bytes = tensor_to_jpeg_bytes(single, quality=95)
                    mime = "image/jpeg"
            except Exception as e:
                raise ValueError(
                    f"Failed to convert reference image {ref_count}: {e}"
                )
            if labels:
                parts.append(
                    types.Part.from_text(text=f"--- [Reference Image {ref_count}] ---")
                )
            parts.append(
                types.Part.from_bytes(data=img_bytes, mime_type=mime)
            )
            ref_count += 1
    return parts, ref_count


def _build_config(
    *,
    modalities=None,
    system_instruction=None,
    thinking_level="NONE",
    thinking_budget=None,
    temperature=None,
    top_p=None,
    top_k=None,
    max_output_tokens=None,
    seed=None,
    aspect_ratio=None,
    image_size=None,
    candidate_count=None,
    safety_settings_json=None,
    response_schema=None,
    response_mime_type=None,
):
    """Build a GenerateContentConfig from keyword arguments."""
    from google.genai import types

    kwargs = {}

    # System instruction
    if system_instruction and system_instruction.strip():
        kwargs["system_instruction"] = [
            types.Part.from_text(text=system_instruction.strip())
        ]

    # Thinking
    if thinking_level and thinking_level != "NONE":
        tc_kwargs = {"thinking_level": thinking_level}
        if thinking_budget and thinking_budget > 0:
            tc_kwargs["thinking_budget"] = thinking_budget
        kwargs["thinking_config"] = types.ThinkingConfig(**tc_kwargs)

    # Image config
    img_kwargs = {}
    if aspect_ratio and aspect_ratio != "AUTO":
        img_kwargs["aspect_ratio"] = aspect_ratio
    if image_size and image_size != "AUTO":
        img_kwargs["image_size"] = image_size
    if img_kwargs:
        kwargs["image_config"] = types.ImageConfig(**img_kwargs)

    # Generation params
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None and top_k > 0:
        kwargs["top_k"] = top_k
    if max_output_tokens is not None and max_output_tokens > 0:
        kwargs["max_output_tokens"] = max_output_tokens
    if seed is not None and seed >= 0:
        kwargs["seed"] = seed
    if candidate_count is not None and candidate_count > 1:
        kwargs["candidate_count"] = candidate_count

    # Response modalities
    if modalities:
        kwargs["response_modalities"] = modalities

    # Safety settings — raise on invalid JSON so the user notices their filters
    # were never applied (previously this silently warned and continued, which
    # could produce harmful output that the user thought was being filtered).
    if safety_settings_json and safety_settings_json.strip():
        try:
            ss = json.loads(safety_settings_json)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(
                f"Invalid safety_settings_json — could not parse: {e}. "
                "Use the 'NanoBanana - Safety Settings' node to generate "
                "a well-formed JSON string."
            )
        if not isinstance(ss, list):
            raise ValueError(
                "safety_settings_json must be a JSON list of "
                "{category, threshold} objects."
            )
        kwargs["safety_settings"] = [types.SafetySetting(**s) for s in ss]

    # Structured output
    if response_mime_type:
        kwargs["response_mime_type"] = response_mime_type
    if response_schema:
        kwargs["response_schema"] = response_schema

    return types.GenerateContentConfig(**kwargs)


def _extract_image_from_stream(client, model, contents, config, return_all=False):
    """Stream generate and extract image(s) from response.

    Iterates ALL parts in ALL chunks (bug fix over original which only
    checked chunk.parts[0]).

    Args:
        return_all: if True, collect EVERY image part (useful when
            candidate_count > 1 — previously v2.0 charged for multiple
            candidates but silently discarded all but the first). Returns
            a batched tensor [N, H, W, C]. If False, returns the first
            image only as [1, H, W, C].

    Returns:
        torch.Tensor of shape [N, H, W, C] (or [1, H, W, C] if return_all=False).

    Raises:
        RuntimeError: if no image data is returned.
    """
    import torch
    model_text = []
    images = []
    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=config
    ):
        if chunk.parts is None:
            continue
        for part in chunk.parts:
            if part.inline_data and part.inline_data.data:
                img = bytes_to_tensor(part.inline_data.data)
                if not return_all:
                    return img
                images.append(img)
            elif part.text:
                model_text.append(part.text)

    if not images:
        text_msg = " ".join(model_text)[:300] if model_text else "No details"
        raise RuntimeError(
            f"NanoBanana - returned no image. Model said: {redact_secret(text_msg)}"
        )

    # Stack candidates — only if same dimensions. Otherwise return first.
    if len(images) == 1:
        return images[0]
    try:
        return torch.cat(images, dim=0)
    except RuntimeError:
        # Different sizes — return first only (rare; happens if model returns
        # mixed-size candidates which Gemini normally doesn't).
        logger.warning(
            "Image candidates have mismatched dimensions; returning first only."
        )
        return images[0]


# ===================================================================
# CONFIG NODES  (NanoBanana2/Config)
# ===================================================================

class NanoBanana_APIKey(BaseAPIKeyNode):
    """NanoBanana - API key provider with GEMINI_API_KEY env-var fallback."""

    ENV_VAR_NAME = "GEMINI_API_KEY"
    SERVICE_NAME = "Google Gemini"
    CATEGORY = "NanoBanana2/Config"


class NanoBanana_ModelSelector:
    """Dropdown selector for Gemini models with optional custom override.

    Note: ComfyUI dropdowns are static at load time, so this node shows
    ALL_MODELS in the picker. Use category_validate to enforce that the
    selected model matches your declared category at runtime — useful for
    catching mistakes like picking 'imagen-...' when you wanted text gen.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (ALL_MODELS, {
                    "default": ALL_MODELS[0],
                    "tooltip": "Select any Gemini/Imagen/Veo/Lyria/Embedding model.",
                }),
                "category_validate": (["any", "text", "image_gen", "imagen", "tts", "veo", "lyria", "embedding"], {
                    "default": "any",
                    "tooltip": "Validate that the selected model matches this category. Raises an error at runtime if not. Helps catch mistakes like 'imagen-...' selected for a text node.",
                }),
            },
            "optional": {
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Override with a custom model ID (leave blank to use dropdown).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model",)
    FUNCTION = "select"
    CATEGORY = "NanoBanana2/Config"

    def select(self, model, category_validate="any", custom_model=""):
        final = _resolve_model(model, custom_model)

        if category_validate != "any" and not custom_model.strip():
            # Only validate dropdown picks, not custom overrides
            category_lists = {
                "text": TEXT_MODELS,
                "image_gen": IMAGE_MODELS,
                "imagen": IMAGEN_MODELS,
                "tts": TTS_MODELS,
                "veo": VEO_MODELS,
                "lyria": LYRIA_MODELS,
                "embedding": EMBEDDING_MODELS,
            }
            allowed = category_lists.get(category_validate, [])
            if final not in allowed:
                raise ValueError(
                    f"Model '{final}' is not in the {category_validate} category. "
                    f"Allowed for {category_validate}: {allowed}"
                )

        return (final,)


class NanoBanana_SafetySettings:
    """Configure per-category safety thresholds. Outputs JSON string for other nodes."""

    _CATEGORIES = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    _THRESHOLDS = [
        "BLOCK_NONE",
        "BLOCK_ONLY_HIGH",
        "BLOCK_MEDIUM_AND_ABOVE",
        "BLOCK_LOW_AND_ABOVE",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "harassment": (cls._THRESHOLDS, {
                    "default": "BLOCK_MEDIUM_AND_ABOVE",
                    "tooltip": "Threshold for harassment content.",
                }),
                "hate_speech": (cls._THRESHOLDS, {
                    "default": "BLOCK_MEDIUM_AND_ABOVE",
                    "tooltip": "Threshold for hate speech content.",
                }),
                "sexually_explicit": (cls._THRESHOLDS, {
                    "default": "BLOCK_MEDIUM_AND_ABOVE",
                    "tooltip": "Threshold for sexually explicit content.",
                }),
                "dangerous_content": (cls._THRESHOLDS, {
                    "default": "BLOCK_MEDIUM_AND_ABOVE",
                    "tooltip": "Threshold for dangerous content.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("safety_settings_json",)
    FUNCTION = "build"
    CATEGORY = "NanoBanana2/Config"

    def build(self, harassment, hate_speech, sexually_explicit, dangerous_content):
        thresholds = [harassment, hate_speech, sexually_explicit, dangerous_content]
        settings = [
            {"category": cat, "threshold": thresh}
            for cat, thresh in zip(self._CATEGORIES, thresholds)
        ]
        return (json.dumps(settings),)


class NanoBanana_ThinkingConfig:
    """Configure thinking level and optional budget. Outputs JSON string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "thinking_level": (THINKING_LEVELS, {
                    "default": "NONE",
                    "tooltip": "How much the model should 'think' before answering.",
                }),
            },
            "optional": {
                "thinking_budget": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "tooltip": "Max thinking tokens (0 = model default).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("thinking_config_json",)
    FUNCTION = "build"
    CATEGORY = "NanoBanana2/Config"

    def build(self, thinking_level, thinking_budget=0):
        cfg = {"thinking_level": thinking_level}
        if thinking_budget and thinking_budget > 0:
            cfg["thinking_budget"] = thinking_budget
        return (json.dumps(cfg),)


# ===================================================================
# TEXT NODES  (NanoBanana2/Text)
# ===================================================================

class NanoBanana_TextGen(AlwaysExecuteMixin):
    """Full-featured text generation with Gemini models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var.",
                }),
                "model": (TEXT_MODELS, {
                    "default": "gemini-2.5-flash",
                    "tooltip": "NanoBanana - model for text generation.",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Override with a custom model ID.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The user prompt to send to the model.",
                }),
            },
            "optional": {
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System instruction to guide model behavior.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls randomness. Lower = more deterministic.",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling probability cutoff.",
                }),
                "top_k": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Top-K sampling (0 = disabled).",
                }),
                "max_output_tokens": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 65536,
                    "tooltip": "Max tokens in response (0 = model default).",
                }),
                "thinking_level": (THINKING_LEVELS, {
                    "default": "NONE",
                    "tooltip": "How much the model should reason before answering.",
                }),
                "thinking_budget": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "tooltip": "Max thinking tokens (0 = model default).",
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "Random seed for reproducibility (-1 = random).",
                }),
                "safety_settings_json": ("STRING", {
                    "default": "",
                    "tooltip": "JSON safety settings from Safety Settings node.",
                }),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "NanoBanana2/Text"

    def generate(
        self,
        api_key,
        model,
        custom_model,
        prompt,
        system_instruction="",
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        max_output_tokens=0,
        thinking_level="NONE",
        thinking_budget=0,
        seed=-1,
        safety_settings_json="",
        network=None,
    ):
        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        config = _build_config(
            modalities=["TEXT"],
            system_instruction=system_instruction,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            seed=seed if seed >= 0 else None,
            safety_settings_json=safety_settings_json,
        )

        def _call():
            response = client.models.generate_content(
                model=final_model, contents=prompt, config=config
            )
            return (response.text.strip() if response.text else "",)

        return retry_with_backoff(_call)


class NanoBanana_PromptRefiner(AlwaysExecuteMixin):
    """Refine and optimize prompts using Gemini. Generic system instruction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var.",
                }),
                "model": (TEXT_MODELS, {
                    "default": "gemini-2.5-pro",
                    "tooltip": "NanoBanana - model for prompt refinement.",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Override with a custom model ID.",
                }),
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The prompt to refine and improve.",
                }),
            },
            "optional": {
                "target": (["image", "video", "music", "text", "code"], {
                    "default": "image",
                    "tooltip": "What downstream generator the prompt is for. "
                               "v2.0 hardcoded 'image generator', breaking the "
                               "node for video/music/text refinement.",
                }),
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Instructions for how to refine the prompt. Leave empty for default behavior.",
                }),
                "thinking_level": (THINKING_LEVELS, {
                    "default": "NONE",
                    "tooltip": "How much the model should reason before answering.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls randomness in refinement.",
                }),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("refined_prompt",)
    FUNCTION = "refine_prompt"
    CATEGORY = "NanoBanana2/Text"

    def refine_prompt(
        self,
        api_key,
        model,
        custom_model,
        base_prompt,
        target="image",
        system_instruction="",
        thinking_level="NONE",
        temperature=0.7,
        network=None,
    ):
        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        target_phrase = {
            "image": "an AI Image Generator (think Imagen, Midjourney, SDXL, FLUX)",
            "video": "an AI Video Generator (think Veo, Sora, Runway)",
            "music": "an AI Music Generator (think Lyria, Suno, Udio)",
            "text": "another LLM that will produce a final answer",
            "code": "a code-generating LLM (favor explicit type / API / framework names)",
        }.get(target, "an AI Image Generator")

        sys_text = (
            "You are a world-class Prompt Engineer. "
            f"Your task is to take a user's base concept and expand it into a "
            f"highly detailed, professional prompt for {target_phrase}.\n\n"
        )
        if system_instruction and system_instruction.strip():
            sys_text += f"Refinement Style & Instructions: {system_instruction.strip()}\n\n"
        sys_text += (
            "CRITICAL: Output ONLY the final refined prompt text. "
            "Do not include introductory or concluding conversational text."
        )

        config = _build_config(
            modalities=["TEXT"],
            system_instruction=sys_text,
            thinking_level=thinking_level,
            temperature=temperature,
        )

        def _call():
            response = client.models.generate_content(
                model=final_model, contents=base_prompt, config=config
            )
            return (response.text.strip() if response.text else "",)

        return retry_with_backoff(_call)


class NanoBanana_MultiTurn(AlwaysExecuteMixin):
    """Multi-turn chat with conversation history as STRING input/output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var.",
                }),
                "model": (TEXT_MODELS, {
                    "default": "gemini-2.5-flash",
                    "tooltip": "NanoBanana - model for chat.",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Override with a custom model ID.",
                }),
                "message": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The new user message to send.",
                }),
            },
            "optional": {
                "conversation_history": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON conversation history from previous turn. Leave empty to start fresh.",
                }),
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System instruction for the chat.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls randomness.",
                }),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "conversation_history")
    FUNCTION = "chat"
    CATEGORY = "NanoBanana2/Text"

    def chat(
        self,
        api_key,
        model,
        custom_model,
        message,
        conversation_history="",
        system_instruction="",
        temperature=0.7,
        network=None,
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        # Parse existing history
        history = []
        if conversation_history and conversation_history.strip():
            try:
                history = json.loads(conversation_history)
            except json.JSONDecodeError:
                logger.warning("Invalid conversation history JSON, starting fresh.")

        # Build contents from history
        contents = []
        for entry in history:
            contents.append(
                types.Content(
                    role=entry["role"],
                    parts=[types.Part.from_text(text=entry["text"])],
                )
            )

        # Add new user message
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        )

        config = _build_config(
            modalities=["TEXT"],
            system_instruction=system_instruction,
            temperature=temperature,
        )

        def _call():
            response = client.models.generate_content(
                model=final_model, contents=contents, config=config
            )
            return response.text.strip() if response.text else ""

        reply = retry_with_backoff(_call)

        # Update history
        history.append({"role": "user", "text": message})
        history.append({"role": "model", "text": reply})

        return (reply, json.dumps(history))


class NanoBanana_StructuredOutput(AlwaysExecuteMixin):
    """JSON schema-constrained output from Gemini. Returns parsed JSON string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var.",
                }),
                "model": (TEXT_MODELS, {
                    "default": "gemini-2.5-flash",
                    "tooltip": "NanoBanana - model for structured output.",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Override with a custom model ID.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The prompt describing what data to extract/generate.",
                }),
                "json_schema": ("STRING", {
                    "multiline": True,
                    "default": '{\n  "type": "object",\n  "properties": {\n    "result": {"type": "string"}\n  }\n}',
                    "tooltip": "JSON Schema that constrains the model output format.",
                }),
            },
            "optional": {
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System instruction for the model.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Lower temperature recommended for structured output.",
                }),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)
    FUNCTION = "generate"
    CATEGORY = "NanoBanana2/Text"

    def generate(
        self,
        api_key,
        model,
        custom_model,
        prompt,
        json_schema,
        system_instruction="",
        temperature=0.3,
        network=None,
    ):
        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        try:
            schema = json.loads(json_schema)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON schema: {e}")

        config = _build_config(
            modalities=["TEXT"],
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )

        def _call():
            response = client.models.generate_content(
                model=final_model, contents=prompt, config=config
            )
            # Prefer the SDK's `.parsed` attribute when available — it
            # strips markdown fences and pre-parses the JSON. v2.0 always
            # used `.text` which broke when the model wrapped output in
            # ```json ... ``` fences.
            parsed = getattr(response, "parsed", None)
            if parsed is not None:
                try:
                    return (json.dumps(parsed),)
                except (TypeError, ValueError):
                    pass  # Fall through to text
            return (response.text.strip() if response.text else "{}",)

        return retry_with_backoff(_call)


# ===================================================================
# VISION / IMAGE NODES  (NanoBanana2/Image)
# ===================================================================

class NanoBanana_Vision(AlwaysExecuteMixin):
    """Analyze images with Gemini vision models and return text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var.",
                }),
                "model": (TEXT_MODELS, {
                    "default": "gemini-3.1-flash-lite-preview",
                    "tooltip": "NanoBanana - model for vision analysis.",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Override with a custom model ID.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image.",
                    "tooltip": "What to ask about the image(s).",
                }),
            },
            "optional": {
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System instruction to guide analysis.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls randomness. Lower = more focused analysis.",
                }),
                "lossless": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Encode images as lossless PNG instead of JPEG. "
                               "Use for OCR, small-text recognition, or "
                               "fine-grained classification where JPEG "
                               "artifacts degrade accuracy. Higher bandwidth.",
                }),
                "ref_image_1": ("IMAGE", {"tooltip": "First image to analyze."}),
                "ref_image_2": ("IMAGE", {"tooltip": "Second image to analyze."}),
                "ref_image_3": ("IMAGE", {"tooltip": "Third image to analyze."}),
                "ref_image_4": ("IMAGE", {"tooltip": "Fourth image to analyze."}),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "analyze"
    CATEGORY = "NanoBanana2/Image"

    def analyze(
        self,
        api_key,
        model,
        custom_model,
        prompt,
        system_instruction="",
        temperature=0.1,
        lossless=False,
        ref_image_1=None,
        ref_image_2=None,
        ref_image_3=None,
        ref_image_4=None,
        network=None,
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        # Build image parts — PNG for vision tasks where small details matter
        img_parts, img_count = _build_image_parts(
            [ref_image_1, ref_image_2, ref_image_3, ref_image_4],
            labels=True,
            lossless=lossless,
        )

        # Build contents
        parts = img_parts + [types.Part.from_text(text=prompt)]
        contents = [types.Content(role="user", parts=parts)]

        config = _build_config(
            modalities=["TEXT"],
            system_instruction=system_instruction,
            temperature=temperature,
        )

        def _call():
            response = client.models.generate_content(
                model=final_model, contents=contents, config=config
            )
            result = response.text.strip() if response.text else ""
            # Truncate stdout logging to avoid flooding console
            preview = result[:200] + "..." if len(result) > 200 else result
            print(f"[NanoBanana Vision] {preview}")
            return (result,)

        return retry_with_backoff(_call)


class NanoBanana_ImageGen(AlwaysExecuteMixin):
    """Multi-reference image generation using Gemini / Imagen models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var.",
                }),
                "model": (IMAGE_MODELS, {
                    "default": "gemini-3.1-flash-image-preview",
                    "tooltip": "NanoBanana - image model. For Imagen models, use the dedicated Imagen Image Generation node instead.",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Override with a custom model ID.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Image generation prompt. Use [Reference Image N] to refer to inputs.",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "16:9",
                    "tooltip": "Output image aspect ratio.",
                }),
                "image_size": (IMAGE_SIZES, {
                    "default": "4K",
                    "tooltip": "Output image resolution.",
                }),
            },
            "optional": {
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System instruction for the image generation model.",
                }),
                "thinking_level": (THINKING_LEVELS, {
                    "default": "NONE",
                    "tooltip": "How much the model should reason before generating.",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls randomness in generation.",
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "Random seed for reproducibility (-1 = random).",
                }),
                "candidate_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "tooltip": "Number of image candidates to generate. All "
                               "are returned as a batch — wire into Preview "
                               "Image or save the lot.",
                }),
                "ref_image_1": ("IMAGE", {"tooltip": "First reference image."}),
                "ref_image_2": ("IMAGE", {"tooltip": "Second reference image."}),
                "ref_image_3": ("IMAGE", {"tooltip": "Third reference image."}),
                "ref_image_4": ("IMAGE", {"tooltip": "Fourth reference image."}),
                "safety_settings_json": ("STRING", {
                    "default": "",
                    "tooltip": "JSON safety settings from Safety Settings node.",
                }),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "NanoBanana2/Image"

    def generate_image(
        self,
        api_key,
        model,
        custom_model,
        prompt,
        aspect_ratio,
        image_size,
        system_instruction="",
        thinking_level="NONE",
        temperature=1.0,
        seed=-1,
        candidate_count=1,
        ref_image_1=None,
        ref_image_2=None,
        ref_image_3=None,
        ref_image_4=None,
        safety_settings_json="",
        network=None,
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        # Build image parts
        ref_parts, ref_count = _build_image_parts(
            [ref_image_1, ref_image_2, ref_image_3, ref_image_4],
            labels=True,
        )

        # Build prompt
        final_prompt = prompt
        if ref_count > 1:
            final_prompt = f"--- [Main Prompt] ---\n{final_prompt}"

        parts = ref_parts + [types.Part.from_text(text=final_prompt)]
        contents = [types.Content(role="user", parts=parts)]

        config = _build_config(
            modalities=["IMAGE", "TEXT"],
            system_instruction=system_instruction,
            thinking_level=thinking_level,
            temperature=temperature,
            seed=seed if seed >= 0 else None,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            candidate_count=candidate_count,
            safety_settings_json=safety_settings_json,
        )

        def _call():
            # Return ALL candidates (v2.0 silently dropped 2-4 even though
            # they were paid for); user gets a batched IMAGE tensor.
            return _extract_image_from_stream(
                client, final_model, contents, config,
                return_all=(candidate_count > 1),
            )

        result = retry_with_backoff(_call)
        return (result,)


class NanoBanana_ImageEdit(AlwaysExecuteMixin):
    """Text-guided image editing: image + edit instruction + optional mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var.",
                }),
                "model": (IMAGE_MODELS, {
                    "default": "gemini-3.1-flash-image-preview",
                    "tooltip": "Image-capable Gemini model for editing.",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Override with a custom model ID.",
                }),
                "image": ("IMAGE", {
                    "tooltip": "The image to edit.",
                }),
                "edit_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describe what edits to make to the image.",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask — which areas to edit. White = edit, black = keep. Works with any MASK source: SAM, Grounding DINO, segmentation, alpha channels, RMBG. Gemini doesn't do pixel-perfect masking — it uses this as visual guidance alongside your instructions.",
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional reference image (e.g., a face to swap into the masked region).",
                }),
                "reference_image_2": ("IMAGE", {
                    "tooltip": "Optional second reference image.",
                }),
                "reference_image_3": ("IMAGE", {
                    "tooltip": "Optional third reference image.",
                }),
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System instruction for the editing model.",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "AUTO",
                    "tooltip": "Output image aspect ratio.",
                }),
                "image_size": (IMAGE_SIZES, {
                    "default": "AUTO",
                    "tooltip": "Output image resolution.",
                }),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "NanoBanana2/Image"

    def edit(
        self,
        api_key,
        model,
        custom_model,
        image,
        edit_instruction,
        mask=None,
        reference_image=None,
        reference_image_2=None,
        reference_image_3=None,
        system_instruction="",
        aspect_ratio="AUTO",
        image_size="AUTO",
        network=None,
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        parts = []

        # Source image (the one being edited) — PNG for lossless reference
        img_bytes = tensor_to_png_bytes(image)
        parts.append(types.Part.from_text(text="--- [Source Image — the image to edit] ---"))
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        # Optional mask — auto-resize to match image dims (Gemini 400s on mismatch),
        # PNG for crisp boundaries
        if mask is not None:
            mask_aligned = resize_mask_to_image(mask, image)
            mask_bytes = mask_to_png_bytes(mask_aligned)
            parts.append(types.Part.from_text(text="--- [Edit Mask — white pixels = edit this area, black = keep unchanged] ---"))
            parts.append(types.Part.from_bytes(data=mask_bytes, mime_type="image/png"))

        # Optional reference images (for face swaps, style refs, etc.) — PNG
        # preserves identity better than JPEG
        ref_idx = 1
        for ref in (reference_image, reference_image_2, reference_image_3):
            if ref is not None:
                ref_bytes = tensor_to_png_bytes(ref)
                parts.append(types.Part.from_text(text=f"--- [Reference Image {ref_idx} — use as source for the edited region] ---"))
                parts.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/png"))
                ref_idx += 1

        # Edit instruction
        parts.append(types.Part.from_text(text=f"Edit instruction: {edit_instruction}"))
        contents = [types.Content(role="user", parts=parts)]

        config = _build_config(
            modalities=["IMAGE", "TEXT"],
            system_instruction=system_instruction,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
        )

        def _call():
            return _extract_image_from_stream(client, final_model, contents, config)

        result = retry_with_backoff(_call)
        return (result,)


class NanoBanana_Inpaint(AlwaysExecuteMixin):
    """Inpainting: fill masked areas of an image guided by a prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var.",
                }),
                "model": (IMAGE_MODELS, {
                    "default": "gemini-3.1-flash-image-preview",
                    "tooltip": "Image-capable Gemini model for inpainting.",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Override with a custom model ID.",
                }),
                "image": ("IMAGE", {
                    "tooltip": "The source image to inpaint.",
                }),
                "mask": ("MASK", {
                    "tooltip": "Mask (white = fill this area, black = keep). Accepts any ComfyUI MASK — from SAM, Grounding DINO, alpha channels, etc.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describe what should fill the masked area.",
                }),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional reference image (e.g., object/face to place in the masked area).",
                }),
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System instruction for the inpainting model.",
                }),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint"
    CATEGORY = "NanoBanana2/Image"

    def inpaint(
        self,
        api_key,
        model,
        custom_model,
        image,
        mask,
        prompt,
        reference_image=None,
        system_instruction="",
        network=None,
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        parts = []

        # Source image — PNG so mask boundary alignment is pixel-exact
        img_bytes = tensor_to_png_bytes(image)
        parts.append(types.Part.from_text(text="--- [Source Image] ---"))
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        # Mask must match image dims (Gemini returns 400 otherwise) — resize
        # if upstream MASK source (SAM/depth/etc) produced a different size.
        # Use PNG to keep mask edges crisp; JPEG smears them ~2px.
        mask_aligned = resize_mask_to_image(mask, image)
        mask_bytes = mask_to_png_bytes(mask_aligned)
        parts.append(types.Part.from_text(text="--- [Inpaint Mask (white = fill area)] ---"))
        parts.append(types.Part.from_bytes(data=mask_bytes, mime_type="image/png"))

        # Optional reference image — PNG, lossless ref preserves identity
        if reference_image is not None:
            ref_bytes = tensor_to_png_bytes(reference_image)
            parts.append(types.Part.from_text(text="--- [Reference Image — use as source for the inpainted region] ---"))
            parts.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/png"))

        # Prompt
        parts.append(
            types.Part.from_text(
                text=f"Inpaint the masked area with: {prompt}"
            )
        )
        contents = [types.Content(role="user", parts=parts)]

        sys_text = system_instruction or (
            "You are an expert image inpainter. Fill the white-masked area "
            "seamlessly, matching the surrounding context and the user's prompt."
        )

        config = _build_config(
            modalities=["IMAGE", "TEXT"],
            system_instruction=sys_text,
        )

        def _call():
            return _extract_image_from_stream(client, final_model, contents, config)

        result = retry_with_backoff(_call)
        return (result,)


class NanoBanana_Outpaint(AlwaysExecuteMixin):
    """Outpainting: extend an image in a given direction with a prompt."""

    _DIRECTIONS = ["right", "left", "up", "down", "all"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var.",
                }),
                "model": (IMAGE_MODELS, {
                    "default": "gemini-3.1-flash-image-preview",
                    "tooltip": "Image-capable Gemini model for outpainting.",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Override with a custom model ID.",
                }),
                "image": ("IMAGE", {
                    "tooltip": "The source image to extend.",
                }),
                "direction": (cls._DIRECTIONS, {
                    "default": "right",
                    "tooltip": "Direction to extend the image.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describe what should appear in the extended area.",
                }),
            },
            "optional": {
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System instruction for the outpainting model.",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "AUTO",
                    "tooltip": "Target aspect ratio after outpainting.",
                }),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "outpaint"
    CATEGORY = "NanoBanana2/Image"

    def outpaint(
        self,
        api_key,
        model,
        custom_model,
        image,
        direction,
        prompt,
        system_instruction="",
        aspect_ratio="AUTO",
        network=None,
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        parts = []

        # Source image
        img_bytes = tensor_to_jpeg_bytes(image, quality=95)
        parts.append(types.Part.from_text(text="--- [Source Image] ---"))
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        # Outpaint instruction
        parts.append(
            types.Part.from_text(
                text=(
                    f"Extend this image to the {direction}. "
                    f"Fill the new area with: {prompt}"
                )
            )
        )
        contents = [types.Content(role="user", parts=parts)]

        sys_text = system_instruction or (
            "You are an expert image outpainter. Extend the image seamlessly "
            "in the specified direction, maintaining style and context consistency."
        )

        config = _build_config(
            modalities=["IMAGE", "TEXT"],
            system_instruction=sys_text,
            aspect_ratio=aspect_ratio,
        )

        def _call():
            return _extract_image_from_stream(client, final_model, contents, config)

        result = retry_with_backoff(_call)
        return (result,)


# ===================================================================
# List Available Models (queries API for what your key can actually access)
# ===================================================================

class NanoBanana_ListModels(AlwaysExecuteMixin):
    """List models available for your API key.

    Useful when model IDs change between API versions or between Vertex AI
    and the Developer API. Returns the raw list so you know what to put in
    custom_model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var.",
                }),
            },
            "optional": {
                "filter": (["all", "image_generation", "text_only", "multimodal"], {
                    "default": "all",
                    "tooltip": "Filter by capability. 'image_generation' shows only models that can output images.",
                }),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("models_text", "models_json")
    FUNCTION = "list_models"
    CATEGORY = "NanoBanana2/Config"

    def list_models(self, api_key, filter="all", network=None):
        key = get_api_key(api_key)
        client = get_client(key, network=network)

        all_models = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            # Strip "models/" prefix if present
            clean_name = name.replace("models/", "")
            methods = list(getattr(m, "supported_actions", []) or [])
            # Also check older attribute name
            if not methods:
                methods = list(getattr(m, "supported_generation_methods", []) or [])
            display = getattr(m, "display_name", "") or ""
            desc = getattr(m, "description", "") or ""

            entry = {
                "name": clean_name,
                "display_name": display,
                "description": desc[:200],
                "methods": methods,
            }
            all_models.append(entry)

        # Apply filter
        filtered = all_models
        if filter == "image_generation":
            filtered = [m for m in all_models
                        if "image" in m["name"].lower()
                        or any("image" in x.lower() for x in m["methods"])
                        or "imagen" in m["name"].lower()]
        elif filter == "text_only":
            filtered = [m for m in all_models
                        if "image" not in m["name"].lower() and "vision" not in m["name"].lower()]
        elif filter == "multimodal":
            filtered = [m for m in all_models
                        if any(m["name"].lower().startswith(p) for p in ["gemini", "models/gemini"])]

        # Build human-readable text
        lines = []
        for m in filtered:
            method_str = ", ".join(m["methods"]) if m["methods"] else "?"
            lines.append(f"{m['name']}  |  {m['display_name']}  |  [{method_str}]")
        text_out = "\n".join(lines) if lines else "(no matching models)"

        return (text_out, json.dumps(filtered, indent=2))


# ===================================================================
# Imagen Generation (uses generate_images endpoint, not generate_content)
# ===================================================================

class NanoBanana_ImagenGen(AlwaysExecuteMixin):
    """Generate images using Google Imagen models.

    Imagen uses a separate API endpoint (generate_images) from Gemini's image
    generation. It supports Imagen 4 Ultra / Standard / Fast, plus Imagen 3.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var.",
                }),
                "model": (IMAGEN_MODELS, {
                    "default": "imagen-4.0-generate-001",
                    "tooltip": "Imagen model. Ultra = highest quality, Standard = balanced, Fast = cheaper/faster.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describe the image to generate.",
                }),
            },
            "optional": {
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Override with a custom model ID (takes priority over dropdown).",
                }),
                "number_of_images": ("INT", {
                    "default": 1, "min": 1, "max": 4,
                    "tooltip": "How many images to generate in one call (Imagen supports up to 4).",
                }),
                "aspect_ratio": (IMAGEN_ASPECT_RATIOS, {
                    "default": "1:1",
                    "tooltip": "Output aspect ratio. Imagen supports: 1:1, 3:4, 4:3, 9:16, 16:9.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Things to avoid in the image.",
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2147483647,
                    "tooltip": "Seed for reproducibility. -1 = random. "
                               "(0 is a valid seed for some models — v2.0 silently "
                               "dropped seed=0; v2.1 honors it. Use -1 for random.)",
                }),
                "safety_filter_level": (
                    IMAGEN_SAFETY_LEVELS,
                    {"default": "block_low_and_above",
                     "tooltip": "Safety filter threshold. Developer API often only "
                                "accepts block_low_and_above on free tier; Vertex AI "
                                "supports all four. API surfaces a clear error if "
                                "the level isn't accepted for your model/account."}
                ),
                "person_generation": (
                    ["dont_allow", "allow_adult", "allow_all"],
                    {"default": "allow_adult",
                     "tooltip": "Whether/how to generate people. Some models enforce stricter defaults."}
                ),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "NanoBanana2/Image"

    def generate(
        self,
        api_key,
        model,
        prompt,
        custom_model="",
        number_of_images=1,
        aspect_ratio="1:1",
        negative_prompt="",
        seed=-1,
        safety_filter_level="block_low_and_above",
        person_generation="allow_adult",
        network=None,
    ):
        from google.genai import types
        import torch
        import numpy as np

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        cfg_kwargs = {
            "number_of_images": number_of_images,
            "aspect_ratio": aspect_ratio,
            "safety_filter_level": safety_filter_level,
            "person_generation": person_generation,
        }
        if negative_prompt and negative_prompt.strip():
            cfg_kwargs["negative_prompt"] = negative_prompt.strip()
        # v2.0 used `if seed > 0` which silently dropped seed=0. Now -1 means
        # "random, don't send" and 0..N is forwarded faithfully.
        if seed >= 0:
            cfg_kwargs["seed"] = seed

        config = types.GenerateImagesConfig(**cfg_kwargs)

        def _call():
            response = client.models.generate_images(
                model=final_model,
                prompt=prompt,
                config=config,
            )
            return response

        response = retry_with_backoff(_call)

        generated = getattr(response, "generated_images", None) or []
        if not generated:
            raise RuntimeError(
                f"Imagen returned no images. Model: {final_model}. "
                f"Prompt may have been blocked by safety filters."
            )

        # Each generated_image has .image with .image_bytes
        tensors = []
        for gen_img in generated:
            img_bytes = None
            # Different SDK versions expose this differently — handle both
            if hasattr(gen_img, "image") and gen_img.image is not None:
                img_bytes = getattr(gen_img.image, "image_bytes", None)
            if img_bytes is None and hasattr(gen_img, "image_bytes"):
                img_bytes = gen_img.image_bytes
            if img_bytes is None:
                continue
            tensors.append(bytes_to_tensor(img_bytes))

        if not tensors:
            raise RuntimeError("Imagen returned results but no image data could be extracted.")

        # Concatenate into a batch. Single-image case skips cat (cheaper).
        # If candidates somehow returned mixed sizes (rare), fall back to
        # the first to avoid a confusing torch.cat ValueError.
        if len(tensors) == 1:
            return (tensors[0],)
        try:
            return (torch.cat(tensors, dim=0),)
        except RuntimeError:
            logger.warning(
                "Imagen candidates have mismatched sizes (%s); returning first only.",
                [tuple(t.shape) for t in tensors],
            )
            return (tensors[0],)


# ===================================================================
# Gemini TTS (text-to-speech, audio output modality)
# ===================================================================

class NanoBanana_TTS(AlwaysExecuteMixin):
    """NanoBanana - Text-to-Speech. Uses generate_content with audio response modality.

    Supports 30+ prebuilt voices (Zephyr, Puck, Kore, etc.). Can voice natural
    speech with emotion, tone, and pacing. For multi-speaker dialogue, use
    plain text with speaker tags like 'Alice: Hi!  Bob: Hello.'
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True,
                    "tooltip": "NanoBanana - API key. Leave blank to use GEMINI_API_KEY env var."}),
                "model": (TTS_MODELS, {"default": "gemini-2.5-flash-preview-tts",
                    "tooltip": "TTS model. Pro = higher quality, Flash = faster."}),
                "text": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Text to speak. Can include speaker tags for multi-speaker dialogue."}),
                "voice": (TTS_VOICES, {"default": "Kore",
                    "tooltip": "Prebuilt voice. Each has different characteristics."}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                "style_prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Optional style instruction prepended to text (e.g., 'Say cheerfully:')."}),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "NanoBanana2/Audio"

    def generate(self, api_key, model, text, voice, custom_model="", style_prompt="", network=None):
        from google.genai import types
        import torch
        import numpy as np

        # Gemini TTS hard limit is ~32k chars per request; warn loudly above 16k.
        if len(text) > 32000:
            raise ValueError(
                f"TTS text is {len(text)} chars; Gemini TTS limit is ~32,000. "
                "Split into multiple calls and concatenate the resulting audio."
            )
        if len(text) > 16000:
            print(f"[NanoBanana TTS] WARNING: text is {len(text)} chars; "
                  "may approach API limits, consider splitting.")

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        full_text = f"{style_prompt.strip()}\n{text}" if style_prompt.strip() else text

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        )

        def _call():
            return client.models.generate_content(
                model=final_model, contents=full_text, config=config
            )

        response = retry_with_backoff(_call)

        # Extract raw PCM audio from response
        audio_bytes = None
        for cand in response.candidates or []:
            for part in (cand.content.parts or []):
                if part.inline_data and part.inline_data.data:
                    audio_bytes = part.inline_data.data
                    break
            if audio_bytes:
                break

        if audio_bytes is None:
            raise RuntimeError("NanoBanana - TTS returned no audio data.")

        # Gemini TTS returns 24kHz signed 16-bit PCM
        sample_rate = 24000
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)

        return ({"waveform": waveform, "sample_rate": sample_rate},)


# ===================================================================
# Gemini Embeddings (text vectors)
# ===================================================================

class NanoBanana_Embed(AlwaysExecuteMixin):
    """Generate text embeddings with Gemini embedding models.

    Useful for semantic search, clustering, classification, or RAG pipelines.
    Returns the vector as JSON. gemini-embedding-001 produces 768-dim vectors
    by default (configurable via output_dim).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (EMBEDDING_MODELS, {"default": "gemini-embedding-001"}),
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                "task_type": (["SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING",
                               "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "QUESTION_ANSWERING",
                               "FACT_VERIFICATION", "CODE_RETRIEVAL_QUERY"],
                              {"default": "SEMANTIC_SIMILARITY",
                               "tooltip": "Optimize the embedding for this downstream task."}),
                "output_dim": ("INT", {"default": 768, "min": 128, "max": 3072, "step": 128,
                    "tooltip": "Output dimensionality. 768 is default. Larger = more expressive."}),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("embedding_json", "dim")
    FUNCTION = "embed"
    CATEGORY = "NanoBanana2/Embeddings"

    def embed(self, api_key, model, text, custom_model="",
              task_type="SEMANTIC_SIMILARITY", output_dim=768, network=None):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dim,
        )

        def _call():
            return client.models.embed_content(
                model=final_model, contents=text, config=config
            )

        response = retry_with_backoff(_call)
        embeddings = response.embeddings or []
        if not embeddings:
            raise RuntimeError("Embedding request returned no vectors.")
        vec = list(embeddings[0].values)
        return (json.dumps(vec), len(vec))


# ===================================================================
# Veo Video Generation
# ===================================================================

class NanoBanana_VideoGen(AlwaysExecuteMixin):
    """Generate video using Google Veo models.

    Uses the predictLongRunning endpoint with polling. Veo 3 produces ~8-second
    videos with native audio. Veo 2 is silent video. Can be text-to-video or
    image-to-video (provide an optional source image).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (VEO_MODELS, {"default": "veo-3.0-fast-generate-001"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                "source_image": ("IMAGE", {"tooltip": "Optional start frame image."}),
                "aspect_ratio": (VEO_ASPECT_RATIOS, {"default": "16:9"}),
                "number_of_videos": ("INT", {"default": 1, "min": 1, "max": 4}),
                "negative_prompt": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647,
                    "tooltip": "-1 = random. 0 IS a valid seed (v2.0 silently dropped it)."}),
                "duration_seconds": ("INT", {"default": 8, "min": 4, "max": 8,
                    "tooltip": "Veo 2 supports 5-8s; Veo 3 supports 4, 6, or 8s "
                               "depending on model variant."}),
                "timeout_seconds": ("INT", {"default": 600, "min": 60, "max": 1800,
                    "tooltip": "Max time to wait for video generation."}),
                "poll_interval": ("INT", {"default": 10, "min": 2, "max": 60,
                    "tooltip": "Seconds between operation polls. Lower = "
                               "more responsive cancel, higher = less API noise."}),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_file_path", "video_url")
    FUNCTION = "generate"
    CATEGORY = "NanoBanana2/Video"

    def generate(self, api_key, model, prompt, custom_model="", source_image=None,
                 aspect_ratio="16:9", number_of_videos=1, negative_prompt="",
                 seed=-1, duration_seconds=8, timeout_seconds=600,
                 poll_interval=10, network=None):
        from google.genai import types
        import time
        import os
        import uuid
        from urllib.parse import urlparse

        try:
            from .shared.retry import stream_to_file
        except ImportError:
            from shared.retry import stream_to_file

        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except ImportError:
            output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)

        key = get_api_key(api_key)
        # Sanitize so a malicious custom_model can't escape /models/ in URL.
        final_model = sanitize_model_id(_resolve_model(model, custom_model))
        client = get_client(key, network=network)

        cfg_kwargs = {
            "aspect_ratio": aspect_ratio,
            "number_of_videos": number_of_videos,
        }
        if negative_prompt.strip():
            cfg_kwargs["negative_prompt"] = negative_prompt.strip()
        # Honor seed=0 (v2.0 silently dropped it).
        if seed >= 0:
            cfg_kwargs["seed"] = seed
        if duration_seconds:
            cfg_kwargs["duration_seconds"] = duration_seconds

        config = types.GenerateVideosConfig(**cfg_kwargs)

        kwargs = {"model": final_model, "prompt": prompt, "config": config}
        if source_image is not None:
            # PNG preserves details for img2vid identity transfer
            img_bytes = tensor_to_png_bytes(source_image)
            kwargs["image"] = types.Image(image_bytes=img_bytes, mime_type="image/png")

        print(f"[Gemini Veo] Starting video generation with {final_model}...")
        operation = client.models.generate_videos(**kwargs)

        # Poll. ComfyUI worker is single-threaded so this is unavoidably
        # blocking, but at least make the interval configurable and log
        # progress so users know it's not hung.
        start = time.time()
        while not operation.done:
            if time.time() - start > timeout_seconds:
                raise RuntimeError(
                    f"Veo generation timed out after {timeout_seconds}s "
                    f"(operation={getattr(operation, 'name', '?')})"
                )
            elapsed = int(time.time() - start)
            print(f"[Gemini Veo] [{elapsed}s] Polling...")
            time.sleep(max(1, int(poll_interval)))
            operation = client.operations.get(operation)

        result = getattr(operation, "response", None) or getattr(operation, "result", None)
        videos = getattr(result, "generated_videos", []) or []
        if not videos:
            raise RuntimeError("Veo operation completed but returned no videos.")

        # Save ALL videos (v2.0 dropped 2..N). First file_path / URI returned
        # as scalars for backward compat; full set in a JSON list.
        saved_paths = []
        saved_uris = []
        for i, vid in enumerate(videos):
            video_obj = getattr(vid, "video", None) or vid
            filename = f"gemini_veo_{uuid.uuid4().hex[:8]}_{i}.mp4"
            file_path = os.path.join(output_dir, filename)

            if hasattr(video_obj, "save"):
                video_obj.save(file_path)
            elif hasattr(video_obj, "uri") and video_obj.uri:
                # SSRF guard — the SDK should only ever hand us googleapis or
                # googleusercontent URIs. Refuse anything else.
                netloc = urlparse(video_obj.uri).netloc.lower()
                if not (netloc.endswith(".googleapis.com")
                        or netloc.endswith(".googleusercontent.com")
                        or netloc == "googleapis.com"
                        or netloc == "googleusercontent.com"):
                    raise RuntimeError(
                        f"Refusing to download video from non-Google host: {netloc}"
                    )
                # stream_to_file enforces size cap + cleans up on failure.
                stream_to_file(
                    video_obj.uri, file_path,
                    timeout=300, max_bytes=2 * 1024 * 1024 * 1024,  # 2 GiB cap
                )
            elif hasattr(video_obj, "video_bytes") and video_obj.video_bytes:
                with open(file_path, "wb") as f:
                    f.write(video_obj.video_bytes)
            else:
                raise RuntimeError(
                    f"Could not extract video data from response: {dir(video_obj)}"
                )

            saved_paths.append(file_path)
            saved_uris.append(getattr(video_obj, "uri", "") or "")
            print(f"[Gemini Veo] Saved video {i + 1}/{len(videos)}: {filename}")

        # Return first path + uri (backward compatible) — additional videos
        # are on disk for downstream loaders / VHS combine.
        return (saved_paths[0], saved_uris[0])


# ===================================================================
# Lyria Music Generation
# ===================================================================

class NanoBanana_MusicGen(AlwaysExecuteMixin):
    """Generate music using Google Lyria models.

    Lyria 3 Clip produces ~30 second clips, Lyria 3 Pro is higher quality.
    Uses predict endpoint for synchronous generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (LYRIA_MODELS, {"default": "lyria-3-clip-preview"}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Describe the music (genre, mood, instruments, tempo)."}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "sample_count": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "NanoBanana2/Audio"

    def generate(self, api_key, model, prompt, custom_model="",
                 negative_prompt="", seed=0, sample_count=1):
        import torch
        import numpy as np
        import requests
        import io

        key = get_api_key(api_key)
        # Hard-validate model ID — without this a malicious custom_model like
        # "../some-other-endpoint" could escape the /models/ path (URL-path
        # injection / SSRF within googleapis.com).
        final_model = sanitize_model_id(_resolve_model(model, custom_model))

        # Lyria uses the :predict endpoint directly. Pass key via header (NOT
        # query param) so it can't end up in proxy/CDN access logs or URL-
        # truncation tracebacks.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{final_model}:predict"
        instances = [{"prompt": prompt}]
        if negative_prompt.strip():
            instances[0]["negativePrompt"] = negative_prompt.strip()

        parameters = {"sampleCount": sample_count}
        # v2.0 used `if seed > 0` which silently dropped seed=0; v2.1 honors it.
        if seed >= 0:
            parameters["seed"] = seed

        body = {"instances": instances, "parameters": parameters}

        def _call():
            # Context-manage so the connection is returned to the pool even
            # if .json() throws or we raise mid-handling.
            with requests.post(
                url, json=body,
                headers={"x-goog-api-key": key},
                timeout=600,
            ) as resp:
                if resp.status_code >= 400:
                    # Redact in case the server echoes back the request headers
                    # (some Google error pages do for debug purposes).
                    raise RuntimeError(
                        f"Lyria API error {resp.status_code}: "
                        f"{redact_secret(resp.text[:400])}"
                    )
                return resp.json()

        data = retry_with_backoff(_call)

        predictions = data.get("predictions", [])
        if not predictions:
            raise RuntimeError(f"Lyria returned no audio. Response: {data}")

        # Extract audio (base64-encoded)
        import base64
        pred = predictions[0]
        audio_b64 = pred.get("bytesBase64Encoded") or pred.get("audio") or ""
        if not audio_b64:
            raise RuntimeError(f"Could not extract audio from Lyria response: {pred}")

        audio_bytes = base64.b64decode(audio_b64)

        # Lyria output is typically 48kHz stereo PCM or WAV
        try:
            import soundfile as sf
            buf = io.BytesIO(audio_bytes)
            audio_np, sample_rate = sf.read(buf, dtype="float32")
            if audio_np.ndim == 1:
                waveform = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio_np.T).unsqueeze(0)
        except Exception:
            # Fallback: assume 48kHz int16 PCM
            sample_rate = 48000
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)

        return ({"waveform": waveform, "sample_rate": sample_rate},)


# ===================================================================
# Token Counter (utility)
# ===================================================================

class NanoBanana_CountTokens(AlwaysExecuteMixin):
    """Count tokens in a prompt for a given model. Useful for cost estimation
    and context window checks. Doesn't charge against your quota."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (TEXT_MODELS, {"default": "gemini-2.5-flash"}),
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this request through that proxy (e.g. US egress)."}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("token_count",)
    FUNCTION = "count"
    CATEGORY = "NanoBanana2/Config"

    def count(self, api_key, model, text, custom_model="", network=None):
        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key, network=network)

        response = client.models.count_tokens(model=final_model, contents=text)
        total = getattr(response, "total_tokens", 0) or 0
        print(f"[Gemini] Token count for {final_model}: {total}")
        return (total,)


# ===================================================================
# NODE MAPPINGS
# ===================================================================

NODE_CLASS_MAPPINGS = {
    # Config (6)
    "NanoBanana_APIKey": NanoBanana_APIKey,
    "NanoBanana_ModelSelector": NanoBanana_ModelSelector,
    "NanoBanana_SafetySettings": NanoBanana_SafetySettings,
    "NanoBanana_ThinkingConfig": NanoBanana_ThinkingConfig,
    "NanoBanana_ListModels": NanoBanana_ListModels,
    "NanoBanana_CountTokens": NanoBanana_CountTokens,
    # Text (4)
    "NanoBanana_TextGen": NanoBanana_TextGen,
    "NanoBanana_PromptRefiner": NanoBanana_PromptRefiner,
    "NanoBanana_MultiTurn": NanoBanana_MultiTurn,
    "NanoBanana_StructuredOutput": NanoBanana_StructuredOutput,
    # Image (6)
    "NanoBanana_Vision": NanoBanana_Vision,
    "NanoBanana_ImageGen": NanoBanana_ImageGen,
    "NanoBanana_ImagenGen": NanoBanana_ImagenGen,
    "NanoBanana_ImageEdit": NanoBanana_ImageEdit,
    "NanoBanana_Inpaint": NanoBanana_Inpaint,
    "NanoBanana_Outpaint": NanoBanana_Outpaint,
    # Audio (2)
    "NanoBanana_TTS": NanoBanana_TTS,
    "NanoBanana_MusicGen": NanoBanana_MusicGen,
    # Video (1)
    "NanoBanana_VideoGen": NanoBanana_VideoGen,
    # Embeddings (1)
    "NanoBanana_Embed": NanoBanana_Embed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Config
    "NanoBanana_APIKey": "NanoBanana - API Key",
    "NanoBanana_ModelSelector": "NanoBanana - Model Selector",
    "NanoBanana_SafetySettings": "NanoBanana - Safety Settings",
    "NanoBanana_ThinkingConfig": "NanoBanana - Thinking Config",
    "NanoBanana_ListModels": "NanoBanana - List Available Models",
    "NanoBanana_CountTokens": "NanoBanana - Token Counter",
    # Text
    "NanoBanana_TextGen": "NanoBanana - Text Generation",
    "NanoBanana_PromptRefiner": "NanoBanana - Prompt Refiner",
    "NanoBanana_MultiTurn": "NanoBanana - Multi-Turn Chat",
    "NanoBanana_StructuredOutput": "NanoBanana - Structured Output (JSON)",
    # Image
    "NanoBanana_Vision": "NanoBanana - Vision Analysis",
    "NanoBanana_ImageGen": "NanoBanana - Image Generation (Nano Banana)",
    "NanoBanana_ImagenGen": "Imagen Image Generation",
    "NanoBanana_ImageEdit": "NanoBanana - Image Edit",
    "NanoBanana_Inpaint": "NanoBanana - Inpaint",
    "NanoBanana_Outpaint": "NanoBanana - Outpaint",
    # Audio
    "NanoBanana_TTS": "NanoBanana - Text-to-Speech",
    "NanoBanana_MusicGen": "NanoBanana - Music Generation (Lyria)",
    # Video
    "NanoBanana_VideoGen": "NanoBanana - Video Generation (Veo)",
    # Embeddings
    "NanoBanana_Embed": "NanoBanana - Text Embeddings",
}
