"""ComfyUI nodes for Google Gemini API.

Provides 13 nodes across Image, Text, and Config subcategories.
"""

import json
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared imports  (google-genai imported lazily inside methods)
# ---------------------------------------------------------------------------

try:
    from .shared.node_utils import AlwaysExecuteMixin
    from .shared.auth import BaseAPIKeyNode
    from .shared.conversions import tensor_to_jpeg_bytes, bytes_to_tensor
except ImportError:
    # Fallback for direct testing / flat layouts
    from shared.node_utils import AlwaysExecuteMixin
    from shared.auth import BaseAPIKeyNode
    from shared.conversions import tensor_to_jpeg_bytes, bytes_to_tensor

from .gemini_client import (
    get_client,
    get_api_key,
    retry_with_backoff,
    TEXT_MODELS,
    IMAGE_MODELS,
    IMAGEN_MODELS,
    IMAGEN_ASPECT_RATIOS,
    ALL_MODELS,
    ASPECT_RATIOS,
    THINKING_LEVELS,
    IMAGE_SIZES,
)


# ===================================================================
# Helper functions
# ===================================================================

def _resolve_model(model, custom_model):
    """Return custom_model if non-empty, else model."""
    cm = custom_model.strip() if custom_model else ""
    return cm if cm else model


def _build_image_parts(ref_images, labels=True):
    """Convert a list of optional image tensors into genai Part objects.

    Args:
        ref_images: list of (tensor_or_None) image inputs.
        labels: whether to prepend text labels.

    Returns:
        (parts_list, ref_count) - list of genai Part objects and count of images added.
    """
    from google.genai import types

    parts = []
    ref_count = 1
    for tensor_batch in ref_images:
        if tensor_batch is None:
            continue
        try:
            for i in range(tensor_batch.shape[0]):
                single = tensor_batch[i : i + 1]
                img_bytes = tensor_to_jpeg_bytes(single, quality=95)
                if labels:
                    parts.append(
                        types.Part.from_text(text=f"--- [Reference Image {ref_count}] ---")
                    )
                parts.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                )
                ref_count += 1
        except Exception as e:
            raise ValueError(
                f"Failed to convert reference image {ref_count} to JPEG: {e}"
            )
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

    # Safety settings
    if safety_settings_json and safety_settings_json.strip():
        try:
            ss = json.loads(safety_settings_json)
            if isinstance(ss, list):
                kwargs["safety_settings"] = [
                    types.SafetySetting(**s) for s in ss
                ]
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid safety_settings JSON, ignoring.")

    # Structured output
    if response_mime_type:
        kwargs["response_mime_type"] = response_mime_type
    if response_schema:
        kwargs["response_schema"] = response_schema

    return types.GenerateContentConfig(**kwargs)


def _extract_image_from_stream(client, model, contents, config):
    """Stream generate and extract first image from response.

    Iterates ALL parts in ALL chunks (bug fix over original which only
    checked chunk.parts[0]).

    Returns:
        torch.Tensor of shape [1, H, W, C]

    Raises:
        RuntimeError: if no image data is returned.
    """
    model_text = []
    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=config
    ):
        if chunk.parts is None:
            continue
        for part in chunk.parts:
            if part.inline_data and part.inline_data.data:
                return bytes_to_tensor(part.inline_data.data)
            elif part.text:
                model_text.append(part.text)
    text_msg = " ".join(model_text)[:300] if model_text else "No details"
    raise RuntimeError(f"NanoBanana - returned no image. Model said: {text_msg}")


# ===================================================================
# CONFIG NODES  (NanoBanana2/Config)
# ===================================================================

class NanoBanana_APIKey(BaseAPIKeyNode):
    """NanoBanana - API key provider with GEMINI_API_KEY env-var fallback."""

    ENV_VAR_NAME = "GEMINI_API_KEY"
    SERVICE_NAME = "Google Gemini"
    CATEGORY = "NanoBanana2/Config"


class NanoBanana_ModelSelector:
    """Dropdown selector for Gemini models with optional custom override."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["all", "text", "image"], {
                    "default": "all",
                    "tooltip": "Filter model list by capability.",
                }),
                "model": (ALL_MODELS, {
                    "default": ALL_MODELS[0],
                    "tooltip": "Select a Gemini model.",
                }),
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

    def select(self, model_type, model, custom_model):
        return (_resolve_model(model, custom_model),)


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
    ):
        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key)

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
        system_instruction="",
        thinking_level="NONE",
        temperature=0.7,
    ):
        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key)

        # Build system context
        sys_text = (
            "You are a world-class Prompt Engineer. "
            "Your task is to take a user's base concept and expand it into a highly detailed, "
            "professional prompt for an AI Image Generator.\n\n"
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
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key)

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
    ):
        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key)

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
                "ref_image_1": ("IMAGE", {"tooltip": "First image to analyze."}),
                "ref_image_2": ("IMAGE", {"tooltip": "Second image to analyze."}),
                "ref_image_3": ("IMAGE", {"tooltip": "Third image to analyze."}),
                "ref_image_4": ("IMAGE", {"tooltip": "Fourth image to analyze."}),
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
        ref_image_1=None,
        ref_image_2=None,
        ref_image_3=None,
        ref_image_4=None,
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key)

        # Build image parts
        img_parts, img_count = _build_image_parts(
            [ref_image_1, ref_image_2, ref_image_3, ref_image_4],
            labels=True,
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
            print(f"[API Toolkit Gemini Vision] {preview}")
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
                    "default": "imagen-4.0-generate-001",
                    "tooltip": "Gemini/Imagen model for image generation.",
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
                    "tooltip": "Number of image candidates to generate (returns first).",
                }),
                "ref_image_1": ("IMAGE", {"tooltip": "First reference image."}),
                "ref_image_2": ("IMAGE", {"tooltip": "Second reference image."}),
                "ref_image_3": ("IMAGE", {"tooltip": "Third reference image."}),
                "ref_image_4": ("IMAGE", {"tooltip": "Fourth reference image."}),
                "safety_settings_json": ("STRING", {
                    "default": "",
                    "tooltip": "JSON safety settings from Safety Settings node.",
                }),
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
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key)

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
            return _extract_image_from_stream(client, final_model, contents, config)

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
                "mask": ("IMAGE", {
                    "tooltip": "Optional mask indicating which areas to edit (white = edit).",
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
        system_instruction="",
        aspect_ratio="AUTO",
        image_size="AUTO",
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key)

        parts = []

        # Source image
        img_bytes = tensor_to_jpeg_bytes(image, quality=95)
        parts.append(types.Part.from_text(text="--- [Source Image] ---"))
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        # Optional mask
        if mask is not None:
            mask_bytes = tensor_to_jpeg_bytes(mask, quality=95)
            parts.append(types.Part.from_text(text="--- [Edit Mask (white = edit area)] ---"))
            parts.append(types.Part.from_bytes(data=mask_bytes, mime_type="image/jpeg"))

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
                "mask": ("IMAGE", {
                    "tooltip": "Mask image (white = area to fill).",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describe what should fill the masked area.",
                }),
            },
            "optional": {
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System instruction for the inpainting model.",
                }),
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
        system_instruction="",
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key)

        parts = []

        # Source image
        img_bytes = tensor_to_jpeg_bytes(image, quality=95)
        parts.append(types.Part.from_text(text="--- [Source Image] ---"))
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        # Mask
        mask_bytes = tensor_to_jpeg_bytes(mask, quality=95)
        parts.append(types.Part.from_text(text="--- [Inpaint Mask (white = fill area)] ---"))
        parts.append(types.Part.from_bytes(data=mask_bytes, mime_type="image/jpeg"))

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
    ):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key)

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
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("models_text", "models_json")
    FUNCTION = "list_models"
    CATEGORY = "NanoBanana2/Config"

    def list_models(self, api_key, filter="all"):
        key = get_api_key(api_key)
        client = get_client(key)

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
                    "default": 0, "min": 0, "max": 2147483647,
                    "tooltip": "Seed for reproducibility. 0 = random.",
                }),
                "safety_filter_level": (
                    ["block_low_and_above"],
                    {"default": "block_low_and_above",
                     "tooltip": "Safety filter threshold. The Developer API only accepts block_low_and_above. (Vertex AI supports more levels.)"}
                ),
                "person_generation": (
                    ["dont_allow", "allow_adult", "allow_all"],
                    {"default": "allow_adult",
                     "tooltip": "Whether/how to generate people. Some models enforce stricter defaults."}
                ),
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
        seed=0,
        safety_filter_level="block_medium_and_above",
        person_generation="allow_adult",
    ):
        from google.genai import types
        import torch
        import numpy as np

        key = get_api_key(api_key)
        final_model = _resolve_model(model, custom_model)
        client = get_client(key)

        cfg_kwargs = {
            "number_of_images": number_of_images,
            "aspect_ratio": aspect_ratio,
            "safety_filter_level": safety_filter_level,
            "person_generation": person_generation,
        }
        if negative_prompt and negative_prompt.strip():
            cfg_kwargs["negative_prompt"] = negative_prompt.strip()
        if seed > 0:
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

        # Concatenate into a batch
        batch = torch.cat(tensors, dim=0)
        return (batch,)


# ===================================================================
# NODE MAPPINGS
# ===================================================================

NODE_CLASS_MAPPINGS = {
    # Config (5)
    "NanoBanana_APIKey": NanoBanana_APIKey,
    "NanoBanana_ModelSelector": NanoBanana_ModelSelector,
    "NanoBanana_SafetySettings": NanoBanana_SafetySettings,
    "NanoBanana_ThinkingConfig": NanoBanana_ThinkingConfig,
    "NanoBanana_ListModels": NanoBanana_ListModels,
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Config
    "NanoBanana_APIKey": "NanoBanana - API Key",
    "NanoBanana_ModelSelector": "NanoBanana - Model Selector",
    "NanoBanana_SafetySettings": "NanoBanana - Safety Settings",
    "NanoBanana_ThinkingConfig": "NanoBanana - Thinking Config",
    "NanoBanana_ListModels": "NanoBanana - List Available Models",
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
}
