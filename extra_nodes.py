"""ComfyUI nodes added in NanoBanana2 v2.1.

Eight new nodes informed by parallel Gemini Pro code reviews. Each new
node fills a real gap in the v2.0 coverage:

 - NanoBanana_FilesUpload      — upload reusable file handles (PDFs, video, audio)
 - NanoBanana_VisionWithFile   — ask a question about a previously uploaded file
 - NanoBanana_TextGenSearch    — TextGen + Google Search grounding tool
 - NanoBanana_TextGenCode      — TextGen + code execution tool
 - NanoBanana_TTSMultiSpeaker  — multi-voice dialogue (uses MultiSpeakerVoiceConfig)
 - NanoBanana_AudioTranscribe  — transcribe an AUDIO input via Gemini multimodal
 - NanoBanana_EmbedSave        — save embedding vector as .npy
 - NanoBanana_VisionOCR        — preset Vision node tuned for OCR / fine text
 - NanoBanana_CostEstimate     — token-based cost estimate for prompts/images
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

try:
    from .shared.node_utils import AlwaysExecuteMixin
    from .shared.conversions import (
        tensor_to_jpeg_bytes, tensor_to_png_bytes, comfy_to_audio_bytes,
    )
    from .gemini_client import (
        get_client, get_api_key, retry_with_backoff,
        TEXT_MODELS, TTS_MODELS, TTS_VOICES, EMBEDDING_MODELS,
        THINKING_LEVELS,
    )
except ImportError:
    from shared.node_utils import AlwaysExecuteMixin
    from shared.conversions import (
        tensor_to_jpeg_bytes, tensor_to_png_bytes, comfy_to_audio_bytes,
    )
    from gemini_client import (
        get_client, get_api_key, retry_with_backoff,
        TEXT_MODELS, TTS_MODELS, TTS_VOICES, EMBEDDING_MODELS,
        THINKING_LEVELS,
    )


# ===================================================================
# Files API — upload once, reuse across prompts
# ===================================================================

class NanoBanana_FilesUpload(AlwaysExecuteMixin):
    """Upload a local file to Gemini Files API; returns a reusable handle.

    Wire the output GEMINI_FILE handle into VisionWithFile / chat nodes to
    avoid re-uploading the same PDF / video / audio on every query. Files
    live for ~48h then expire (server-side).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to the file (PDF, video, audio, image).",
                }),
            },
            "optional": {
                "display_name": ("STRING", {
                    "default": "",
                    "tooltip": "Optional human-readable name for the upload.",
                }),
                "mime_type": ("STRING", {
                    "default": "",
                    "tooltip": "Override the MIME type if auto-detection fails.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_uri", "file_name")
    FUNCTION = "upload"
    CATEGORY = "NanoBanana2/Files"

    def upload(self, api_key, file_path, display_name="", mime_type=""):
        key = get_api_key(api_key)
        client = get_client(key)

        if not file_path or not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path!r}")

        cfg = {}
        if display_name.strip():
            cfg["display_name"] = display_name.strip()
        if mime_type.strip():
            cfg["mime_type"] = mime_type.strip()

        def _call():
            f = client.files.upload(file=file_path, config=cfg or None)
            return f

        f = retry_with_backoff(_call)
        uri = getattr(f, "uri", "") or ""
        name = getattr(f, "name", "") or ""
        print(f"[NanoBanana Files] Uploaded {name} → {uri}")
        return (uri, name)


class NanoBanana_VisionWithFile(AlwaysExecuteMixin):
    """Ask a question about a previously-uploaded file (PDF, video, audio, image).

    Pair with FilesUpload. The Files API supports large files (PDFs up to
    1000 pages, audio/video up to 1 GiB) that won't fit in an inline
    multimodal prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (TEXT_MODELS, {"default": "gemini-2.5-flash"}),
                "file_uri": ("STRING", {
                    "default": "",
                    "tooltip": "URI from NanoBanana Files Upload.",
                }),
                "mime_type": ("STRING", {
                    "default": "application/pdf",
                    "tooltip": "MIME type of the uploaded file (application/pdf, video/mp4, audio/wav, image/png, ...).",
                }),
                "prompt": ("STRING", {
                    "multiline": True, "default": "Summarize this document.",
                }),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "thinking_level": (THINKING_LEVELS, {"default": "NONE"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "ask"
    CATEGORY = "NanoBanana2/Files"

    def ask(self, api_key, model, file_uri, mime_type, prompt,
            custom_model="", temperature=0.1, thinking_level="NONE"):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = custom_model.strip() if custom_model and custom_model.strip() else model
        client = get_client(key)

        if not file_uri.strip():
            raise ValueError("file_uri is empty — upload via NanoBanana Files Upload first.")

        cfg_kwargs = {"temperature": temperature}
        if thinking_level != "NONE":
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        config = types.GenerateContentConfig(**cfg_kwargs)

        contents = [
            types.Content(role="user", parts=[
                types.Part.from_uri(file_uri=file_uri, mime_type=mime_type),
                types.Part.from_text(text=prompt),
            ])
        ]

        def _call():
            r = client.models.generate_content(
                model=final_model, contents=contents, config=config,
            )
            return r.text.strip() if r.text else ""

        return (retry_with_backoff(_call),)


# ===================================================================
# Search Grounding + Code Execution
# ===================================================================

class NanoBanana_TextGenSearch(AlwaysExecuteMixin):
    """TextGen with Google Search grounding (lets the model fetch fresh facts).

    Use for current-events questions, recent product info, anything where
    training-cutoff stale data would mislead the model. Returns the text
    answer and a `citations_json` list of (url, title) tuples the model used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (TEXT_MODELS, {"default": "gemini-2.5-flash"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "citations_json")
    FUNCTION = "generate"
    CATEGORY = "NanoBanana2/Text"

    def generate(self, api_key, model, prompt, custom_model="",
                 system_instruction="", temperature=0.3):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = custom_model.strip() if custom_model and custom_model.strip() else model
        client = get_client(key)

        cfg_kwargs = {
            "tools": [types.Tool(google_search=types.GoogleSearch())],
            "temperature": temperature,
        }
        if system_instruction.strip():
            cfg_kwargs["system_instruction"] = [types.Part.from_text(text=system_instruction.strip())]
        config = types.GenerateContentConfig(**cfg_kwargs)

        def _call():
            return client.models.generate_content(
                model=final_model, contents=prompt, config=config,
            )

        response = retry_with_backoff(_call)
        text = response.text.strip() if response.text else ""

        # Extract grounding citations
        citations = []
        for cand in response.candidates or []:
            gm = getattr(cand, "grounding_metadata", None)
            if not gm:
                continue
            for chunk in getattr(gm, "grounding_chunks", []) or []:
                web = getattr(chunk, "web", None)
                if web:
                    citations.append({
                        "url": getattr(web, "uri", ""),
                        "title": getattr(web, "title", ""),
                    })

        return (text, json.dumps(citations, indent=2))


class NanoBanana_TextGenCode(AlwaysExecuteMixin):
    """TextGen with code execution tool — the model can run Python and return results.

    Useful for math, data analysis, plotting (returns text + executable_code +
    code_execution_result). The `code_output` STRING contains the model's
    final text answer; `executed_code` contains any Python it actually ran.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (TEXT_MODELS, {"default": "gemini-2.5-pro"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "executed_code", "exec_result")
    FUNCTION = "generate"
    CATEGORY = "NanoBanana2/Text"

    def generate(self, api_key, model, prompt, custom_model="", temperature=0.1):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = custom_model.strip() if custom_model and custom_model.strip() else model
        client = get_client(key)

        # ToolCodeExecution is the correct type name in google-genai 1.x
        # (older name "CodeExecution" was removed).
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())],
            temperature=temperature,
        )

        def _call():
            return client.models.generate_content(
                model=final_model, contents=prompt, config=config,
            )

        response = retry_with_backoff(_call)
        text = response.text.strip() if response.text else ""

        # Pull executed code and result by walking the parts
        codes, results = [], []
        for cand in response.candidates or []:
            for part in (cand.content.parts or []):
                ec = getattr(part, "executable_code", None)
                if ec:
                    codes.append(getattr(ec, "code", "") or "")
                cer = getattr(part, "code_execution_result", None)
                if cer:
                    results.append(getattr(cer, "output", "") or "")

        return (text, "\n---\n".join(codes), "\n---\n".join(results))


# ===================================================================
# Multi-Speaker TTS — v2.0 only supported single voice
# ===================================================================

class NanoBanana_TTSMultiSpeaker(AlwaysExecuteMixin):
    """Multi-speaker TTS using MultiSpeakerVoiceConfig (v2.0 was single-voice only).

    Pass dialogue with speaker tags like:
        Alice: Hi Bob, how are you?
        Bob: I'm great, thanks!
    and supply Alice/Bob voice names. Up to 2 speakers per request.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (TTS_MODELS, {"default": "gemini-2.5-flash-preview-tts"}),
                "dialogue": ("STRING", {
                    "multiline": True,
                    "default": "Alice: Hi Bob, how are you?\nBob: I'm great, thanks!",
                    "tooltip": "Plain-text dialogue with `Speaker:` prefixes.",
                }),
                "speaker_1_name": ("STRING", {"default": "Alice"}),
                "speaker_1_voice": (TTS_VOICES, {"default": "Kore"}),
                "speaker_2_name": ("STRING", {"default": "Bob"}),
                "speaker_2_voice": (TTS_VOICES, {"default": "Puck"}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "NanoBanana2/Audio"

    def generate(self, api_key, model, dialogue,
                 speaker_1_name, speaker_1_voice,
                 speaker_2_name, speaker_2_voice, custom_model=""):
        from google.genai import types
        import torch
        import numpy as np

        key = get_api_key(api_key)
        final_model = custom_model.strip() if custom_model and custom_model.strip() else model
        client = get_client(key)

        speaker_configs = [
            types.SpeakerVoiceConfig(
                speaker=speaker_1_name,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=speaker_1_voice)
                ),
            ),
            types.SpeakerVoiceConfig(
                speaker=speaker_2_name,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=speaker_2_voice)
                ),
            ),
        ]

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_configs,
                ),
            ),
        )

        def _call():
            return client.models.generate_content(
                model=final_model, contents=dialogue, config=config,
            )

        response = retry_with_backoff(_call)

        audio_bytes = None
        for cand in response.candidates or []:
            for part in (cand.content.parts or []):
                if part.inline_data and part.inline_data.data:
                    audio_bytes = part.inline_data.data
                    break
            if audio_bytes:
                break

        if audio_bytes is None:
            raise RuntimeError("Multi-speaker TTS returned no audio data.")

        sample_rate = 24000
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
        return ({"waveform": waveform, "sample_rate": sample_rate},)


# ===================================================================
# Audio Transcription
# ===================================================================

class NanoBanana_AudioTranscribe(AlwaysExecuteMixin):
    """Transcribe an AUDIO input via Gemini multimodal.

    Accepts ComfyUI AUDIO (waveform + sample_rate), encodes as WAV,
    sends to Gemini with a transcription prompt. Returns the transcript
    text. For long audio, upload via FilesUpload + VisionWithFile instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (TEXT_MODELS, {"default": "gemini-2.5-flash"}),
                "audio": ("AUDIO",),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Transcribe this audio verbatim. Include speaker labels if multiple speakers are present.",
                }),
                "include_timestamps": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcript",)
    FUNCTION = "transcribe"
    CATEGORY = "NanoBanana2/Audio"

    def transcribe(self, api_key, model, audio, custom_model="",
                   prompt="Transcribe this audio verbatim.",
                   include_timestamps=False):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = custom_model.strip() if custom_model and custom_model.strip() else model
        client = get_client(key)

        wav_bytes = comfy_to_audio_bytes(audio, fmt="wav")
        if include_timestamps:
            prompt = (
                "Transcribe this audio with timestamps in [HH:MM:SS] format at "
                "the start of each line.\n\n" + prompt
            )

        contents = [
            types.Content(role="user", parts=[
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                types.Part.from_text(text=prompt),
            ])
        ]

        def _call():
            r = client.models.generate_content(
                model=final_model, contents=contents,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return r.text.strip() if r.text else ""

        return (retry_with_backoff(_call),)


# ===================================================================
# Save embedding to .npy
# ===================================================================

class NanoBanana_EmbedSave(AlwaysExecuteMixin):
    """Save an embedding vector (JSON string from EmbedContent) to a .npy file.

    Useful for vector databases, similarity search, or downstream ML pipelines.
    Returns the absolute path of the written file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embedding_json": ("STRING", {
                    "default": "",
                    "tooltip": "JSON list of floats (output of NanoBanana Text Embeddings node).",
                }),
                "filename": ("STRING", {
                    "default": "embedding.npy",
                    "tooltip": "Filename (with or without .npy extension). Written to ComfyUI output dir.",
                }),
            },
            "optional": {
                "subdirectory": ("STRING", {
                    "default": "embeddings",
                    "tooltip": "Subdirectory under ComfyUI's output dir.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "NanoBanana2/Embeddings"
    OUTPUT_NODE = True

    def save(self, embedding_json, filename, subdirectory="embeddings"):
        import numpy as np
        try:
            import folder_paths
            base = folder_paths.get_output_directory()
        except ImportError:
            base = os.path.join(os.getcwd(), "output")

        vec = json.loads(embedding_json)
        if not isinstance(vec, list):
            raise ValueError("embedding_json must be a JSON list of floats")
        arr = np.asarray(vec, dtype=np.float32)

        # Reject path-traversal in filename / subdir (basename only).
        safe_name = os.path.basename(filename).strip()
        safe_sub = os.path.basename(subdirectory).strip()
        if not safe_name:
            raise ValueError("filename must not be empty")
        if not safe_name.endswith(".npy"):
            safe_name += ".npy"

        out_dir = os.path.join(base, safe_sub) if safe_sub else base
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, safe_name)
        np.save(out_path, arr)
        print(f"[NanoBanana EmbedSave] Saved {arr.shape} vector → {out_path}")
        return (out_path,)


# ===================================================================
# Vision OCR preset
# ===================================================================

class NanoBanana_VisionOCR(AlwaysExecuteMixin):
    """OCR-tuned Vision node: lossless PNG encoding + structured-output prompt.

    Use for receipts, screenshots, documents, anything with small text.
    The Vision node converts to JPEG-95 which loses small-character detail;
    this one always uses PNG and a system prompt tuned for verbatim
    extraction.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (TEXT_MODELS, {"default": "gemini-2.5-pro"}),
                "image": ("IMAGE",),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                "mode": (["plain_text", "structured_json", "markdown"], {
                    "default": "plain_text",
                    "tooltip": "Output format. structured_json returns "
                               "{lines: [{text, bbox, confidence}, ...]}.",
                }),
                "language_hint": ("STRING", {
                    "default": "",
                    "tooltip": "Optional language hint (e.g. 'Japanese', 'Hindi').",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "ocr"
    CATEGORY = "NanoBanana2/Image"

    def ocr(self, api_key, model, image, custom_model="",
            mode="plain_text", language_hint=""):
        from google.genai import types

        key = get_api_key(api_key)
        final_model = custom_model.strip() if custom_model and custom_model.strip() else model
        client = get_client(key)

        # Lossless PNG — small text in JPEG-95 loses ~30% character accuracy.
        img_bytes = tensor_to_png_bytes(image)

        if mode == "structured_json":
            instr = (
                "You are an OCR engine. Extract ALL visible text from the image "
                "and return a JSON object: "
                '{"lines": [{"text": "...", "bbox": [x1,y1,x2,y2]}]}. '
                "bbox coordinates are normalized 0..1. Output JSON ONLY, no prose."
            )
        elif mode == "markdown":
            instr = (
                "You are an OCR engine. Extract all visible text from the image "
                "and format it as Markdown, preserving headings, lists, tables, "
                "and code blocks. Output Markdown ONLY, no prose."
            )
        else:
            instr = (
                "You are an OCR engine. Extract ALL visible text from the image "
                "VERBATIM, preserving line breaks and approximate layout. "
                "Output the extracted text ONLY, no commentary."
            )
        if language_hint.strip():
            instr += f"\n\nLanguage hint: {language_hint.strip()}."

        contents = [
            types.Content(role="user", parts=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                types.Part.from_text(text="Extract all text."),
            ])
        ]

        cfg_kwargs = {
            "system_instruction": [types.Part.from_text(text=instr)],
            "temperature": 0.0,
        }
        if mode == "structured_json":
            cfg_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**cfg_kwargs)

        def _call():
            r = client.models.generate_content(
                model=final_model, contents=contents, config=config,
            )
            return r.text.strip() if r.text else ""

        return (retry_with_backoff(_call),)


# ===================================================================
# Cost Estimator — uses count_tokens (no charge) to estimate USD
# ===================================================================

# Per-model price hints. Numbers are USD-per-1M-tokens (Google AI pricing
# page, snapshot 2026-05-17). Adjust as needed — this is a rough estimator.
_PRICE_PER_MTOK = {
    # (input, output)
    "gemini-2.5-flash":        (0.30, 2.50),
    "gemini-2.5-flash-lite":   (0.10, 0.40),
    "gemini-2.5-pro":          (1.25, 10.00),
    "gemini-3.1-pro-preview":  (1.25, 10.00),
    "gemini-3-pro-preview":    (1.25, 10.00),
    "gemini-3.1-flash-lite-preview": (0.10, 0.40),
    "gemini-pro-latest":       (1.25, 10.00),
    "gemini-flash-latest":     (0.30, 2.50),
    "gemini-flash-lite-latest": (0.10, 0.40),
}


class NanoBanana_CostEstimate(AlwaysExecuteMixin):
    """Estimate USD cost of a prompt+expected output for a model.

    Counts input tokens via the (free) count_tokens API; combines with an
    estimated output token count and the model's price-per-million tokens.
    Useful to pre-flight expensive batched workloads.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (TEXT_MODELS, {"default": "gemini-2.5-flash"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                "expected_output_tokens": ("INT", {
                    "default": 500, "min": 0, "max": 65536,
                    "tooltip": "Your estimate of how many tokens the model will produce.",
                }),
                "runs": ("INT", {
                    "default": 1, "min": 1, "max": 1_000_000,
                    "tooltip": "Multiplier for batched workflows.",
                }),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("input_tokens", "estimated_usd", "breakdown")
    FUNCTION = "estimate"
    CATEGORY = "NanoBanana2/Config"

    def estimate(self, api_key, model, prompt, custom_model="",
                 expected_output_tokens=500, runs=1):
        key = get_api_key(api_key)
        final_model = custom_model.strip() if custom_model and custom_model.strip() else model
        client = get_client(key)

        resp = client.models.count_tokens(model=final_model, contents=prompt)
        in_tok = int(getattr(resp, "total_tokens", 0) or 0)

        prices = _PRICE_PER_MTOK.get(final_model)
        if prices is None:
            # Fallback: warn but estimate at flash rates so the output isn't 0.
            in_price, out_price = (0.30, 2.50)
            note = (f"WARNING: no price table entry for {final_model!r}. "
                    "Estimate uses gemini-2.5-flash rates as a fallback.")
        else:
            in_price, out_price = prices
            note = ""

        per_run = (in_tok * in_price + expected_output_tokens * out_price) / 1_000_000
        total = per_run * max(1, runs)

        breakdown = (
            f"Model: {final_model}\n"
            f"Input tokens (counted): {in_tok}\n"
            f"Expected output tokens: {expected_output_tokens}\n"
            f"Price (USD per 1M tokens): in={in_price}, out={out_price}\n"
            f"Per-run cost: ${per_run:.6f}\n"
            f"Total ({runs} run(s)): ${total:.4f}"
        )
        if note:
            breakdown = note + "\n" + breakdown

        return (in_tok, float(total), breakdown)


# ===================================================================
# Public mappings — merged into nanobanana_node's mappings at __init__
# ===================================================================

NEW_NODE_CLASS_MAPPINGS = {
    "NanoBanana_FilesUpload":     NanoBanana_FilesUpload,
    "NanoBanana_VisionWithFile":  NanoBanana_VisionWithFile,
    "NanoBanana_TextGenSearch":   NanoBanana_TextGenSearch,
    "NanoBanana_TextGenCode":     NanoBanana_TextGenCode,
    "NanoBanana_TTSMultiSpeaker": NanoBanana_TTSMultiSpeaker,
    "NanoBanana_AudioTranscribe": NanoBanana_AudioTranscribe,
    "NanoBanana_EmbedSave":       NanoBanana_EmbedSave,
    "NanoBanana_VisionOCR":       NanoBanana_VisionOCR,
    "NanoBanana_CostEstimate":    NanoBanana_CostEstimate,
}

NEW_NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana_FilesUpload":     "NanoBanana - Files Upload",
    "NanoBanana_VisionWithFile":  "NanoBanana - Ask Uploaded File (PDF / video / audio)",
    "NanoBanana_TextGenSearch":   "NanoBanana - Text Gen + Google Search",
    "NanoBanana_TextGenCode":     "NanoBanana - Text Gen + Code Execution",
    "NanoBanana_TTSMultiSpeaker": "NanoBanana - TTS Multi-Speaker Dialogue",
    "NanoBanana_AudioTranscribe": "NanoBanana - Audio Transcribe",
    "NanoBanana_EmbedSave":       "NanoBanana - Save Embedding (.npy)",
    "NanoBanana_VisionOCR":       "NanoBanana - Vision OCR (lossless PNG)",
    "NanoBanana_CostEstimate":    "NanoBanana - Cost Estimator",
}
