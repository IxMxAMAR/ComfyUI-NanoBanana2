# ComfyUI-NanoBanana2

Yes, the name is ridiculous. No, we're not changing it.

"NanoBanana" is a community nickname for Google's Gemini image generation models. This started as a humble 3-node Gemini image generator. It has since gotten completely out of hand. We are now at **32 nodes** covering text, vision, image generation (two different endpoints), audio, music, video, embeddings, file upload + reuse, Google Search grounding, code execution, multi-speaker TTS, audio transcription, OCR, and cost estimation. At some point this stopped being a ComfyUI node pack and became a full Gemini SDK replacement in node-graph form.

The name still fits, somehow.

Also available as part of [ComfyUI-API-Toolkit](https://github.com/IxMxAMAR/ComfyUI-API-Toolkit) alongside other API integrations.

**v2.7** — catches up with the current Gemini API: Gemini 3.x text models, GA Nano Banana image models, Gemini 3.8 TTS, Lyria 3.5, Gemini Omni video and `gemini-3.5-transcribe`, plus a URL Context node. Models Google has shut down were removed. See [CHANGELOG.md](CHANGELOG.md).

---

## Installation

**ComfyUI Manager** (recommended)

Search for `NanoBanana2` in the ComfyUI Manager and install.

**Registry**

```
comfy node registry-install nanobanana2
```

**Manual**

```bash
git clone https://github.com/IxMxAMAR/ComfyUI-NanoBanana2
pip install "google-genai>=2.3.0"
```

---

## Getting an API Key

Go to [aistudio.google.com](https://aistudio.google.com), hit "Get API Key", copy it. That's it.

Paste it into the API Key node (password-masked) or set the `GEMINI_API_KEY` environment variable and the node will pick it up automatically.

---

## Nodes

### Config (6 nodes)

The connective tissue. Wire these into your generative nodes as needed.

| Node | What it does |
|---|---|
| **API Key** | Password-masked key input. Reads `GEMINI_API_KEY` env var if left empty. |
| **Model Selector** | Pick from text, image, or all models. Includes a custom override field for whatever Google released last Tuesday. |
| **Safety Settings** | Per-category harm thresholds. For when the defaults are either too strict or not strict enough for your workflow. |
| **Thinking Config** | Set thinking level and token budget. Defaults to NONE -- thinking is opt-in because it costs more and you probably don't need it for a caption node. |
| **List Available Models** | Queries your API key and returns what's actually accessible on your account. Useful when you're not sure if you have access to a preview model. |
| **Token Counter** | Count tokens for a given prompt before you burn them. Feed it your text and optional images, get back a number. |

### Text (4 nodes)

| Node | What it does |
|---|---|
| **Text Generation** | Full-parameter text gen: temperature, top_p, top_k, thinking, seed. 34 model options including latest aliases, Gemini 3.1 to 3.8, 2.5, Gemma, and specialized models. If it's a knob, it's exposed. Google deprecated temperature, top_p and top_k on Gemini 3.x models. |
| **Prompt Refiner** | Feed it a rough prompt, get back a polished one. Useful before hitting your image nodes. |
| **Multi-Turn Chat** | Stateful conversation node. Maintains message history across runs. |
| **Structured Output** | JSON schema-constrained generation. Tell it exactly what shape of data you want back. |

### Image (6 nodes)

Two different endpoints, both covered.

| Node | What it does |
|---|---|
| **Vision Analysis** | Describe, analyze, or interrogate an image. Good for captioning or feeding into a downstream prompt. `media_resolution` sets how many tokens each image gets. |
| **Image Generation** | Generate images via `generate_content`. Supports Nano Banana, Nano Banana 2, Nano Banana 2 Lite, and Nano Banana Pro. Up to 4 reference images, full aspect ratio selection (including 1:4, 4:1, 1:8, 8:1), 512 / 1K / 2K / 4K output, optional Google Search grounding (web and image search), seed control. |
| **Imagen Image Generation** | Imagen 4 node using the `generate_images` endpoint. Google scheduled every Imagen 4 model for shutdown on 2026-08-17; use Image Generation with `gemini-3.1-flash-image` instead. |
| **Image Edit** | Text-guided image editing. Describe what you want changed. |
| **Inpaint** | Mask-based inpainting. Feed it an image and a mask, tell it what should be there. |
| **Outpaint** | Extend an image outward. Choose your expansion direction. |

### Audio (2 nodes)

| Node | What it does |
|---|---|
| **Text-to-Speech** | Convert text to speech using Gemini TTS models, including Gemini 3.8 Flash TTS and Flash-Lite TTS. 30 prebuilt voices: Zephyr, Puck, Kore, Charon, and more. On 3.8 models `style_prompt` becomes a speech style and `custom_voice` takes extended-library voices or `voice_...` / `voicekey_...` IDs. |
| **Music Generation** | Generate music via Lyria 3.5 (full-length songs) or Lyria 3 Clip / Pro. Text prompt (with optional lyrics) and up to 10 reference images in, audio out. Uses the Interactions API. |

### Video (2 nodes)

| Node | What it does |
|---|---|
| **Video Generation (Veo)** | Text-to-video, image-to-video and first/last-frame video via Veo 3.1 (including fast and lite), with 720p / 1080p / 4k output. Uses `predictLongRunning` under the hood because video takes a minute. Google scheduled Veo 3.1 for shutdown on 2026-10-22. |
| **Video Generation (Gemini Omni)** | Text-to-video, image-to-video, first/last-frame and subject-reference video with `gemini-omni-1.1-flash` (native audio, 360p to 4k). Edit or extend by passing a Files API video URI, or chain `interaction_id` into `previous_interaction_id` for conversational edits. Uses the Interactions API. |

### Embeddings (2 nodes)

| Node | What it does |
|---|---|
| **Text Embeddings** | Generate text embeddings at 768 to 3072 dimensions. Includes task-type optimization for retrieval, clustering, classification, and semantic similarity use cases. |
| **Save Embedding (.npy)** | Write the embedding vector to disk as a NumPy `.npy` for downstream vector DBs / similarity search. Basename-sanitized. |

### Files (2 nodes) — new in v2.1

| Node | What it does |
|---|---|
| **Files Upload** | Upload a local PDF / video / audio / image to the Gemini Files API. Returns a reusable URI (lives ~48h server-side) so you don't have to re-encode big files into every prompt. |
| **Ask Uploaded File** | Pair with Files Upload. Send the URI + mime_type + a question, get text back. Supports PDF up to 1000 pages, video / audio up to 1 GiB. |

### Tool-using Text (3 nodes)

| Node | What it does |
|---|---|
| **Text Gen + Google Search** | TextGen with the GoogleSearch grounding tool wired in. Returns the answer plus a `citations_json` list of `(url, title)` sources the model used. For current events, recent product info, etc. |
| **Text Gen + Code Execution** | TextGen with the ToolCodeExecution tool. Returns three strings: the final answer, any Python the model ran, and the execution output. Math, stats, plotting, JSON wrangling. |
| **Text Gen + URL Context** | The model reads the web pages linked in the prompt (up to 20 URLs). Optional Google Search on top. Uses the Interactions API. |

### Vision + Audio additions (2 nodes) — new in v2.1

| Node | What it does |
|---|---|
| **Vision OCR (lossless PNG)** | OCR-tuned Vision node. Three modes: `plain_text`, `structured_json` (returns line-level bboxes), or `markdown`. Always-PNG encoding so small text isn't smeared by JPEG. |
| **TTS Multi-Speaker Dialogue** | Two-voice TTS. Write `Alice: Hi Bob.\nBob: Hi Alice.` and assign each speaker a voice. Works with the 2.5 / 3.1 models and Gemini 3.8 TTS. |
| **Audio Transcribe** | Feed a ComfyUI `AUDIO`, get a transcript back. Optional `[HH:MM:SS]` timestamps. Also supports `gemini-3.5-transcribe` with language codes, custom vocabulary, speaker diarization and SMART (disfluency-free) mode. |

### Utilities

| Node | What it does |
|---|---|
| **Network Route** | Produces a proxy config (`NB_NETWORK`) that every other node accepts, so a workflow's Gemini calls can leave through a chosen region. Can probe the egress IP. |
| **Cost Estimator** | Counts input tokens via the free `count_tokens` API, multiplies by a model price table + your output-token estimate + a run multiplier, returns total USD plus a breakdown string. Pre-flight expensive batched workloads. |

---

## Supported Models

### generateContent (text and multimodal)

**Gemini 3 (stable)**
- `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`
- `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`

**Previews**
- `gemini-3-flash-preview`
- `gemini-3.1-pro-preview`, `gemini-3.1-pro-preview-customtools`

**Gemini 2.5** (not deprecated, but Google only serves them to projects that used them before)
- `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`

**Latest aliases**
- `gemini-pro-latest`, `gemini-flash-latest`, `gemini-flash-lite-latest`

**Gemma**
- `gemma-3-1b-it`, `gemma-3-4b-it`, `gemma-3-12b-it`, `gemma-3-27b-it`
- `gemma-3n-e2b-it`, `gemma-3n-e4b-it`
- `gemma-4-26b-a4b-it`, `gemma-4-31b-it`

**Specialized**
- `gemini-robotics-er-2-preview`
- `deep-research-pro-preview-12-2025`, `deep-research-preview-04-2026`, `deep-research-max-preview-04-2026`
- `nano-banana-pro-preview`

### generateContent (image output)

- `gemini-3.1-flash-image` -- Nano Banana 2
- `gemini-3.1-flash-lite-image` -- Nano Banana 2 Lite (1K only)
- `gemini-3-pro-image` -- Nano Banana Pro
- `gemini-2.5-flash-image` -- Nano Banana (scheduled for shutdown 2026-10-02)

### predict (Imagen)

- `imagen-4.0-ultra-generate-001`, `imagen-4.0-generate-001`, `imagen-4.0-fast-generate-001` (all scheduled for shutdown 2026-08-17)

### Text-to-speech

- `gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts` (Interactions API)
- `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`, `gemini-3.1-flash-tts-preview` (generateContent)
- 30 prebuilt voices, plus extended-library and custom voices on the 3.8 models

### Speech-to-text

- `gemini-3.5-transcribe` (generateContent with `audio_transcription_config`)

### Interactions API (Lyria music)

- `lyria-3.5`, `lyria-3-pro-preview`, `lyria-3-clip-preview`

### Video

- `gemini-omni-1.1-flash` (Interactions API)
- `veo-3.1-generate-preview`, `veo-3.1-fast-generate-preview`, `veo-3.1-lite-generate-preview` (`predictLongRunning`, scheduled for shutdown 2026-10-22)

### embedContent (embeddings)

- `gemini-embedding-001`
- `gemini-embedding-2`, `gemini-embedding-2-preview` (no `task_type`)

All nodes also have a **custom_model** override field. When Google drops something new you can use it immediately without waiting for a package update.

---

## Aspect Ratios

1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, plus 1:4, 4:1, 1:8, 8:1 on Nano Banana 2 and 2 Lite.

---

## Technical Notes

A few things that were done deliberately:

- **IS_CHANGED on all nodes** -- every node re-executes on every run even with identical inputs. Generative nodes should be generative.
- **Retry with jittered exponential backoff** -- transient API errors are retried with random jitter to avoid thundering-herd retries when multiple workers hit a 429 wave.
- **Bounded, hashed client cache** -- LRU cap of 16 clients, keyed by SHA-256 of the API key (the raw key never lives in a dict key). Rotating keys across workflows can't OOM the worker.
- **Full chunk iteration** -- responses scan all parts, not just `parts[0]`. You won't silently lose content from multi-part responses.
- **Image candidates returned as a batch** -- ImageGen with `candidate_count > 1` returns ALL images as a batched IMAGE tensor (previously you paid for 4, got 1).
- **Mask auto-resize** -- Inpaint / ImageEdit silently match mask dims to image dims (Gemini 400s on mismatch).
- **Lossless PNG option** -- Vision node has a `lossless` toggle for OCR / small-text / fine-grained tasks; new VisionOCR node uses it by default.
- **API-key redaction** -- every error string surfaced to the UI / logs is run through a regex that strips `AIza[...]` keys and `x-goog-api-key:` / `Authorization:` headers.
- **URL-path injection guarded** -- custom Lyria / Omni / Veo model IDs are validated against `^[A-Za-z0-9._-]+$` so a malicious `../other_endpoint` can't escape the `/models/` path.
- **SSRF guards** -- Veo only downloads from `*.googleapis.com` / `*.googleusercontent.com`; `download_file` defaults `allow_redirects=False` and enforces a 256 MiB cap by default.
- **Safety refusals are descriptive** -- when Gemini refuses a request, the error tells you which category triggered it and surfaces the model's own explanation.
- **Tooltips everywhere** -- hover over any input for a description of what it does.
- **Password-masked API keys** -- the key input field is masked.
- **Environment variable fallback** -- set `GEMINI_API_KEY` and all nodes pick it up automatically. Quote stripping handles the common `GEMINI_API_KEY="AIza..."` `.env` mistake.

---

## Requirements

- Python 3.10+
- `google-genai >= 2.3.0`
- A Google AI Studio API key

## Tests

```
python -m pytest tests/
```

111 tests, no network. Covers secret redaction, model-ID sanitization,
mask/image conversions, retry control flow, response size caps, every
node's registration, and request shapes for the Interactions API nodes
(Lyria, Gemini 3.8 TTS, Omni video, URL Context) against fake clients.

---

## License

MIT

---

Made by [IxMxAMAR](https://github.com/IxMxAMAR)
