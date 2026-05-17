# ComfyUI-NanoBanana2

Yes, the name is ridiculous. No, we're not changing it.

"NanoBanana" is a community nickname for Google's Gemini image generation models. This started as a humble 3-node Gemini image generator. It has since gotten completely out of hand. We are now at **29 nodes** covering text, vision, image generation (two different endpoints), audio, music, video, embeddings, file upload + reuse, Google Search grounding, code execution, multi-speaker TTS, audio transcription, OCR, and cost estimation. At some point this stopped being a ComfyUI node pack and became a full Gemini SDK replacement in node-graph form.

The name still fits, somehow.

Also available as part of [ComfyUI-API-Toolkit](https://github.com/IxMxAMAR/ComfyUI-API-Toolkit) alongside other API integrations.

**v2.1.0** — audit-driven release. Two Gemini Pro code reviews + 64 tests fixed 10 verified bugs, hardened secret handling / URL injection / SSRF / cache memory growth, and added 9 new nodes. See [CHANGELOG.md](CHANGELOG.md).

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
pip install google-genai
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
| **Text Generation** | Full-parameter text gen: temperature, top_p, top_k, thinking, seed. 33 model options including latest aliases, Gemini 3 previews, 2.5/2.0 stable, Gemma, and specialized models. If it's a knob, it's exposed. |
| **Prompt Refiner** | Feed it a rough prompt, get back a polished one. Useful before hitting your image nodes. |
| **Multi-Turn Chat** | Stateful conversation node. Maintains message history across runs. |
| **Structured Output** | JSON schema-constrained generation. Tell it exactly what shape of data you want back. |

### Image (6 nodes)

Two different endpoints, both covered.

| Node | What it does |
|---|---|
| **Vision Analysis** | Describe, analyze, or interrogate an image. Good for captioning or feeding into a downstream prompt. |
| **Image Generation** | Generate images via `generate_content`. Supports Nano Banana, Nano Banana 2, and Nano Banana Pro. Up to 4 reference images, full aspect ratio selection, seed control. |
| **Imagen Image Generation** | Dedicated Imagen 4 node using the `generate_images` endpoint. Ultra, Standard, and Fast variants. Different endpoint, different characteristics, same node graph. |
| **Image Edit** | Text-guided image editing. Describe what you want changed. |
| **Inpaint** | Mask-based inpainting. Feed it an image and a mask, tell it what should be there. |
| **Outpaint** | Extend an image outward. Choose your expansion direction. |

### Audio (2 nodes)

| Node | What it does |
|---|---|
| **Text-to-Speech** | Convert text to speech using Gemini TTS models. 30+ prebuilt voices: Zephyr, Puck, Kore, Charon, and more. Flash and Pro variants. |
| **Music Generation** | Generate music clips via Lyria 3. Clip and Pro model variants. Text prompt in, audio out. |

### Video (1 node)

| Node | What it does |
|---|---|
| **Video Generation** | Text-to-video and image-to-video via Veo. Supports Veo 3.1 (including fast and lite), Veo 3.0, and Veo 2.0. Uses `predictLongRunning` under the hood because video takes a minute. |

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

### Tool-using Text (2 nodes) — new in v2.1

| Node | What it does |
|---|---|
| **Text Gen + Google Search** | TextGen with the GoogleSearch grounding tool wired in. Returns the answer plus a `citations_json` list of `(url, title)` sources the model used. For current events, recent product info, etc. |
| **Text Gen + Code Execution** | TextGen with the ToolCodeExecution tool. Returns three strings: the final answer, any Python the model ran, and the execution output. Math, stats, plotting, JSON wrangling. |

### Vision + Audio additions (2 nodes) — new in v2.1

| Node | What it does |
|---|---|
| **Vision OCR (lossless PNG)** | OCR-tuned Vision node. Three modes: `plain_text`, `structured_json` (returns line-level bboxes), or `markdown`. Always-PNG encoding so small text isn't smeared by JPEG. |
| **TTS Multi-Speaker Dialogue** | Proper two-voice TTS using `MultiSpeakerVoiceConfig`. Write `Alice: Hi Bob.\nBob: Hi Alice.` and assign each speaker a voice. |
| **Audio Transcribe** | Feed a ComfyUI `AUDIO`, get a transcript back. Optional `[HH:MM:SS]` timestamps. |

### Utilities (new in v2.1)

| Node | What it does |
|---|---|
| **Cost Estimator** | Counts input tokens via the free `count_tokens` API, multiplies by a model price table + your output-token estimate + a run multiplier, returns total USD plus a breakdown string. Pre-flight expensive batched workloads. |

---

## Supported Models

### generateContent (text and multimodal)

**Latest aliases**
- `gemini-pro-latest`, `gemini-flash-latest`, `gemini-flash-lite-latest`

**Previews**
- `gemini-3-pro-preview`, `gemini-3-flash-preview`
- `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`

**Stable**
- `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- `gemini-2.0-flash`, `gemini-2.0-flash-lite`, `gemini-2.0-flash-001`, `gemini-2.0-flash-lite-001`

**Gemma**
- `gemma-3-1b-it`, `gemma-3-4b-it`, `gemma-3-12b-it`, `gemma-3-27b-it`
- `gemma-3n-e2b-it`, `gemma-3n-e4b-it`
- `gemma-4-26b-a4b-it`, `gemma-4-31b-it`

**Specialized**
- `gemini-robotics-er-1.5-preview`, `gemini-robotics-er-1.6-preview`
- `gemini-2.5-computer-use-preview`
- `deep-research-pro-preview`
- `nano-banana-pro-preview`

### generateContent (image output)

- `gemini-3.1-flash-image-preview` -- Nano Banana 2
- `gemini-3-pro-image-preview` -- Nano Banana Pro
- `gemini-2.5-flash-image` -- Nano Banana

### predict (Imagen)

- `imagen-4.0-ultra-generate-001`
- `imagen-4.0-generate-001`
- `imagen-4.0-fast-generate-001`

### generateContent (TTS audio output)

- `gemini-2.5-flash-preview-tts`
- `gemini-2.5-pro-preview-tts`
- `gemini-3.1-flash-tts-preview`
- 30+ prebuilt voices

### predict (Lyria music)

- `lyria-3-pro-preview`
- `lyria-3-clip-preview`

### predictLongRunning (Veo video)

- `veo-3.1-generate-preview`, `veo-3.1-generate-fast`, `veo-3.1-generate-lite`
- `veo-3.0-generate-001`, `veo-3.0-generate-fast`
- `veo-2.0-generate-001`

### embedContent (embeddings)

- `gemini-embedding-001`
- `gemini-embedding-2-preview`

All nodes also have a **custom_model** override field. When Google drops something new you can use it immediately without waiting for a package update.

---

## Aspect Ratios

1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9

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
- **URL-path injection guarded** -- custom Lyria / Veo model IDs are validated against `^[A-Za-z0-9._-]+$` so a malicious `../other_endpoint` can't escape the `/models/` path.
- **SSRF guards** -- Veo only downloads from `*.googleapis.com` / `*.googleusercontent.com`; `download_file` defaults `allow_redirects=False` and enforces a 256 MiB cap by default.
- **Safety refusals are descriptive** -- when Gemini refuses a request, the error tells you which category triggered it and surfaces the model's own explanation.
- **Tooltips everywhere** -- hover over any input for a description of what it does.
- **Password-masked API keys** -- the key input field is masked.
- **Environment variable fallback** -- set `GEMINI_API_KEY` and all nodes pick it up automatically. Quote stripping handles the common `GEMINI_API_KEY="AIza..."` `.env` mistake.

---

## Requirements

- Python 3.10+
- `google-genai >= 0.8.0`
- A Google AI Studio API key

## Tests

```
C:/ComfyUI/venv/Scripts/python -m pytest tests/
```

64 tests, no network. Covers secret redaction, model-ID sanitization,
mask/image conversions, retry control flow, response size caps, every
new node's registration, and regressions on each verified v2.0 bug.

---

## License

MIT

---

Made by [IxMxAMAR](https://github.com/IxMxAMAR)
