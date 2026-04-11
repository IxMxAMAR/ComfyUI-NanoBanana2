# ComfyUI-NanoBanana2

Yes, the name is ridiculous. No, we're not changing it.

"NanoBanana" is a community nickname for Google's Gemini image generation models, and honestly it fits. This started as a humble 3-node Gemini image generator, and has since ripened into a full 13-node suite covering text generation, vision, chat, structured output, image editing, inpainting, outpainting -- the whole banana.

**13 nodes. Full Google Gemini API. One delightful package name.**

Also available as part of [ComfyUI-API-Toolkit](https://github.com/IxMxAMAR/ComfyUI-API-Toolkit) alongside other API integrations.

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

You can either paste it directly into the API Key node (password-masked, don't worry) or set the `GEMINI_API_KEY` environment variable and the node will pick it up automatically.

---

## Nodes

### Config (4 nodes)

These are the building blocks. Wire them into your generative nodes as needed.

| Node | What it does |
|---|---|
| **API Key** | Password-masked key input. Reads `GEMINI_API_KEY` env var if left empty. |
| **Model Selector** | Pick from text, image, or all models. Includes a custom override field for whatever Google released last Tuesday. |
| **Safety Settings** | Per-category harm thresholds. For when the defaults are either too strict or not strict enough for your workflow. |
| **Thinking Config** | Set thinking level and token budget. Defaults to NONE -- thinking is opt-in because it costs more and you probably don't need it for a caption node. |

### Image (5 nodes)

| Node | What it does |
|---|---|
| **Image Generation** | Generate images from text. Supports up to 4 reference images, full aspect ratio selection, and seed control. |
| **Image Edit** | Text-guided image editing. Describe what you want changed, Gemini does the rest. |
| **Inpaint** | Mask-based inpainting. Feed it an image and a mask, tell it what should be there. |
| **Outpaint** | Extend an image outward. Choose your expansion direction. |
| **Vision Analysis** | Describe, analyze, or interrogate an image. Great for captioning or feeding into a downstream prompt. |

### Text (4 nodes)

| Node | What it does |
|---|---|
| **Text Generation** | Full-parameter text gen: temperature, top_p, top_k, thinking, seed. If it's a knob, it's exposed. |
| **Prompt Refiner** | Feed it a rough prompt, get back a polished one. Useful before hitting your image nodes. |
| **Multi-Turn Chat** | Stateful conversation node. Maintains message history across runs. |
| **Structured Output** | JSON schema-constrained generation. Tell it exactly what shape of data you want back. |

---

## Supported Models

### Text Models
- `gemini-3.1-pro`
- `gemini-3-pro`
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.0-flash`
- `gemini-3.1-flash-lite`

### Image Models
- `imagen-4.0-ultra`
- `imagen-4.0`
- `imagen-4.0-fast`
- `gemini-3.1-flash-image`
- `gemini-3-pro-image`
- `gemini-2.5-flash-image`

All Model Selector nodes also have a **custom_model** override field, so when Google drops something new you can use it without waiting for an update.

---

## Aspect Ratios

1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9

---

## Technical Notes

A few things that were done deliberately, in case you're wondering:

- **Retry with exponential backoff** -- transient API errors are retried automatically instead of blowing up your workflow.
- **Client caching** -- one connection per API key. If you're running multiple nodes with the same key, they share a client instead of spinning up new ones.
- **Full chunk iteration** -- responses scan ALL parts, not just `parts[0]`. You won't silently lose content from multi-part responses.
- **Safety refusals are descriptive** -- when Gemini refuses a request, the error tells you which safety category triggered it. You'll know why, not just that it failed.
- **Reference image errors are loud** -- if a reference image fails, it errors clearly rather than silently dropping it and generating something unexpected.
- **IS_CHANGED on all generative nodes** -- each run re-executes even with identical inputs. Generative nodes should be generative.
- **Tooltips everywhere** -- hover over any input for a description of what it does.

---

## Requirements

- Python 3.10+
- `google-genai >= 0.8.0`
- A Google AI Studio API key

---

## License

MIT

---

Made by [IxMxAMAR](https://github.com/IxMxAMAR)
