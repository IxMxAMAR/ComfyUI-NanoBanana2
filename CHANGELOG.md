# Changelog

All notable changes to ComfyUI-NanoBanana2 are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] — 2026-05-17

Audit-driven release. Two parallel Gemini 3.1 Pro reviews (one focused on
the API/HTTP/auth/retry layer, one on node correctness/UX) plus first-party
verification of every Critical and High finding produced 10 verified bug
fixes, four security hardenings, nine new feature nodes, and 64 tests.

### Security — fix immediately if you run shared workflows

- **API-key redaction on every error surface.** `gemini_client.redact_secret()`
  strips `AIza[...]` keys AND any echoed `x-goog-api-key` / `Authorization`
  headers before exception text is logged, printed, or raised to the
  ComfyUI UI. Wired into `retry_with_backoff`, the Lyria error path, and
  `_extract_image_from_stream`'s "model said..." breadcrumb. Previously a
  google-genai SDK exception whose `__str__` echoed the request URL would
  put the user's key directly into the workflow logs.

- **Lyria / Veo URL-path injection plug.** Both endpoints interpolate
  `custom_model` into a `:predict` URL on `generativelanguage.googleapis.com`.
  A `custom_model` like `"../other_endpoint"` could escape the `/models/`
  path. v2.1 validates every model ID against `^[A-Za-z0-9._-]+$` via
  `sanitize_model_id()` and raises a clear `ValueError` on anything else
  (no slashes, no `..`, no `?`, no `:`).

- **SSRF guard on Veo video download.** The SDK returns a `video_obj.uri`
  for completed Veo operations. Before downloading, v2.1 asserts the host
  ends with `.googleapis.com` / `.googleusercontent.com`. Refuses anything
  else, so a spoofed/compromised operation response cannot force the
  ComfyUI worker to download arbitrary URLs (potentially internal hosts
  like `169.254.169.254`).

- **Hardened key cache.** `_client_cache` is now an LRU bounded at 16
  entries and keyed by SHA-256 hash of the API key (the raw key never
  appears as a dict key). Previously keys-rotated-per-workflow could grow
  the cache unbounded, and keeping every key in a dict key indefinitely
  increased exposure to memory dumps.

- **`download_file` enforces `max_bytes`** (default 256 MiB). A
  misconfigured or hostile server can no longer stream an infinite body
  into the worker until OOM. Goes hand-in-hand with `allow_redirects=False`
  by default — set to `True` only when you trust the URL (closes the
  HTTP→`file://` / HTTP→internal-IP redirect SSRF surface).

- **New `stream_to_file()` helper** used by Veo: streams to disk with the
  same size cap, removes the partial file on any failure, and always
  closes the HTTP response via context manager. Veo v2.0 leaked the
  connection if `iter_content` threw mid-stream.

### Fixed — Critical correctness

- **`THINKING_LEVELS = ["NONE", "LOW", "NORMAL", "HIGH"]` → `["NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH"]`.**
  The actual `google.genai.types.ThinkingLevel` enum is
  `MINIMAL/LOW/MEDIUM/HIGH`. v2.0's `"NORMAL"` was silently rejected by
  the API; any node that picked `"NORMAL"` (anything besides text gen
  with default `NONE`) effectively sent an invalid `thinking_config`.

- **Imagen / Veo `seed=0` no longer silently dropped.** The condition was
  `if seed > 0` — seed 0 is a perfectly valid Gemini seed. v2.1 uses
  `seed >= 0` and switches the input UI to `default=-1, min=-1` so `-1`
  is the explicit "random" sentinel and 0..N are honored.

- **Inpaint / ImageEdit mask now auto-resizes to image dims.** Gemini
  returns 400 if mask H×W ≠ image H×W; ComfyUI MASK tensors from SAM,
  depth, alpha, etc. routinely come in at different sizes. New
  `resize_mask_to_image()` helper does a bilinear resize and clamps to
  [0,1]. PNG is used for mask + image so the edge is pixel-exact (JPEG
  smears the mask boundary ~2 px).

- **`ImageGen` returns ALL candidates, not just the first.** The
  `candidate_count` input now actually does what it says (returns a
  batched `IMAGE` tensor `[N, H, W, C]`). v2.0 happily charged you for
  4 candidates then discarded 2-4.

- **Imagen `safety_filter_level` dropdown matches the SDK enum.** v2.0's
  dropdown had a single value (`block_low_and_above`) and the method
  signature defaulted to a different value (`block_medium_and_above`) that
  wasn't in the dropdown. v2.1 exposes all four
  `block_{none,only_high,medium_and_above,low_and_above}` levels.

- **`ImagenGen` mismatched-size candidates no longer crash `torch.cat`.**
  Single-image case skips `cat` (cheaper); on the rare multi-size case it
  logs a warning and returns the first.

- **`Veo` saves ALL videos to disk** (previously only first), returns the
  first path for backward compat.

- **`StructuredOutput` prefers `response.parsed`** when the SDK provides
  it — strips markdown ` ```json … ``` ` fences automatically and falls
  back to `.text` if `.parsed` is unavailable.

- **`Vision` node has a `lossless` toggle.** When True, encodes references
  as PNG instead of JPEG-95. Use for OCR / small-text / fine-detail tasks
  where JPEG artifacts cost ~30% character accuracy. New dedicated
  `VisionOCR` node ships with this pre-set.

### Fixed — High

- **`is_transient_error` no longer false-positives on prompt content.**
  v2.0 used substring matching, so `"image 500x500"` in a prompt or
  `"max_tokens=5000"` in an error message would be classified as an HTTP
  500 and pointlessly retried. v2.1 inspects `google.genai.errors.APIError`
  typed exceptions when available, and falls back to *word-boundary*
  regex (`\b500\b`) so embedded numbers don't trip the retry path.

- **Quota-exhaustion 429s are now permanent.** Both `is_transient_error`
  and `parse_error_response` check the response body for
  "quota"/"exhausted"/"billing"/"exceeded your current" and return
  `APIQuotaError` (a permanent subclass), so the worker doesn't burn
  credits retrying a known-permanent quota error.

- **Invalid `safety_settings_json` now raises.** v2.0 caught the parse
  error, logged a warning, and silently sent the request WITHOUT any
  safety filter. The user could believe their workflow had filtering when
  none was applied. v2.1 hard-raises with a clear message.

- **TTS text-length guards.** TTS hard-raises >32 000 chars (Gemini's
  documented limit) and warns >16 000.

- **`_extract_image_from_stream` returns all candidates when requested.**
  Adds `return_all=True` for the ImageGen multi-candidate case.

- **`Vision` console banner is no longer `[API Toolkit Gemini Vision]`.**
  That was a left-over from a previous project's migration.

### Fixed — Medium

- **Quote stripping in `get_api_key`.** `.env` mistake
  `GEMINI_API_KEY="AIza..."` (with literal quotes) now works — v2.0's
  `.strip()` left the quotes in, causing 401s.

- **Jittered exponential backoff.** All retry sites (gemini_client,
  shared/retry) multiply the computed delay by `random.uniform(0.8, 1.2)`
  to prevent thundering-herd retries when many workers hit a 429 wave
  simultaneously.

- **`Retry-After` HTTP-date parsing.** RFC 7231 allows the header to be
  either delta-seconds or an HTTP-date. v2.0 only handled integers and
  silently fell through to exponential backoff for dates. v2.1 parses
  both via `email.utils.parsedate_to_datetime`.

- **Connection pooling.** `api_request_with_retry` and `download_file` now
  default to a module-level `requests.Session()` (was a fresh handshake
  per call).

- **`api_request_with_retry` rejects `max_retries < 0`** with a clear
  `ValueError` (v2.0 returned `None` silently).

- **`audio_to_comfy` no longer masks decode errors.** Previously
  `except: pass` swallowed every soundfile failure. v2.1 captures both
  soundfile AND torchaudio errors and re-raises them together so missing
  deps vs. corrupt audio is distinguishable.

- **`PromptRefiner` is no longer image-only.** New `target` input lets you
  refine prompts for `image / video / music / text / code`. v2.0
  hardcoded "for an AI Image Generator" so the node was useless for
  TextGen / Veo / Lyria preflight.

- **`VideoGen` polling interval is configurable** (default 10s, range 2-60s).

### Added — 9 new nodes (extra_nodes.py)

- **`NanoBanana - Files Upload`** (`NanoBanana2/Files`) — upload a local
  file via `client.files.upload`; returns a `(file_uri, file_name)` pair.
  Gemini Files live ~48h server-side and let you reuse a PDF / video /
  audio handle across many prompts instead of re-uploading. Required for
  multi-MB inputs that won't fit inline.

- **`NanoBanana - Ask Uploaded File`** — pair with Files Upload. Send the
  URI + mime_type + a question, get text back. Supports PDF up to 1000
  pages, video / audio up to 1 GiB.

- **`NanoBanana - Text Gen + Google Search`** — TextGen with the
  `GoogleSearch` grounding tool wired in. Returns both the answer and a
  `citations_json` list of `(url, title)` sources the model consulted.
  Use for current-events questions, recent product info, anything where
  the training cutoff produces stale answers.

- **`NanoBanana - Text Gen + Code Execution`** — TextGen with the
  `ToolCodeExecution` tool. Returns three strings: the natural-language
  answer, any Python the model actually ran, and the execution output.
  Useful for math, stats, plotting, JSON wrangling.

- **`NanoBanana - TTS Multi-Speaker Dialogue`** — proper two-voice TTS
  using `MultiSpeakerVoiceConfig`. v2.0's TTS node claimed multi-speaker
  worked "via speaker tags" but actually used a single `VoiceConfig`, so
  tags were either read aloud literally or ignored.

- **`NanoBanana - Audio Transcribe`** — feed a ComfyUI `AUDIO`, get a
  verbatim transcript (optionally with `[HH:MM:SS]` timestamps).

- **`NanoBanana - Save Embedding (.npy)`** — write the JSON-encoded
  embedding vector (output of the existing `Embed` node) to a `.npy` file
  in the ComfyUI output directory. Basename-sanitized: subdirectory and
  filename strip `../` traversal attempts.

- **`NanoBanana - Vision OCR (lossless PNG)`** — OCR-tuned Vision node
  with three output modes (`plain_text` / `structured_json` / `markdown`),
  always-PNG encoding, temperature locked at 0, optional language hint.
  `structured_json` returns `{"lines":[{"text":..,"bbox":[…]}, …]}` and
  sets `response_mime_type=application/json`.

- **`NanoBanana - Cost Estimator`** — counts input tokens via the (free)
  `count_tokens` API, multiplies by a model price table + your
  `expected_output_tokens` estimate, multiplies by `runs`, returns total
  USD plus a breakdown string. Pre-flight expensive batched workloads.

### Added — Internal helpers

- `shared.conversions`:
  - `tensor_to_png_bytes` (lossless encoding)
  - `mask_to_png_bytes` (crisp mask edges)
  - `resize_mask_to_image` (bilinear, clamped)
  - `tensor_to_pil(preserve_alpha=True)` — RGBA preserved (was silently
    flattened to black-on-RGB)
- `shared.retry`:
  - `stream_to_file(url, path, max_bytes=..., allow_redirects=False)` —
    SSRF-safe, size-bounded, atomic-on-failure file download
- `shared.node_utils`:
  - `OptionalRerunMixin` — opt-in caching via `force_rerun` boolean (the
    expensive Veo / Imagen / Lyria nodes still default to always re-run
    for backward compat, but workflow authors can now use this mixin for
    new expensive nodes)
- `gemini_client`:
  - `redact_secret()`
  - `sanitize_model_id()`
  - `clear_client_cache()`
  - `IMAGEN_SAFETY_LEVELS` constant

### Tests

- New `tests/` directory with **64 tests**, all passing. Organized into:
  - `test_security.py` (26 tests) — secret redaction, model ID
    sanitization, API-key resolution, transient-error classification,
    retry control flow
  - `test_conversions.py` (15 tests) — RGB/RGBA handling, PNG/JPEG round
    trip, mask resize edge cases, bytes_to_tensor preserves alpha
  - `test_retry.py` (9 tests) — `max_bytes` cap, `allow_redirects=False`,
    `Retry-After` HTTP-date parsing, partial-file cleanup
  - `test_nodes.py` (14 tests) — registration, schema validity,
    regressions on each verified v2.0 bug, path-traversal rejection in
    `EmbedSave`

Run with:

```
C:/ComfyUI/venv/Scripts/python -m pytest tests/
```

### Deferred to v2.2

- Async / streaming text node (requires deeper ComfyUI execution hookup
  than a simple node mixin)
- Disk-based response cache keyed on `(model, prompt, params)` —
  designed but not shipped this release
- Global token-bucket rate limiter (cross-node coordination needed)
- Native ComfyUI `VIDEO` type for Veo output (currently returns
  `(file_path, uri)` strings for VHS / external compositing)
- Function calling with declared tools as JSON

## [2.0.0] — 2026-04-26

- `ImageGen` default model bumped to `gemini-3.1-flash-image-preview`
  (Nano Banana 2).
- `ModelSelector` adds `category_validate` so picking an Imagen model in
  a text-only slot is a clear runtime error instead of a confusing API
  rejection.
- `CountTokens` switched to use the cached Gemini client.
- Lyria `:predict` HTTPS call moved the API key from a query parameter
  to the `x-goog-api-key` header.

## [Earlier]

- Initial 3-node Gemini image gen pack, then grew through TTS,
  embeddings, Veo, Lyria, Imagen, etc. to the 20-node v2.0 baseline.
  See `git log` for blow-by-blow.
