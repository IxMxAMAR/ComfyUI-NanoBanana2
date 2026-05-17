"""Image, audio, and tensor conversion utilities for ComfyUI nodes.

v2.1 changes:
  - `tensor_to_pil` supports both 3-channel (RGB) and 4-channel (RGBA)
    tensors. Alpha is preserved (instead of silently flattened to black).
  - `tensor_to_jpeg_bytes` adds an `assume_rgb` switch; alpha-bearing
    tensors are flattened over white when JPEG is requested.
  - New `tensor_to_png_bytes` — lossless encoding for vision/OCR tasks
    where JPEG artifacts degrade small-text recognition.
  - New `resize_mask_to_image` — Gemini Inpaint requires mask dims to
    match image dims, otherwise the API returns 400.
  - `mask_to_png_bytes` (alongside the existing JPEG variant) — masks
    really need lossless edges; JPEG smears the boundary by ~2px.
"""

import base64
import io
import struct
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor, preserve_alpha: bool = True) -> Image.Image:
    """Convert a ComfyUI image tensor to a PIL Image.

    Accepts [B,H,W,C] or [H,W,C]. Takes the first item from a batch.
    If C==4 and preserve_alpha=True, returns RGBA. Otherwise RGB.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() != 3:
        raise ValueError(
            f"tensor_to_pil expects [B,H,W,C] or [H,W,C], got shape {tuple(tensor.shape)}"
        )
    arr = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()
    c = arr.shape[-1]
    if c == 4 and preserve_alpha:
        return Image.fromarray(arr, mode="RGBA")
    if c == 4:
        return Image.fromarray(arr[..., :3], mode="RGB")
    if c == 3:
        return Image.fromarray(arr, mode="RGB")
    if c == 1:
        return Image.fromarray(arr[..., 0], mode="L")
    raise ValueError(f"tensor_to_pil: unsupported channel count {c}")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a ComfyUI image tensor [1,H,W,C] float32.

    RGBA → 4 channels preserved; L/P/etc → converted to RGB.
    """
    if image.mode in ("RGB", "RGBA"):
        pass
    elif image.mode == "L":
        image = image.convert("RGB")
    else:
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # grayscale safety net
        arr = arr[..., None]
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_base64(tensor: Optional[torch.Tensor], fmt: str = "PNG") -> Optional[str]:
    """Convert a ComfyUI image tensor to a base64-encoded string."""
    if tensor is None:
        return None
    img = tensor_to_pil(tensor, preserve_alpha=(fmt.upper() == "PNG"))
    if fmt.upper() == "JPEG" and img.mode == "RGBA":
        img = _flatten_alpha(img, background=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def tensor_to_jpeg_bytes(tensor: torch.Tensor, quality: int = 95) -> bytes:
    """Convert a ComfyUI image tensor to JPEG bytes.

    RGBA inputs are flattened over white. For images where every pixel
    matters (OCR, vision parsing), prefer `tensor_to_png_bytes`.
    """
    img = tensor_to_pil(tensor, preserve_alpha=True)
    if img.mode == "RGBA":
        img = _flatten_alpha(img, background=(255, 255, 255))
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a ComfyUI image tensor to LOSSLESS PNG bytes.

    Use this for vision tasks where JPEG artifacts matter — OCR of small
    text, recognizing logos/seals, fine-grained classification.
    """
    img = tensor_to_pil(tensor, preserve_alpha=True)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _flatten_alpha(img: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """Composite an RGBA image over an opaque background."""
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, background)
    bg.paste(img, mask=img.split()[3])
    return bg


def mask_to_jpeg_bytes(mask: torch.Tensor, quality: int = 95) -> bytes:
    """Convert a ComfyUI MASK to a B/W JPEG (white=mask, black=keep).

    NOTE: JPEG quantizes mask edges. For inpainting where every pixel
    matters, prefer `mask_to_png_bytes`.
    """
    if mask.dim() == 3:
        mask = mask[0]
    arr = (mask.clamp(0, 1) * 255).byte().cpu().numpy()
    rgb = np.stack([arr, arr, arr], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def mask_to_png_bytes(mask: torch.Tensor) -> bytes:
    """Convert a ComfyUI MASK to a LOSSLESS B/W PNG (white=mask, black=keep)."""
    if mask.dim() == 3:
        mask = mask[0]
    arr = (mask.clamp(0, 1) * 255).byte().cpu().numpy()
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def resize_mask_to_image(
    mask: torch.Tensor, image: torch.Tensor
) -> torch.Tensor:
    """Resize a MASK tensor [B,H,W] to match an IMAGE tensor [B,H,W,C].

    Gemini Inpaint requires the mask to share dimensions with the source
    image (the API enforces this and returns 400 on mismatch). ComfyUI
    masks frequently come from different upstream sources (SAM, depth,
    alpha channels) at arbitrary sizes — we resize with bilinear
    interpolation to match.
    """
    # Image: [B,H,W,C]; take H,W
    if image.dim() != 4:
        raise ValueError(f"resize_mask_to_image: image must be [B,H,W,C], got {tuple(image.shape)}")
    target_h, target_w = int(image.shape[1]), int(image.shape[2])

    m = mask
    if m.dim() == 2:  # [H,W]
        m = m.unsqueeze(0)
    if m.dim() != 3:
        raise ValueError(f"resize_mask_to_image: mask must be [B,H,W] or [H,W], got {tuple(mask.shape)}")
    cur_h, cur_w = int(m.shape[1]), int(m.shape[2])
    if cur_h == target_h and cur_w == target_w:
        return m

    # F.interpolate wants [B,C,H,W]
    import torch.nn.functional as F
    resized = F.interpolate(
        m.unsqueeze(1), size=(target_h, target_w),
        mode="bilinear", align_corners=False,
    )
    return resized.squeeze(1).clamp(0, 1)


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Convert image bytes (any PIL-supported format) to a ComfyUI tensor [1,H,W,C].

    Preserves alpha channel when present (RGBA → 4-channel tensor).
    """
    img = Image.open(io.BytesIO(data))
    # Load into memory before the BytesIO closes
    img.load()
    return pil_to_tensor(img)


def audio_to_comfy(audio_bytes: bytes, output_format: str = "wav") -> dict:
    """Convert raw audio bytes to ComfyUI AUDIO dict format.

    Tries soundfile first, then falls back to torchaudio. The previous
    silent `except: pass` was replaced with logging so install/format
    issues surface instead of being masked.
    """
    sf_err = None
    try:
        import soundfile as sf

        buf = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buf, dtype="float32")
        if audio_data.ndim == 1:
            audio_data = audio_data[:, np.newaxis]
        waveform = torch.from_numpy(audio_data.T).unsqueeze(0).float()
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception as e:
        sf_err = e

    ta_err = None
    try:
        import torchaudio

        buf = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(buf, format=output_format)
        waveform = waveform.unsqueeze(0).float()
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception as e:
        ta_err = e

    raise RuntimeError(
        f"Failed to decode audio. soundfile: {sf_err!r}. torchaudio: {ta_err!r}. "
        f"Install one of: `pip install soundfile` or `pip install torchaudio`."
    )


def comfy_to_audio_bytes(audio: dict, fmt: str = "wav") -> bytes:
    """Convert ComfyUI AUDIO dict to WAV bytes."""
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if waveform.dim() == 3:
        waveform = waveform[0]

    try:
        import soundfile as sf

        buf = io.BytesIO()
        data = waveform.cpu().numpy().T
        sf.write(buf, data, sample_rate, format="WAV")
        return buf.getvalue()
    except Exception:
        pass

    # Fallback: manual WAV encoding
    data = waveform.cpu().numpy()
    channels = data.shape[0]
    samples = data.shape[1]

    int_data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    interleaved = np.empty(channels * samples, dtype=np.int16)
    for ch in range(channels):
        interleaved[ch::channels] = int_data[ch]

    raw_bytes = interleaved.tobytes()
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2

    buf = io.BytesIO()
    data_size = len(raw_bytes)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(raw_bytes)

    return buf.getvalue()
