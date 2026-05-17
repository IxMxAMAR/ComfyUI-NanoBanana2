"""Tests for shared/conversions.py — alpha handling, mask resize, PNG output."""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from shared.conversions import (
    tensor_to_pil, pil_to_tensor,
    tensor_to_jpeg_bytes, tensor_to_png_bytes,
    mask_to_jpeg_bytes, mask_to_png_bytes,
    resize_mask_to_image, bytes_to_tensor,
)


def _checker_tensor(h=16, w=16, c=3):
    """Create a deterministic checkerboard tensor for round-trip tests."""
    arr = np.zeros((h, w, c), dtype=np.float32)
    arr[::2, ::2] = 1.0
    arr[1::2, 1::2] = 1.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, C]


class TestTensorToPil:
    def test_rgb_3channel(self):
        t = _checker_tensor(c=3)
        img = tensor_to_pil(t)
        assert img.mode == "RGB"
        assert img.size == (16, 16)

    def test_rgba_preserved(self):
        # 4-channel input should round-trip as RGBA
        t = _checker_tensor(c=4)
        img = tensor_to_pil(t, preserve_alpha=True)
        assert img.mode == "RGBA"

    def test_rgba_flattened_when_requested(self):
        t = _checker_tensor(c=4)
        img = tensor_to_pil(t, preserve_alpha=False)
        assert img.mode == "RGB"


class TestPngVsJpeg:
    def test_png_lossless(self):
        t = _checker_tensor()
        png_bytes = tensor_to_png_bytes(t)
        # Decoding round trip must yield identical pixels
        img = Image.open(io.BytesIO(png_bytes))
        arr_back = np.array(img, dtype=np.float32) / 255.0
        arr_orig = t[0].numpy()
        np.testing.assert_allclose(arr_back, arr_orig, atol=0)  # zero tolerance

    def test_jpeg_round_trips_to_rgb(self):
        # JPEG is lossy and compression sizes are content-dependent — for a
        # tiny checkerboard PNG can beat JPEG by ~20x because PNG's filtered
        # zlib loves repetition while JPEG has unavoidable header + dct
        # baseline overhead. Just verify JPEG is valid and decodes to RGB.
        t = _checker_tensor(h=128, w=128)
        jpg = tensor_to_jpeg_bytes(t, quality=80)
        img = Image.open(io.BytesIO(jpg))
        assert img.mode == "RGB"
        assert img.size == (128, 128)

    def test_rgba_to_jpeg_flattens_alpha(self):
        # Pure-RGBA with transparent alpha should not throw
        t = torch.zeros((1, 8, 8, 4), dtype=torch.float32)
        t[..., 3] = 0.5  # half-opaque
        jpg = tensor_to_jpeg_bytes(t)
        img = Image.open(io.BytesIO(jpg))
        assert img.mode == "RGB"


class TestMaskHelpers:
    def test_mask_to_png_is_grayscale(self):
        m = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        b = mask_to_png_bytes(m)
        img = Image.open(io.BytesIO(b))
        assert img.mode == "L"
        arr = np.array(img)
        assert arr.shape == (2, 2)
        assert arr[0, 0] == 0 and arr[0, 1] == 255

    def test_mask_to_jpeg_round_trips(self):
        m = torch.zeros((1, 8, 8))
        m[0, :4, :] = 1.0  # top half white
        b = mask_to_jpeg_bytes(m)
        img = Image.open(io.BytesIO(b))
        # JPEG of B&W mask should still be ~grayscale RGB
        assert img.mode == "RGB"


class TestResizeMaskToImage:
    def test_no_resize_when_matching(self):
        img = torch.zeros((1, 64, 64, 3))
        mask = torch.zeros((1, 64, 64))
        out = resize_mask_to_image(mask, img)
        assert out.shape == (1, 64, 64)

    def test_resizes_up(self):
        # Common Gemini-Inpaint failure mode: SAM mask is 256, image is 1024
        img = torch.zeros((1, 1024, 1024, 3))
        mask = torch.zeros((1, 256, 256))
        out = resize_mask_to_image(mask, img)
        assert out.shape == (1, 1024, 1024)

    def test_resizes_down(self):
        img = torch.zeros((1, 128, 128, 3))
        mask = torch.zeros((1, 512, 512))
        mask[0, 0, 0] = 1.0
        out = resize_mask_to_image(mask, img)
        assert out.shape == (1, 128, 128)
        assert out.dtype == torch.float32

    def test_clamps_to_unit_range(self):
        img = torch.zeros((1, 64, 64, 3))
        mask = torch.full((1, 32, 32), 1.5)
        out = resize_mask_to_image(mask, img)
        assert float(out.max()) <= 1.0
        assert float(out.min()) >= 0.0

    def test_2d_mask_accepted(self):
        img = torch.zeros((1, 64, 64, 3))
        mask = torch.zeros((32, 32))  # [H, W] not [B, H, W]
        out = resize_mask_to_image(mask, img)
        assert out.shape == (1, 64, 64)


class TestBytesToTensor:
    def test_rgb_round_trip(self):
        t = _checker_tensor()
        png = tensor_to_png_bytes(t)
        back = bytes_to_tensor(png)
        assert back.shape == t.shape
        np.testing.assert_allclose(back.numpy(), t.numpy(), atol=1e-6)

    def test_preserves_rgba(self):
        t = torch.zeros((1, 8, 8, 4), dtype=torch.float32)
        png = tensor_to_png_bytes(t)
        back = bytes_to_tensor(png)
        assert back.shape[-1] == 4  # alpha preserved
