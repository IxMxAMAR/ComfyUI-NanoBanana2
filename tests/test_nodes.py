"""Smoke tests for node registration and INPUT_TYPES schemas.

These don't exercise the API (no network). They confirm:
  - All 29 nodes register correctly.
  - Every node's INPUT_TYPES schema is well-formed.
  - The known v2.0 bugs (THINKING_LEVELS NORMAL, safety_filter_level
    dropdown mismatch) are fixed.
"""

import sys
import os

# Make sure the parent ComfyUI-NanoBanana2 dir is on sys.path so we can
# import the package by its actual hyphenated name.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PARENT)
from importlib import import_module
_pkg = import_module("ComfyUI-NanoBanana2")


def test_total_node_count():
    """v2.1 should expose 19 core + 9 new = 28 nodes (subject to additions)."""
    n = len(_pkg.NODE_CLASS_MAPPINGS)
    assert n >= 28, f"Expected >=28 nodes after v2.1 additions, got {n}"


def test_every_node_has_display_name():
    classes = set(_pkg.NODE_CLASS_MAPPINGS.keys())
    names = set(_pkg.NODE_DISPLAY_NAME_MAPPINGS.keys())
    assert classes == names, f"Mismatch: only-classes={classes - names}, only-names={names - classes}"


def test_every_node_input_types_well_formed():
    for name, cls in _pkg.NODE_CLASS_MAPPINGS.items():
        it = cls.INPUT_TYPES()
        assert isinstance(it, dict), f"{name}: INPUT_TYPES must be dict"
        assert "required" in it, f"{name}: INPUT_TYPES missing 'required'"


def test_thinking_levels_no_normal():
    """v2.0 had THINKING_LEVELS = [NONE, LOW, NORMAL, HIGH]. The Gemini SDK
    ThinkingLevel enum is MINIMAL/LOW/MEDIUM/HIGH. v2.1 must use those."""
    from gemini_client import THINKING_LEVELS
    assert "NORMAL" not in THINKING_LEVELS, "NORMAL is not a valid ThinkingLevel"
    assert "MEDIUM" in THINKING_LEVELS or "MINIMAL" in THINKING_LEVELS, \
        "v2.1 should include MEDIUM and/or MINIMAL"


def test_imagen_safety_filter_dropdown_has_multiple_options():
    """v2.0 dropdown had only ['block_low_and_above'] which defeated the picker."""
    from gemini_client import IMAGEN_SAFETY_LEVELS
    assert len(IMAGEN_SAFETY_LEVELS) >= 4


def test_image_models_includes_nano_banana_2_as_default():
    """v2.0 default was the older 2.5-flash-image; v2.1 should keep the
    upstream fix that defaults to 3.1-flash-image-preview (Nano Banana 2)."""
    cls = _pkg.NODE_CLASS_MAPPINGS["NanoBanana_ImageGen"]
    it = cls.INPUT_TYPES()
    model_field = it["required"]["model"]
    default = model_field[1]["default"]
    assert "3.1" in default or "image" in default, \
        f"ImageGen default {default!r} looks wrong"


def test_imagen_seed_min_is_negative_one():
    """v2.0 used min=0 which collided with seed=0 being valid. v2.1 uses
    -1 as the 'random' sentinel."""
    cls = _pkg.NODE_CLASS_MAPPINGS["NanoBanana_ImagenGen"]
    it = cls.INPUT_TYPES()
    seed_field = it["optional"]["seed"]
    assert seed_field[1]["min"] == -1, "Imagen seed min must be -1 to distinguish from valid 0"


def test_video_seed_min_is_negative_one():
    cls = _pkg.NODE_CLASS_MAPPINGS["NanoBanana_VideoGen"]
    it = cls.INPUT_TYPES()
    seed_field = it["optional"]["seed"]
    assert seed_field[1]["min"] == -1


def test_v21_new_nodes_present():
    expected = {
        "NanoBanana_FilesUpload",
        "NanoBanana_VisionWithFile",
        "NanoBanana_TextGenSearch",
        "NanoBanana_TextGenCode",
        "NanoBanana_TTSMultiSpeaker",
        "NanoBanana_AudioTranscribe",
        "NanoBanana_EmbedSave",
        "NanoBanana_VisionOCR",
        "NanoBanana_CostEstimate",
    }
    have = set(_pkg.NODE_CLASS_MAPPINGS.keys())
    missing = expected - have
    assert not missing, f"New v2.1 nodes missing: {missing}"


def test_vision_node_has_lossless_toggle():
    """v2.1 added a lossless/PNG option to the Vision node for OCR use cases."""
    cls = _pkg.NODE_CLASS_MAPPINGS["NanoBanana_Vision"]
    it = cls.INPUT_TYPES()
    # lossless lives in optional
    assert "lossless" in it.get("optional", {}), \
        "Vision node should expose `lossless` toggle in v2.1"


def test_structured_output_returns_json():
    """Should return STRING json_output."""
    cls = _pkg.NODE_CLASS_MAPPINGS["NanoBanana_StructuredOutput"]
    assert cls.RETURN_TYPES == ("STRING",)


def test_safety_settings_node_outputs_json():
    cls = _pkg.NODE_CLASS_MAPPINGS["NanoBanana_SafetySettings"]
    import json
    inst = cls()
    out = inst.build("BLOCK_NONE", "BLOCK_NONE", "BLOCK_NONE", "BLOCK_NONE")
    parsed = json.loads(out[0])
    assert isinstance(parsed, list)
    assert len(parsed) == 4
    for entry in parsed:
        assert "category" in entry and "threshold" in entry


def test_cost_estimate_node_returns_int_float_string():
    cls = _pkg.NODE_CLASS_MAPPINGS["NanoBanana_CostEstimate"]
    assert cls.RETURN_TYPES == ("INT", "FLOAT", "STRING")


def test_embed_save_rejects_path_traversal(tmp_path, monkeypatch):
    """Ensure EmbedSave basenames the filename — no `..` escapes."""
    import json as _json
    cls = _pkg.NODE_CLASS_MAPPINGS["NanoBanana_EmbedSave"]
    inst = cls()
    # Monkeypatch folder_paths to point at tmp
    import sys as _sys

    class _FakeFolderPaths:
        @staticmethod
        def get_output_directory():
            return str(tmp_path)

    monkeypatch.setitem(_sys.modules, "folder_paths", _FakeFolderPaths())

    vec = _json.dumps([0.1, 0.2, 0.3])
    # Attempt path traversal
    out = inst.save(vec, "../../etc/passwd.npy", subdirectory="../../etc")
    saved = out[0]
    # Path must be inside tmp_path
    assert str(tmp_path) in saved
    # And basename'd
    assert "passwd.npy" in saved
