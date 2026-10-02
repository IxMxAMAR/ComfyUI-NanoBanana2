"""Tests for the v2.7 surface: current model IDs, Interactions API nodes
(Lyria, Gemini 3.8 TTS, Omni video, URL context) and new parameters.

Every Gemini client is a fake; nothing touches the network.
"""

import base64
import io
import os
import sys
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
_pkg = import_module("ComfyUI-NanoBanana2")
core = import_module("ComfyUI-NanoBanana2.nanobanana_node")
extra = import_module("ComfyUI-NanoBanana2.extra_nodes")
client_mod = import_module("ComfyUI-NanoBanana2.gemini_client")

NODES = _pkg.NODE_CLASS_MAPPINGS


def _wav_bytes(sample_rate=24000, channels=1, seconds=0.1):
    samples = np.zeros((int(sample_rate * seconds), channels), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class FakeInteractions:
    def __init__(self, **response):
        self.response = SimpleNamespace(**response)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def _audio_interaction():
    return FakeInteractions(
        output_audio=SimpleNamespace(data=base64.b64encode(_wav_bytes()).decode())
    )


def _patch_client(monkeypatch, module, client):
    monkeypatch.setattr(module, "get_client", lambda key, network=None: client)


def _image(batch=1):
    return torch.zeros((batch, 8, 8, 3), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------

def test_shut_down_models_are_gone():
    shut_down = {
        "gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview",
        "gemini-3-pro-preview", "gemini-3.1-flash-lite-preview",
        "gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001", "gemini-robotics-er-1.5-preview",
        "gemini-robotics-er-1.6-preview", "gemini-2.5-computer-use-preview-10-2025",
        "veo-3.0-generate-001", "veo-3.0-fast-generate-001", "veo-2.0-generate-001",
    }
    assert not shut_down & set(client_mod.ALL_MODELS)


def test_new_models_present():
    assert {"gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
            "gemini-3.5-flash", "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite"} <= set(client_mod.TEXT_MODELS)
    assert {"gemini-3.1-flash-image", "gemini-3.1-flash-lite-image",
            "gemini-3-pro-image"} <= set(client_mod.IMAGE_MODELS)
    assert {"gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"} <= set(client_mod.TTS_MODELS)
    assert "lyria-3.5" in client_mod.LYRIA_MODELS
    assert client_mod.OMNI_MODELS == ["gemini-omni-1.1-flash"]
    assert client_mod.TRANSCRIBE_MODELS == ["gemini-3.5-transcribe"]


def test_every_combo_default_is_a_listed_value():
    for name, cls in NODES.items():
        it = cls.INPUT_TYPES()
        for section in ("required", "optional"):
            for field, spec in it.get(section, {}).items():
                if isinstance(spec[0], list) and len(spec) > 1 and "default" in spec[1]:
                    assert spec[1]["default"] in spec[0], f"{name}.{field}"


def test_defaults_moved_off_shut_down_models():
    video = NODES["NanoBanana_VideoGen"].INPUT_TYPES()
    assert video["required"]["model"][1]["default"] == "veo-3.1-fast-generate-preview"
    vision = NODES["NanoBanana_Vision"].INPUT_TYPES()
    assert vision["required"]["model"][1]["default"] == "gemini-3.1-flash-lite"
    for node in ("NanoBanana_ImageGen", "NanoBanana_ImageEdit",
                 "NanoBanana_Inpaint", "NanoBanana_Outpaint"):
        spec = NODES[node].INPUT_TYPES()["required"]["model"]
        assert spec[1]["default"] == "gemini-3.1-flash-image"


def test_image_option_additions():
    assert {"1:4", "4:1", "1:8", "8:1"} <= set(client_mod.ASPECT_RATIOS)
    assert "512" in client_mod.IMAGE_SIZES


# ---------------------------------------------------------------------------
# New inputs stay at the end of "optional" so saved widget values keep lining up
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("node, appended", [
    ("NanoBanana_Vision", ["network", "media_resolution"]),
    ("NanoBanana_ImageGen", ["network", "search_grounding"]),
    ("NanoBanana_VisionOCR", ["network", "media_resolution"]),
    ("NanoBanana_TTS", ["network", "custom_voice"]),
    ("NanoBanana_MusicGen", ["sample_count", "image", "network"]),
    ("NanoBanana_VideoGen", ["network", "resolution", "last_frame"]),
    ("NanoBanana_AudioTranscribe",
     ["network", "language_codes", "custom_vocabulary", "diarization", "transcription_mode"]),
])
def test_new_inputs_are_appended_last(node, appended):
    keys = list(NODES[node].INPUT_TYPES()["optional"])
    assert keys[-len(appended):] == appended


def test_new_node_registration():
    for key in ("NanoBanana_OmniVideoGen", "NanoBanana_TextGenURL"):
        assert key in NODES
        assert key in _pkg.NODE_DISPLAY_NAME_MAPPINGS
    assert NODES["NanoBanana_OmniVideoGen"].RETURN_TYPES == ("STRING", "STRING")
    assert NODES["NanoBanana_TextGenURL"].RETURN_TYPES == ("STRING",)


# ---------------------------------------------------------------------------
# generate_content config
# ---------------------------------------------------------------------------

class TestBuildConfig:
    def test_level_wins_over_budget(self):
        cfg = core._build_config(thinking_level="HIGH", thinking_budget=2000)
        assert cfg.thinking_config.thinking_level.value == "HIGH"
        assert cfg.thinking_config.thinking_budget is None

    def test_budget_applies_without_level(self):
        cfg = core._build_config(thinking_level="NONE", thinking_budget=2000)
        assert cfg.thinking_config.thinking_budget == 2000
        assert cfg.thinking_config.thinking_level is None

    def test_no_thinking_by_default(self):
        assert core._build_config().thinking_config is None

    def test_media_resolution(self):
        cfg = core._build_config(media_resolution="HIGH")
        assert cfg.media_resolution.value == "MEDIA_RESOLUTION_HIGH"
        assert core._build_config(media_resolution="AUTO").media_resolution is None

    def test_search_grounding(self):
        assert core._build_config().tools is None
        web = core._build_config(search_grounding="web").tools[0].google_search
        assert web is not None and web.search_types is None
        both = core._build_config(search_grounding="web + image").tools[0].google_search
        assert both.search_types.web_search is not None
        assert both.search_types.image_search is not None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class FakeModels:
    def __init__(self, **response):
        self.response = SimpleNamespace(**response)
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


@pytest.mark.parametrize("model, sends_task_type", [
    ("gemini-embedding-001", True),
    ("gemini-embedding-2", False),
    ("gemini-embedding-2-preview", False),
])
def test_embed_task_type_only_for_models_that_take_it(monkeypatch, model, sends_task_type):
    models = FakeModels(embeddings=[SimpleNamespace(values=[0.1, 0.2])])
    _patch_client(monkeypatch, core, SimpleNamespace(models=models))
    out = NODES["NanoBanana_Embed"]().embed("key", model, "hello", output_dim=256)
    config = models.calls[0]["config"]
    assert (config.task_type is not None) == sends_task_type
    assert config.output_dimensionality == 256
    assert out[1] == 2


# ---------------------------------------------------------------------------
# Gemini 3.8 TTS (Interactions API)
# ---------------------------------------------------------------------------

def test_wav_bytes_to_audio_layout():
    audio = client_mod.wav_bytes_to_audio(_wav_bytes(sample_rate=44100, channels=2))
    assert audio["sample_rate"] == 44100
    assert audio["waveform"].shape == (1, 2, 4410)
    assert audio["waveform"].dtype == torch.float32


def test_tts_38_uses_interactions(monkeypatch):
    interactions = _audio_interaction()
    _patch_client(monkeypatch, core, SimpleNamespace(interactions=interactions))
    out = NODES["NanoBanana_TTS"]().generate(
        "key", "gemini-3.8-flash-tts", "Have a nice day", "Kore",
        style_prompt="cheerful",
    )
    request = interactions.requests[0]
    assert request["model"] == "gemini-3.8-flash-tts"
    assert request["response_format"] == {"type": "audio"}
    assert request["generation_config"] == {"speech_config": [{"voice": "Kore"}]}
    turn = request["input"][0]
    assert turn["type"] == "user_input"
    assert turn["content"] == [{
        "type": "text", "text": "Have a nice day",
        "annotations": [{"type": "speech_metadata", "style": "cheerful"}],
    }]
    assert out[0]["sample_rate"] == 24000


def test_tts_38_custom_voice_overrides_dropdown(monkeypatch):
    interactions = _audio_interaction()
    _patch_client(monkeypatch, core, SimpleNamespace(interactions=interactions))
    NODES["NanoBanana_TTS"]().generate(
        "key", "gemini-3.8-flash-lite-tts", "Hi", "Kore", custom_voice=" voice_abc123 ",
    )
    request = interactions.requests[0]
    assert request["generation_config"]["speech_config"] == [{"voice": "voice_abc123"}]
    assert "annotations" not in request["input"][0]["content"][0]


def test_tts_25_still_uses_generate_content(monkeypatch):
    models = FakeModels(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
        SimpleNamespace(inline_data=SimpleNamespace(data=b"\x00\x00" * 100))]))])
    _patch_client(monkeypatch, core, SimpleNamespace(models=models))
    out = NODES["NanoBanana_TTS"]().generate("key", "gemini-2.5-flash-preview-tts", "Hi", "Kore")
    assert models.calls[0]["model"] == "gemini-2.5-flash-preview-tts"
    assert out[0]["waveform"].shape == (1, 1, 100)


def test_dialogue_turns_split_by_speaker():
    speakers = [{"speaker": "Alice", "voice": "Kore"}, {"speaker": "Bob", "voice": "Puck"}]
    turns = extra._dialogue_turns("Alice: Hi Bob\nBob: Hi Alice\n\nstill Bob", speakers)
    assert [t["text"] for t in turns] == ["Hi Bob", "Hi Alice\nstill Bob"]
    assert [t["annotations"][0]["speaker"] for t in turns] == ["Alice", "Bob"]


def test_dialogue_must_open_with_a_speaker():
    with pytest.raises(ValueError):
        extra._dialogue_turns("no speaker here", [{"speaker": "Alice", "voice": "Kore"}])


def test_multispeaker_38_uses_interactions(monkeypatch):
    interactions = _audio_interaction()
    _patch_client(monkeypatch, extra, SimpleNamespace(interactions=interactions))
    NODES["NanoBanana_TTSMultiSpeaker"]().generate(
        "key", "gemini-3.8-flash-tts", "Alice: Hi\nBob: Hello",
        "Alice", "Kore", "Bob", "Puck",
    )
    request = interactions.requests[0]
    assert request["generation_config"] == {"speech_config": {"speakers": [
        {"speaker": "Alice", "voice": "Kore"}, {"speaker": "Bob", "voice": "Puck"},
    ]}}
    assert len(request["input"][0]["content"]) == 2


# ---------------------------------------------------------------------------
# Lyria (Interactions API)
# ---------------------------------------------------------------------------

def test_music_text_only_request(monkeypatch):
    interactions = _audio_interaction()
    _patch_client(monkeypatch, core, SimpleNamespace(interactions=interactions))
    out = NODES["NanoBanana_MusicGen"]().generate("key", "lyria-3.5", "lofi beat")
    request = interactions.requests[0]
    assert request["model"] == "lyria-3.5"
    assert request["input"] == "lofi beat"
    assert request["response_format"] == {"type": "audio"}
    assert out[0]["waveform"].shape[0] == 1


def test_music_negative_prompt_and_images(monkeypatch):
    interactions = _audio_interaction()
    _patch_client(monkeypatch, core, SimpleNamespace(interactions=interactions))
    NODES["NanoBanana_MusicGen"]().generate(
        "key", "lyria-3-clip-preview", "calm piano", negative_prompt="drums", image=_image(2),
    )
    content = interactions.requests[0]["input"]
    assert content[0] == {"type": "text", "text": "calm piano\nAvoid: drums"}
    assert [c["type"] for c in content[1:]] == ["image", "image"]
    assert content[1]["mime_type"] == "image/png"
    base64.b64decode(content[1]["data"])
    assert "response_format" not in interactions.requests[0]


def test_music_without_audio_raises(monkeypatch):
    _patch_client(monkeypatch, core, SimpleNamespace(interactions=FakeInteractions(output_audio=None)))
    with pytest.raises(RuntimeError, match="no audio"):
        NODES["NanoBanana_MusicGen"]().generate("key", "lyria-3.5", "x")


def test_music_rejects_bad_model_id(monkeypatch):
    _patch_client(monkeypatch, core, SimpleNamespace(interactions=_audio_interaction()))
    with pytest.raises(ValueError):
        NODES["NanoBanana_MusicGen"]().generate("key", "lyria-3.5", "x", custom_model="../x")


# ---------------------------------------------------------------------------
# Gemini Omni video
# ---------------------------------------------------------------------------

@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_output_dir", lambda: str(tmp_path))
    return tmp_path


def test_omni_text_to_video_base64(monkeypatch, output_dir):
    video = SimpleNamespace(data=base64.b64encode(b"mp4-bytes").decode())
    interactions = FakeInteractions(output_video=video, id="v1_abc")
    _patch_client(monkeypatch, core, SimpleNamespace(interactions=interactions))
    path, interaction_id = NODES["NanoBanana_OmniVideoGen"]().generate(
        "key", "gemini-omni-1.1-flash", "a marble rolling",
    )
    request = interactions.requests[0]
    assert request["input"] == "a marble rolling"
    assert request["response_format"] == {
        "type": "video", "aspect_ratio": "16:9", "resolution": "720p", "delivery": "base64",
    }
    assert "previous_interaction_id" not in request
    assert open(path, "rb").read() == b"mp4-bytes"
    assert interaction_id == "v1_abc"


def test_omni_media_inputs_and_edit_chain(monkeypatch, output_dir):
    video = SimpleNamespace(data=base64.b64encode(b"x").decode())
    interactions = FakeInteractions(output_video=video, id="v1_def")
    _patch_client(monkeypatch, core, SimpleNamespace(interactions=interactions))
    NODES["NanoBanana_OmniVideoGen"]().generate(
        "key", "gemini-omni-1.1-flash", "make it night", image=_image(2),
        video_uri="https://generativelanguage.googleapis.com/v1beta/files/abc",
        previous_interaction_id="v1_abc", aspect_ratio="9:16", resolution="1080p",
    )
    request = interactions.requests[0]
    kinds = [c["type"] for c in request["input"]]
    assert kinds == ["image", "image", "video", "text"]
    assert request["input"][2]["uri"].endswith("/files/abc")
    assert request["previous_interaction_id"] == "v1_abc"
    assert request["response_format"]["aspect_ratio"] == "9:16"
    assert request["response_format"]["resolution"] == "1080p"


def test_omni_uri_delivery_polls_then_downloads(monkeypatch, output_dir):
    video = SimpleNamespace(uri="https://generativelanguage.googleapis.com/v1beta/files/abc123")
    states = iter(["PROCESSING", "ACTIVE"])
    polled = []
    downloads = []

    def get(name):
        polled.append(name)
        return SimpleNamespace(state=SimpleNamespace(name=next(states)))

    def download(file, destination):
        downloads.append((file, destination))

    client = SimpleNamespace(
        interactions=FakeInteractions(output_video=video, id="v1_x"),
        files=SimpleNamespace(get=get, download=download),
    )
    _patch_client(monkeypatch, core, client)
    monkeypatch.setattr("time.sleep", lambda s: None)
    path, _ = NODES["NanoBanana_OmniVideoGen"]().generate(
        "key", "gemini-omni-1.1-flash", "sunset", delivery="uri",
    )
    assert polled == [video.uri, video.uri]
    assert downloads == [(video.uri, path)]
    assert client.interactions.requests[0]["response_format"]["delivery"] == "uri"


def test_omni_uri_delivery_failed_file_raises(monkeypatch, output_dir):
    video = SimpleNamespace(uri="https://generativelanguage.googleapis.com/v1beta/files/abc123")
    client = SimpleNamespace(
        interactions=FakeInteractions(output_video=video, id="v1_x"),
        files=SimpleNamespace(get=lambda name: SimpleNamespace(state=SimpleNamespace(name="FAILED"))),
    )
    _patch_client(monkeypatch, core, client)
    with pytest.raises(RuntimeError, match="failed"):
        NODES["NanoBanana_OmniVideoGen"]().generate(
            "key", "gemini-omni-1.1-flash", "sunset", delivery="uri",
        )


def test_omni_without_video_raises(monkeypatch, output_dir):
    _patch_client(monkeypatch, core, SimpleNamespace(interactions=FakeInteractions(output_video=None)))
    with pytest.raises(RuntimeError, match="no video"):
        NODES["NanoBanana_OmniVideoGen"]().generate("key", "gemini-omni-1.1-flash", "x")


# ---------------------------------------------------------------------------
# Veo new parameters
# ---------------------------------------------------------------------------

def test_veo_resolution_and_last_frame(monkeypatch, output_dir):
    saved = []
    video = SimpleNamespace(save=lambda p: saved.append(p), uri="")
    operation = SimpleNamespace(
        done=True, name="op",
        response=SimpleNamespace(generated_videos=[SimpleNamespace(video=video)]),
    )
    models = SimpleNamespace(generate_videos=lambda **kw: calls.append(kw) or operation)
    calls = []
    _patch_client(monkeypatch, core, SimpleNamespace(models=models))
    NODES["NanoBanana_VideoGen"]().generate(
        "key", "veo-3.1-fast-generate-preview", "a walk", source_image=_image(),
        last_frame=_image(), resolution="1080p",
    )
    config = calls[0]["config"]
    assert config.resolution == "1080p"
    assert config.last_frame.mime_type == "image/png"
    assert calls[0]["image"].mime_type == "image/png"
    assert len(saved) == 1


def test_veo_defaults_send_no_new_fields(monkeypatch, output_dir):
    video = SimpleNamespace(save=lambda p: None, uri="")
    operation = SimpleNamespace(
        done=True, name="op",
        response=SimpleNamespace(generated_videos=[SimpleNamespace(video=video)]),
    )
    calls = []
    models = SimpleNamespace(generate_videos=lambda **kw: calls.append(kw) or operation)
    _patch_client(monkeypatch, core, SimpleNamespace(models=models))
    NODES["NanoBanana_VideoGen"]().generate("key", "veo-3.1-generate-preview", "a walk")
    config = calls[0]["config"]
    assert config.resolution is None and config.last_frame is None


# ---------------------------------------------------------------------------
# Transcription, URL context, OCR, cost table
# ---------------------------------------------------------------------------

def _audio_input():
    return {"waveform": torch.zeros((1, 1, 1600)), "sample_rate": 16000}


def test_transcribe_model_sends_config_and_no_prompt(monkeypatch):
    models = FakeModels(text=" hello world ")
    _patch_client(monkeypatch, extra, SimpleNamespace(models=models))
    out = NODES["NanoBanana_AudioTranscribe"]().transcribe(
        "key", "gemini-3.5-transcribe", _audio_input(),
        language_codes="en-US, es-ES", custom_vocabulary="Kubernetes", transcription_mode="SMART",
    )
    call = models.calls[0]
    assert len(call["contents"]) == 1
    cfg = call["config"].audio_transcription_config
    assert cfg.language_codes == ["en-US", "es-ES"]
    assert cfg.custom_vocabulary == ["Kubernetes"]
    assert cfg.mode == "SMART"
    assert cfg.diarization is False
    assert out == ("hello world",)


def test_transcribe_defaults_leave_language_to_auto_detect(monkeypatch):
    models = FakeModels(text="x")
    _patch_client(monkeypatch, extra, SimpleNamespace(models=models))
    NODES["NanoBanana_AudioTranscribe"]().transcribe(
        "key", "gemini-3.5-transcribe", _audio_input(), diarization=True,
    )
    cfg = models.calls[0]["config"].audio_transcription_config
    assert not cfg.language_codes and not cfg.custom_vocabulary
    assert cfg.diarization is True


def test_transcribe_other_models_keep_prompt_path(monkeypatch):
    models = FakeModels(text="hi")
    _patch_client(monkeypatch, extra, SimpleNamespace(models=models))
    NODES["NanoBanana_AudioTranscribe"]().transcribe("key", "gemini-2.5-flash", _audio_input())
    call = models.calls[0]
    assert call["config"].audio_transcription_config is None
    assert len(call["contents"][0].parts) == 2


def test_url_context_request(monkeypatch):
    interactions = FakeInteractions(output_text="  summary  ")
    _patch_client(monkeypatch, extra, SimpleNamespace(interactions=interactions))
    out = NODES["NanoBanana_TextGenURL"]().generate(
        "key", "gemini-3.8-flash", "Summarize https://example.com",
        system_instruction="Be brief", google_search=True,
    )
    request = interactions.requests[0]
    assert request["tools"] == [{"type": "url_context"}, {"type": "google_search"}]
    assert request["system_instruction"] == "Be brief"
    assert request["input"] == "Summarize https://example.com"
    assert out == ("summary",)


def test_url_context_defaults_use_url_tool_only(monkeypatch):
    interactions = FakeInteractions(output_text=None)
    _patch_client(monkeypatch, extra, SimpleNamespace(interactions=interactions))
    out = NODES["NanoBanana_TextGenURL"]().generate("key", "gemini-3.8-flash", "x")
    request = interactions.requests[0]
    assert request["tools"] == [{"type": "url_context"}]
    assert "system_instruction" not in request
    assert out == ("",)


def test_ocr_media_resolution(monkeypatch):
    models = FakeModels(text="text")
    _patch_client(monkeypatch, extra, SimpleNamespace(models=models))
    NODES["NanoBanana_VisionOCR"]().ocr("key", "gemini-2.5-pro", _image(), media_resolution="HIGH")
    NODES["NanoBanana_VisionOCR"]().ocr("key", "gemini-2.5-pro", _image())
    assert models.calls[0]["config"].media_resolution.value == "MEDIA_RESOLUTION_HIGH"
    assert models.calls[1]["config"].media_resolution is None


def test_price_table_covers_current_text_models():
    prices = extra._PRICE_PER_MTOK
    for model in ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"):
        assert model in prices
    assert prices["gemini-3.1-pro-preview"] == (2.00, 12.00)
    assert "gemini-3.1-flash-lite-preview" not in prices


def test_model_selector_new_categories():
    select = NODES["NanoBanana_ModelSelector"]().select
    assert select("gemini-omni-1.1-flash", "omni") == ("gemini-omni-1.1-flash",)
    assert select("gemini-3.5-transcribe", "transcribe") == ("gemini-3.5-transcribe",)
    with pytest.raises(ValueError):
        select("gemini-omni-1.1-flash", "veo")
