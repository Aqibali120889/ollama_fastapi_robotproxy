import sys
import types

import numpy as np

from app import voice_stt


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeWhisperModel:
    def __init__(self, expected_language):
        self.expected_language = expected_language
        self.received_audio = None

    def transcribe(self, audio, language=None, vad_filter=None):
        self.received_audio = audio
        assert language == self.expected_language
        return [_FakeSegment(" hello "), _FakeSegment("world ")], {"language": language}


def test_transcribe_empty_bytes_returns_empty_string():
    assert voice_stt.transcribe(b"") == ""


def test_transcribe_joins_segments_and_converts_audio(monkeypatch):
    fake_model = _FakeWhisperModel(expected_language="en")
    monkeypatch.setattr(voice_stt, "_load_model", lambda *a, **k: fake_model)

    pcm = np.array([0, 16384, -16384, 32767], dtype=np.int16).tobytes()
    result = voice_stt.transcribe(pcm, language="en")

    assert result == "hello world"
    assert fake_model.received_audio.dtype == np.float32
    assert fake_model.received_audio.max() <= 1.0
    assert fake_model.received_audio.min() >= -1.0


def test_pcm16_bytes_to_float32_range():
    pcm = np.array([-32768, 0, 32767], dtype=np.int16).tobytes()
    audio = voice_stt.pcm16_bytes_to_float32(pcm)
    assert audio[0] == -1.0
    assert audio[1] == 0.0
    assert 0.99 < audio[2] <= 1.0


def test_load_model_caches_by_key(monkeypatch):
    voice_stt.reset_model_cache()
    calls = []

    class _FakeWhisperModelClass:
        def __init__(self, model_size, device, compute_type):
            calls.append((model_size, device, compute_type))

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = _FakeWhisperModelClass
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    voice_stt._load_model("small", "cpu", "int8")
    voice_stt._load_model("small", "cpu", "int8")
    voice_stt._load_model("base", "cpu", "int8")

    assert calls == [("small", "cpu", "int8"), ("base", "cpu", "int8")]
    voice_stt.reset_model_cache()
