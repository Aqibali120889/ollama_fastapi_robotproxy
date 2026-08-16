from app import voice_tts


class _FakeSynthesizer:
    output_sample_rate = 24000


class _FakeTTS:
    def __init__(self):
        self.synthesizer = _FakeSynthesizer()
        self.calls = []

    def tts(self, text, speaker=None, language=None):
        self.calls.append((text, speaker, language))
        return [0.0, 0.5, -0.5, 1.0, -1.0]


def test_synthesize_blank_text_returns_empty():
    audio, sample_rate = voice_tts.synthesize("   ")
    assert audio == b""
    assert sample_rate == 0


def test_synthesize_calls_model_and_converts_audio(monkeypatch):
    fake_tts = _FakeTTS()
    monkeypatch.setattr(voice_tts, "_load_tts", lambda *a, **k: fake_tts)

    audio_bytes, sample_rate = voice_tts.synthesize("hello there", speaker="Voice A", language="en")

    assert sample_rate == 24000
    assert fake_tts.calls == [("hello there", "Voice A", "en")]
    assert len(audio_bytes) == 5 * 2  # 5 float samples -> 5 int16 samples -> 10 bytes


def test_load_tts_caches_by_key(monkeypatch):
    voice_tts.reset_model_cache()
    calls = []

    class _FakeTTSClass:
        def __init__(self, model_name):
            calls.append(model_name)

        def to(self, device):
            calls.append(device)
            return self

    import sys
    import types

    fake_module = types.ModuleType("TTS.api")
    fake_module.TTS = _FakeTTSClass
    monkeypatch.setitem(sys.modules, "TTS.api", fake_module)
    monkeypatch.setitem(sys.modules, "TTS", types.ModuleType("TTS"))

    voice_tts._load_tts("model-a", "cpu")
    voice_tts._load_tts("model-a", "cpu")
    voice_tts._load_tts("model-b", "cpu")

    assert calls == ["model-a", "cpu", "model-b", "cpu"]
    voice_tts.reset_model_cache()
