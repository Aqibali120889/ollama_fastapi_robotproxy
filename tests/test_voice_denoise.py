import numpy as np

from app import voice_denoise


class _FakeDFState:
    def __init__(self, sample_rate):
        self._sample_rate = sample_rate

    def sr(self):
        return self._sample_rate


def test_denoise_empty_bytes_returns_empty():
    assert voice_denoise.denoise(b"", 16000) == b""


def test_denoise_round_trips_pcm16_through_identity_enhance(monkeypatch):
    fake_state = _FakeDFState(sample_rate=16000)
    monkeypatch.setattr(voice_denoise, "_load_model", lambda: (object(), fake_state))
    monkeypatch.setattr(
        voice_denoise,
        "_run_enhance",
        lambda model, df_state, audio, sample_rate, df_sample_rate: audio,
    )

    pcm = np.array([0, 16384, -16384, 1000], dtype=np.int16).tobytes()
    result = voice_denoise.denoise(pcm, sample_rate=16000)

    result_arr = np.frombuffer(result, dtype=np.int16)
    original_arr = np.frombuffer(pcm, dtype=np.int16)
    assert np.allclose(result_arr, original_arr, atol=1)


def test_denoise_forwards_both_sample_rates_to_enhance(monkeypatch):
    fake_state = _FakeDFState(sample_rate=48000)
    monkeypatch.setattr(voice_denoise, "_load_model", lambda: (object(), fake_state))

    captured = {}

    def fake_run_enhance(model, df_state, audio, sample_rate, df_sample_rate):
        captured["sample_rate"] = sample_rate
        captured["df_sample_rate"] = df_sample_rate
        return audio

    monkeypatch.setattr(voice_denoise, "_run_enhance", fake_run_enhance)

    pcm = np.array([100, -100], dtype=np.int16).tobytes()
    voice_denoise.denoise(pcm, sample_rate=16000)

    assert captured["sample_rate"] == 16000
    assert captured["df_sample_rate"] == 48000


def test_pcm16_float32_round_trip_helpers():
    pcm = np.array([-32768, 0, 32767], dtype=np.int16).tobytes()
    audio = voice_denoise.pcm16_bytes_to_float32(pcm)
    assert audio[0] == -1.0
    assert audio[1] == 0.0
    back = voice_denoise.float32_to_pcm16_bytes(audio)
    back_arr = np.frombuffer(back, dtype=np.int16)
    assert np.allclose(back_arr, np.frombuffer(pcm, dtype=np.int16), atol=1)
