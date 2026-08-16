import logging

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_df_state = None


def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _load_model():
    """Loads DeepFilterNet once and keeps it resident, same warm-model
    principle as the STT/LLM/TTS models."""
    global _model, _df_state
    if _model is not None:
        return _model, _df_state

    from df.enhance import init_df  # imported lazily: heavy, optional dependency

    logger.info("Loading DeepFilterNet model")
    _model, _df_state, _ = init_df()
    return _model, _df_state


def _model_sample_rate(df_state) -> int:
    return df_state.sr()


def _run_enhance(model, df_state, audio: np.ndarray, sample_rate: int, df_sample_rate: int) -> np.ndarray:
    """The one seam where torch/torchaudio/df.enhance are actually touched --
    numpy in, numpy out, so denoise() itself has no hard dependency on them
    and callers can unit-test the surrounding logic without those packages
    installed."""
    import torch
    from df.enhance import enhance  # imported lazily: heavy, optional dependency

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (channels=1, samples)

    if sample_rate != df_sample_rate:
        import torchaudio

        audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, df_sample_rate)

    enhanced = enhance(model, df_state, audio_tensor)

    if sample_rate != df_sample_rate:
        enhanced = torchaudio.functional.resample(enhanced, df_sample_rate, sample_rate)

    return enhanced.squeeze(0).detach().cpu().numpy()


def denoise(audio_bytes: bytes, sample_rate: int) -> bytes:
    """Runs DeepFilterNet noise suppression on PCM16 mono audio, returning
    cleaned PCM16 bytes at the same sample rate that was passed in."""
    if not audio_bytes:
        return audio_bytes

    model, df_state = _load_model()
    audio = pcm16_bytes_to_float32(audio_bytes)
    df_sample_rate = _model_sample_rate(df_state)

    enhanced = _run_enhance(model, df_state, audio, sample_rate, df_sample_rate)
    return float32_to_pcm16_bytes(enhanced)


def reset_model_cache() -> None:
    """Test helper: forces the next denoise() call to reload the model."""
    global _model, _df_state
    _model = None
    _df_state = None
