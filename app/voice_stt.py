import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_model_key: Optional[tuple] = None


def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def _load_model(model_size: str, device: str, compute_type: str):
    """Loads faster-whisper once and keeps it resident (mirrors the Ollama
    keep_alive warm-model strategy) -- reloading per request would blow the
    latency budget."""
    global _model, _model_key
    key = (model_size, device, compute_type)
    if _model is not None and _model_key == key:
        return _model

    from faster_whisper import WhisperModel  # imported lazily: heavy, optional dependency

    logger.info("Loading faster-whisper model %s (device=%s, compute_type=%s)", model_size, device, compute_type)
    _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _model_key = key
    return _model


def transcribe(
    audio_bytes: bytes,
    model_size: str = "small",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = "en",
) -> str:
    """Transcribes PCM16 mono audio bytes (any sample rate faster-whisper accepts
    as a raw array -- caller is expected to pass 16kHz per Whisper's native rate)."""
    if not audio_bytes:
        return ""

    model = _load_model(model_size, device, compute_type)
    audio = pcm16_bytes_to_float32(audio_bytes)

    segments, _info = model.transcribe(audio, language=language, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


def reset_model_cache() -> None:
    """Test helper: forces the next transcribe() call to reload the model."""
    global _model, _model_key
    _model = None
    _model_key = None
