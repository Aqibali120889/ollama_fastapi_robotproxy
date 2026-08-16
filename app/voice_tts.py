import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_tts = None
_tts_key: Optional[tuple] = None


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _load_tts(model_name: str, device: str):
    """Loads Coqui XTTS-v2 once and keeps it resident, same warm-model
    principle as the STT/LLM/denoise models -- per-request loads would blow
    the latency budget."""
    global _tts, _tts_key
    key = (model_name, device)
    if _tts is not None and _tts_key == key:
        return _tts

    from TTS.api import TTS  # imported lazily: heavy, optional dependency

    logger.info("Loading Coqui TTS model %s (device=%s)", model_name, device)
    _tts = TTS(model_name).to(device)
    _tts_key = key
    return _tts


def synthesize(
    text: str,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    device: str = "cuda",
    speaker: str = "Claribel Dervla",
    language: str = "en",
) -> Tuple[bytes, int]:
    """Synthesizes speech for the given text using XTTS-v2's built-in speaker.
    Returns (pcm16_audio_bytes, sample_rate)."""
    if not text.strip():
        return b"", 0

    tts = _load_tts(model_name, device)
    wav = tts.tts(text=text, speaker=speaker, language=language)
    sample_rate = tts.synthesizer.output_sample_rate

    audio = np.asarray(wav, dtype=np.float32)
    return float32_to_pcm16_bytes(audio), sample_rate


def reset_model_cache() -> None:
    """Test helper: forces the next synthesize() call to reload the model."""
    global _tts, _tts_key
    _tts = None
    _tts_key = None
