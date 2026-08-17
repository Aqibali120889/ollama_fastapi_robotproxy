import asyncio
import json
import os

import numpy as np
import pyaudio
import sounddevice as sd
import websockets
from dotenv import load_dotenv

load_dotenv()

# This client is intentionally thin: it only captures mic audio and plays back
# synthesized speech. Denoising, STT, RAG, LLM generation, and TTS all run
# server-side (see app/main.py's /ws/voice) on the GPU workstation -- the
# Jetson Nano does not have the compute budget to run any of that locally.
VOICE_WS_URL = os.getenv("VOICE_WS_URL", "ws://100.x.x.x:8000/ws/voice")
SAMPLE_RATE = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
CHUNK_SIZE = int(os.getenv("VOICE_CHUNK_SIZE", "1024"))

# Simple energy-threshold silence detection to segment one utterance at a
# time -- no ML-based VAD here on purpose, keep the edge client lightweight.
SILENCE_THRESHOLD = float(os.getenv("VOICE_SILENCE_THRESHOLD", "500"))
SILENCE_DURATION_S = float(os.getenv("VOICE_SILENCE_DURATION_S", "0.8"))
MIN_UTTERANCE_S = float(os.getenv("VOICE_MIN_UTTERANCE_S", "0.4"))
MAX_UTTERANCE_S = float(os.getenv("VOICE_MAX_UTTERANCE_S", "15.0"))

_SILENCE_CHUNKS_NEEDED = max(1, int(SILENCE_DURATION_S * SAMPLE_RATE / CHUNK_SIZE))
_MIN_CHUNKS = max(1, int(MIN_UTTERANCE_S * SAMPLE_RATE / CHUNK_SIZE))
_MAX_CHUNKS = max(_MIN_CHUNKS, int(MAX_UTTERANCE_S * SAMPLE_RATE / CHUNK_SIZE))


def _rms(chunk: bytes) -> float:
    audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def record_utterance(stream) -> bytes:
    """Blocks until one utterance has been captured: waits for audio above
    SILENCE_THRESHOLD, then keeps recording until a run of silence follows."""
    frames = []
    silent_run = 0
    heard_speech = False

    print("Listening...")
    while True:
        chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        level = _rms(chunk)

        if level >= SILENCE_THRESHOLD:
            heard_speech = True
            silent_run = 0
        else:
            silent_run += 1

        if heard_speech:
            frames.append(chunk)

        if heard_speech and silent_run >= _SILENCE_CHUNKS_NEEDED and len(frames) >= _MIN_CHUNKS:
            break
        if len(frames) >= _MAX_CHUNKS:
            break

    return b"".join(frames)


async def _handle_connection():
    async with websockets.connect(VOICE_WS_URL, open_timeout=5, max_size=None) as ws:
        session_msg = json.loads(await ws.recv())
        print(f"Connected. session_id={session_msg.get('session_id')}")

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        try:
            while True:
                utterance = await asyncio.to_thread(record_utterance, stream)
                if not utterance:
                    continue

                await ws.send(utterance)

                # Responses stream in as one sentence at a time (sentence_audio_meta
                # + a binary frame, repeated) rather than a single blob. Messages
                # arrive over the socket in order and this loop processes one at a
                # time, blocking on sd.wait() per chunk -- that alone is enough to
                # play sentences back-to-back correctly, no separate queue needed.
                playback_sample_rate = SAMPLE_RATE
                async for raw in ws:
                    if isinstance(raw, bytes):
                        if raw:
                            sd.play(np.frombuffer(raw, dtype=np.int16), samplerate=playback_sample_rate)
                            sd.wait()
                        continue

                    event = json.loads(raw)
                    event_type = event.get("type")

                    if event_type == "transcript":
                        print(f"Visitor said: {event['text']}")
                    elif event_type == "meta":
                        route = "knowledge base" if event.get("used_rag") else "general chat"
                        print(f"Robot is answering ({route})...")
                    elif event_type == "sentence_audio_meta":
                        playback_sample_rate = event["sample_rate"]
                        print(f"Robot: {event['text']}")
                    elif event_type == "error":
                        print(f"Voice pipeline error: {event['message']}")
                        break
                    elif event_type == "done":
                        break
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()


async def main():
    while True:
        try:
            await _handle_connection()
        except (websockets.ConnectionClosed, OSError) as exc:
            print(f"Connection lost ({exc}); reconnecting in 3s...")
            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
