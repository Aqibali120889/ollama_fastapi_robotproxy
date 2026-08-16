import argparse
import csv
import json
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.voice_denoise import denoise
from app.voice_stt import transcribe


def _read_wav_pcm16_mono(path: Path, expected_sample_rate: int) -> bytes:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"{path} is not mono")
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        if wf.getframerate() != expected_sample_rate:
            raise ValueError(f"{path} sample rate {wf.getframerate()} != expected {expected_sample_rate}")
        return wf.readframes(wf.getnframes())


def main():
    parser = argparse.ArgumentParser(description="Compare Whisper WER with and without DeepFilterNet denoising")
    parser.add_argument("--dataset", default="evaluation/stt_eval_dataset.json")
    parser.add_argument("--out-dir", default="artifacts/evaluation")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    dataset_path = root / args.dataset
    out_dir = root / args.out_dir

    settings = get_settings()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    if not dataset:
        print(
            f"No entries in {args.dataset}. Add {{'audio_path': ..., 'transcript': ...}} pairs "
            "(mono 16-bit PCM WAV at settings.voice_sample_rate, ideally recorded under the "
            "~65dB simulated background noise condition) before this produces meaningful numbers."
        )
        return

    try:
        from jiwer import wer
    except ImportError:
        print("jiwer is not installed. Run: pip install -r requirements-voice.txt")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in dataset:
        audio_path = root / item["audio_path"]
        reference = item["transcript"]
        audio_bytes = _read_wav_pcm16_mono(audio_path, settings.voice_sample_rate)

        hypothesis_raw = transcribe(
            audio_bytes,
            settings.stt_model_size,
            settings.stt_device,
            settings.stt_compute_type,
            settings.stt_language,
        )
        denoised_bytes = denoise(audio_bytes, settings.voice_sample_rate)
        hypothesis_denoised = transcribe(
            denoised_bytes,
            settings.stt_model_size,
            settings.stt_device,
            settings.stt_compute_type,
            settings.stt_language,
        )

        rows.append(
            {
                "audio_path": item["audio_path"],
                "reference": reference,
                "hypothesis_raw": hypothesis_raw,
                "wer_raw": round(wer(reference, hypothesis_raw), 4),
                "hypothesis_denoised": hypothesis_denoised,
                "wer_denoised": round(wer(reference, hypothesis_denoised), 4),
            }
        )

    avg_raw = sum(r["wer_raw"] for r in rows) / len(rows)
    avg_denoised = sum(r["wer_denoised"] for r in rows) / len(rows)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"stt_wer_{ts}.json"
    csv_path = out_dir / f"stt_wer_{ts}.csv"

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "wer_raw_avg": round(avg_raw, 4),
        "wer_denoised_avg": round(avg_denoised, 4),
        "improvement": round(avg_raw - avg_denoised, 4),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"WER without denoising:   {avg_raw:.2%}")
    print(f"WER with DeepFilterNet:  {avg_denoised:.2%}")
    print(f"Improvement:             {avg_raw - avg_denoised:+.2%}")
    print(f"\nSaved:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
