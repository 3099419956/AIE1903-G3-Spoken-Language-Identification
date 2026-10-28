"""Whole-clip LID on AISHELL (cmn) and LibriSpeech (eng), using shared helpers."""

import argparse
import io
import os
import sys
from pathlib import Path

from runtime import configure_cache

CACHE_LOCATION = configure_cache()

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from lid import DEFAULT_TOP_K, TARGET_SR, load_model, predict, read_wav, resolve_model_dir, to_16k_mono

AISHELL_ID = "BAC009S0770W0259"
LIBRI_ID = "6930-75918-0000"
ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "samples"


def find_aishell_wav(cache: Path) -> Path:
    matches = list(Path(cache).rglob(AISHELL_ID + ".wav"))
    if not matches:
        raise FileNotFoundError("AISHELL WAV not found: " + AISHELL_ID)
    return matches[0]


def load_librispeech(parquet_path: Path) -> tuple:
    table = pq.read_table(parquet_path)
    ids = table["id"].to_pylist()
    row = ids.index(LIBRI_ID)
    audio_bytes = table["audio"][row].as_py()["bytes"]
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def predict_one(name: str, waveform, sample_rate: int, model, processor, top_k: int) -> None:
    x = to_16k_mono(waveform, sample_rate)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_wav = SAMPLE_DIR / (name + ".wav")
    sf.write(str(out_wav), x, TARGET_SR)
    duration_s = x.shape[0] / float(TARGET_SR)
    ranked = predict(model, processor, x, top_k=top_k)

    print("Clip:", name)
    print("Saved sample:", out_wav)
    print("Duration used (seconds):", round(duration_s, 3))
    print("Top-1 language:", ranked[0][0], "score:", f"{ranked[0][1]:.6f}")
    print("Top-%d:" % top_k)
    for i, (lang, score) in enumerate(ranked, start=1):
        print(f"  {i}. {lang}  {score:.6f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="LID on AISHELL + LibriSpeech")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()

    cache = os.environ.get("AISHELL_CACHE", "")
    parquet = os.environ.get("LIBRI_PARQUET", "")
    if not cache or not parquet:
        print("This script rebuilds two sample wavs from the original datasets.")
        print("Those datasets are not in the repository.")
        print("Set AISHELL_CACHE to the folder that contains " + AISHELL_ID + ".wav")
        print("Set LIBRI_PARQUET to the LibriSpeech test.clean parquet file.")
        print("To score a wav you already have, use: python predict.py path\\to\\file.wav")
        sys.exit(1)

    print("HF_HOME:", CACHE_LOCATION)
    print("Model dir:", resolve_model_dir(args.model_dir))
    model, processor, device = load_model(args.model_dir)
    print("Model loaded. Device:", device)
    print()

    aishell_wav, aishell_sr = read_wav(find_aishell_wav(Path(cache)))
    predict_one("aishell_cmn_" + AISHELL_ID, aishell_wav, aishell_sr, model, processor, args.top_k)

    libri_wav, libri_sr = load_librispeech(Path(parquet))
    predict_one("librispeech_eng_" + LIBRI_ID, libri_wav, libri_sr, model, processor, args.top_k)


if __name__ == "__main__":
    main()
