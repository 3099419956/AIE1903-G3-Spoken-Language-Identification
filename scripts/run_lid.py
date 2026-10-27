"""Group 3 LID inference: facebook/mms-lid-126 on two Assignment 1/2 clips."""
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"C:\aie1903\hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"C:\aie1903\hf\hub")

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

MODEL_ID = "facebook/mms-lid-126"
TARGET_SR = 16000
MAX_SECONDS = 8.0
TOP_K = 5

AISHELL_ID = "BAC009S0770W0259"
LIBRI_ID = "6930-75918-0000"
ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def find_aishell_wav(cache: Path) -> Path:
    matches = list(cache.rglob(AISHELL_ID + ".wav"))
    if not matches:
        raise FileNotFoundError("AISHELL WAV not found: " + AISHELL_ID)
    return matches[0]


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(str(path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def load_librispeech(parquet_path: Path) -> tuple[np.ndarray, int]:
    table = pq.read_table(parquet_path)
    ids = table["id"].to_pylist()
    row = ids.index(LIBRI_ID)
    audio_bytes = table["audio"][row].as_py()["bytes"]
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def to_16k_mono(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float32)
    if sample_rate != TARGET_SR:
        import librosa

        x = librosa.resample(y=x, orig_sr=sample_rate, target_sr=TARGET_SR)
        x = np.asarray(x, dtype=np.float32)
    max_n = int(MAX_SECONDS * TARGET_SR)
    if x.shape[0] > max_n:
        x = x[:max_n]
    return x


def topk_languages(logits: torch.Tensor, id2label: dict, k: int) -> list[tuple[str, float]]:
    probs = torch.softmax(logits.float(), dim=-1)[0]
    values, indices = torch.topk(probs, k=k)
    out = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        out.append((str(id2label[int(idx)]), float(score)))
    return out


def predict_one(name: str, waveform: np.ndarray, sample_rate: int, processor, model) -> None:
    x = to_16k_mono(waveform, sample_rate)
    out_wav = SAMPLE_DIR / (name + ".wav")
    sf.write(str(out_wav), x, TARGET_SR)
    duration_s = x.shape[0] / float(TARGET_SR)

    inputs = processor(x, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    ranked = topk_languages(logits, model.config.id2label, TOP_K)

    print("Clip:", name)
    print("Saved sample:", out_wav)
    print("Duration used (seconds):", round(duration_s, 3))
    print("Top-1 language:", ranked[0][0], "score:", f"{ranked[0][1]:.6f}")
    print("Top-5:")
    for i, (lang, score) in enumerate(ranked, start=1):
        print(f"  {i}. {lang}  {score:.6f}")
    print()


def main() -> None:
    cache = os.environ.get("AISHELL_CACHE", "")
    parquet = os.environ.get("LIBRI_PARQUET", "")
    if not cache or not parquet:
        print("This script rebuilds two sample wavs from the original datasets.")
        print("Those datasets are not in the repository.")
        print("Set AISHELL_CACHE to the folder that contains " + AISHELL_ID + ".wav")
        print("Set LIBRI_PARQUET to the LibriSpeech test.clean parquet file.")
        print("To score a wav you already have, use: python predict.py path\\to\\file.wav")
        sys.exit(1)

    torch.set_num_threads(1)
    print("HF_HOME:", os.environ["HF_HOME"])
    print("Loading model:", MODEL_ID)
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    print("Model loaded. Device: CPU")
    print()

    aishell_wav, aishell_sr = load_wav(find_aishell_wav(Path(cache)))
    predict_one("aishell_cmn_" + AISHELL_ID, aishell_wav, aishell_sr, processor, model)

    libri_wav, libri_sr = load_librispeech(Path(parquet))
    predict_one("librispeech_eng_" + LIBRI_ID, libri_wav, libri_sr, processor, model)


if __name__ == "__main__":
    main()
