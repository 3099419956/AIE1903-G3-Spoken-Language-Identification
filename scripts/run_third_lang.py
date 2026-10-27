"""Third language check: Korean from MINDS-14 vs existing Mandarin/English clips."""
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"C:\aie1903\hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"C:\aie1903\hf\hub")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

MODEL_ID = "facebook/mms-lid-126"
TARGET_SR = 16000
MAX_SECONDS = 8.0
TOP_K = 5
MIN_SECONDS = 2.0

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
CMN_WAV = SAMPLE_DIR / "aishell_cmn_BAC009S0770W0259.wav"
ENG_WAV = SAMPLE_DIR / "librispeech_eng_6930-75918-0000.wav"


def load_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return to_16k_mono(np.asarray(x, dtype=np.float32), int(sr))


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


def decode_audio_cell(cell) -> tuple[np.ndarray, int]:
    audio = cell.as_py()
    if not isinstance(audio, dict):
        raise TypeError("Expected audio struct, got " + type(audio).__name__)
    sr = int(audio.get("sampling_rate") or audio.get("sample_rate") or 0)
    raw = audio.get("bytes")
    if raw is None and isinstance(audio.get("array"), (bytes, bytearray)):
        raw = audio["array"]
    if raw is None:
        raise ValueError("No audio bytes in parquet cell. keys=" + str(list(audio.keys())))
    wav, file_sr = sf.read(io.BytesIO(raw), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr <= 0:
        sr = int(file_sr)
    return np.asarray(wav, dtype=np.float32), sr


def load_korean() -> tuple[str, np.ndarray, int]:
    parquet_path = hf_hub_download(
        repo_id="PolyAI/minds14",
        filename="ko-KR/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path)
    names = set(table.column_names)
    path_col = "path" if "path" in names else None
    n = table.num_rows
    chosen = None
    for i in range(n):
        wav, sr = decode_audio_cell(table["audio"][i])
        dur = wav.shape[0] / float(sr)
        if dur < MIN_SECONDS:
            continue
        label = "row" + str(i)
        if path_col:
            label = str(table[path_col][i].as_py())
        chosen = (label, wav, sr, dur)
        break
    if chosen is None:
        wav, sr = decode_audio_cell(table["audio"][0])
        label = "row0"
        if path_col:
            label = str(table[path_col][0].as_py())
        chosen = (label, wav, sr, wav.shape[0] / float(sr))
    label, wav, sr, _dur = chosen
    stem = Path(str(label).replace("\\", "/")).stem or "minds14_kor"
    return stem, wav, sr


def topk_languages(logits: torch.Tensor, id2label: dict, k: int) -> list[tuple[str, float]]:
    probs = torch.softmax(logits.float(), dim=-1)[0]
    values, indices = torch.topk(probs, k=k)
    return [(str(id2label[int(i)]), float(s)) for s, i in zip(values.tolist(), indices.tolist())]


def predict_one(name: str, gold: str, wav: np.ndarray, processor, model) -> str:
    out_wav = SAMPLE_DIR / (name + ".wav")
    sf.write(str(out_wav), wav, TARGET_SR)
    duration_s = wav.shape[0] / float(TARGET_SR)
    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    ranked = topk_languages(logits, model.config.id2label, TOP_K)
    top1, score = ranked[0]
    flag = "YES" if top1 == gold else "NO"
    top5 = ", ".join(f"{a}:{b:.4f}" for a, b in ranked)
    line = (
        f"{name:<40} gold={gold:<4} dur={duration_s:5.2f}s  "
        f"top1={top1:<6} {score:8.4f}  {flag}  top5={top5}"
    )
    print(line)
    print("Saved:", out_wav)
    return line + "  file=" + str(out_wav)


def main() -> None:
    missing = [p.name for p in (CMN_WAV, ENG_WAV) if not p.is_file()]
    if missing:
        print("Missing sample wavs in", SAMPLE_DIR)
        print("Need:", ", ".join(missing))
        sys.exit(1)
    torch.set_num_threads(1)
    print("Loading model:", MODEL_ID)
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    print("Model loaded. Device: CPU")
    print("Korean source: PolyAI/minds14 ko-KR (one clip only)")
    print()

    kor_id, kor_wav, kor_sr = load_korean()
    kor_16k = to_16k_mono(kor_wav, kor_sr)
    kor_name = "minds14_kor_" + kor_id

    rows = []
    print(f"{'clip':<40} gold  duration   top1           ok   top5")
    print("-" * 120)
    rows.append(predict_one("aishell_cmn_BAC009S0770W0259", "cmn", load_wav(CMN_WAV), processor, model))
    rows.append(predict_one("librispeech_eng_6930-75918-0000", "eng", load_wav(ENG_WAV), processor, model))
    rows.append(predict_one(kor_name, "kor", kor_16k, processor, model))

    out = ROOT.parent / "docs" / "results" / "third_lang.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "Three-language check with MMS-LID-126 (cmn / eng / kor).\n"
        "Korean clip is ONE utterance from PolyAI/minds14 ko-KR, not the full set.\n\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    print()
    print("Wrote", out)


if __name__ == "__main__":
    main()
