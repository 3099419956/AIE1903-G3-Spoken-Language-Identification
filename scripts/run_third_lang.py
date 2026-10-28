"""Third language check: one Korean clip from MINDS-14, plus the cmn/eng samples."""

import argparse
import io
import sys
from pathlib import Path

from runtime import configure_cache

configure_cache()

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

from lid import DEFAULT_TOP_K, TARGET_SR, load_16k, load_model, predict, to_16k_mono

MIN_SECONDS = 2.0
ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "samples"
CMN_WAV = SAMPLE_DIR / "aishell_cmn_BAC009S0770W0259.wav"
ENG_WAV = SAMPLE_DIR / "librispeech_eng_6930-75918-0000.wav"


def decode_audio_cell(cell) -> tuple:
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


def load_korean() -> tuple:
    parquet_path = hf_hub_download(
        repo_id="PolyAI/minds14",
        filename="ko-KR/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path)
    path_col = "path" if "path" in table.column_names else None
    chosen = None
    for i in range(table.num_rows):
        wav, sr = decode_audio_cell(table["audio"][i])
        if wav.shape[0] / float(sr) < MIN_SECONDS:
            continue
        label = str(table[path_col][i].as_py()) if path_col else "row" + str(i)
        chosen = (label, wav, sr)
        break
    if chosen is None:
        wav, sr = decode_audio_cell(table["audio"][0])
        label = str(table[path_col][0].as_py()) if path_col else "row0"
        chosen = (label, wav, sr)
    label, wav, sr = chosen
    stem = Path(str(label).replace("\\", "/")).stem or "minds14_kor"
    return stem, wav, sr


def predict_one(name: str, gold: str, wav, model, processor, top_k: int) -> str:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_wav = SAMPLE_DIR / (name + ".wav")
    sf.write(str(out_wav), wav, TARGET_SR)
    duration_s = wav.shape[0] / float(TARGET_SR)
    ranked = predict(model, processor, wav, top_k=top_k)
    top1, score = ranked[0]
    flag = "YES" if top1 == gold else "NO"
    topk = ", ".join(f"{a}:{b:.4f}" for a, b in ranked)
    line = (
        f"{name:<40} gold={gold:<4} dur={duration_s:5.2f}s  "
        f"top1={top1:<6} {score:8.4f}  {flag}  top5={topk}"
    )
    print(line)
    print("Saved:", out_wav)
    return line + "  file=" + str(out_wav)


def main() -> None:
    ap = argparse.ArgumentParser(description="Three-language check (cmn / eng / kor)")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()

    missing = [p.name for p in (CMN_WAV, ENG_WAV) if not p.is_file()]
    if missing:
        print("Missing sample wavs in", SAMPLE_DIR)
        print("Need:", ", ".join(missing))
        sys.exit(1)

    model, processor, device = load_model(args.model_dir)
    print("Model loaded. Device:", device)
    print("Korean source: PolyAI/minds14 ko-KR (one clip only)")
    print()

    kor_id, kor_wav, kor_sr = load_korean()
    kor_16k = to_16k_mono(kor_wav, kor_sr)
    kor_name = "minds14_kor_" + kor_id

    cmn_16k, _ = load_16k(CMN_WAV)
    eng_16k, _ = load_16k(ENG_WAV)

    rows = []
    print(f"{'clip':<40} gold  duration   top1           ok   top5")
    print("-" * 120)
    rows.append(predict_one("aishell_cmn_BAC009S0770W0259", "cmn", cmn_16k, model, processor, args.top_k))
    rows.append(predict_one("librispeech_eng_6930-75918-0000", "eng", eng_16k, model, processor, args.top_k))
    rows.append(predict_one(kor_name, "kor", kor_16k, model, processor, args.top_k))

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
