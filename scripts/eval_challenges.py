"""Build a challenge set and run baseline MMS-LID-126 (no improvements yet)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"C:\aie1903\hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"C:\aie1903\hf\hub")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

MODEL_ID = "facebook/mms-lid-126"
TARGET_SR = 16000
TOP_K = 3
RNG = np.random.default_rng(0)

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "samples"
CHALLENGE_DIR = ROOT / "challenges"
CHALLENGE_DIR.mkdir(parents=True, exist_ok=True)

CMN_WAV = SAMPLE_DIR / "aishell_cmn_BAC009S0770W0259.wav"
ENG_WAV = SAMPLE_DIR / "librispeech_eng_6930-75918-0000.wav"


def load_16k(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    if int(sr) != TARGET_SR:
        raise ValueError("Expected 16 kHz sample: " + str(path))
    return np.asarray(x, dtype=np.float32)


def speech_start(x: np.ndarray) -> int:
    """First frame whose RMS is at least 10% of max RMS (Assignment 2 energy idea)."""
    frame = 400
    hop = 160
    if x.shape[0] < frame:
        return 0
    n = 1 + (x.shape[0] - frame) // hop
    rms = np.array(
        [np.sqrt(np.mean(x[i * hop : i * hop + frame] ** 2)) for i in range(n)],
        dtype=np.float64,
    )
    thr = 0.10 * float(np.max(rms))
    for i, v in enumerate(rms):
        if v >= thr:
            return int(i * hop)
    return 0


def crop_speech(x: np.ndarray, seconds: float) -> np.ndarray:
    start = speech_start(x)
    n = int(seconds * TARGET_SR)
    y = x[start : start + n]
    if y.shape[0] < n:
        y = np.pad(y, (0, n - y.shape[0]))
    return y.astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    power = float(np.mean(x ** 2) + 1e-12)
    noise = RNG.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    noise_power = float(np.mean(noise ** 2) + 1e-12)
    scale = np.sqrt(power / (noise_power * (10.0 ** (snr_db / 10.0))))
    y = x + (noise * scale)
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def save(name: str, x: np.ndarray) -> Path:
    path = CHALLENGE_DIR / (name + ".wav")
    sf.write(str(path), x, TARGET_SR)
    return path


def topk(logits: torch.Tensor, id2label: dict, k: int) -> list[tuple[str, float]]:
    probs = torch.softmax(logits.float(), dim=-1)[0]
    values, indices = torch.topk(probs, k=k)
    return [(str(id2label[int(i)]), float(s)) for s, i in zip(values.tolist(), indices.tolist())]


def main() -> None:
    missing = [p.name for p in (CMN_WAV, ENG_WAV) if not p.is_file()]
    if missing:
        print("Missing sample wavs in", SAMPLE_DIR)
        print("Need:", ", ".join(missing))
        print("Rebuild them with run_lid.py after setting AISHELL_CACHE and LIBRI_PARQUET.")
        sys.exit(1)
    cmn = load_16k(CMN_WAV)
    eng = load_16k(ENG_WAV)
    cmn_1s = crop_speech(cmn, 1.0)
    eng_1s = crop_speech(eng, 1.0)

    clips: list[tuple[str, str, np.ndarray]] = [
        ("cmn_clean", "cmn", cmn),
        ("eng_clean", "eng", eng),
        ("cmn_0.5s", "cmn", crop_speech(cmn, 0.5)),
        ("cmn_1.0s", "cmn", cmn_1s),
        ("eng_0.5s", "eng", crop_speech(eng, 0.5)),
        ("eng_1.0s", "eng", eng_1s),
        ("cmn_lead_silence_2s", "cmn", np.concatenate([np.zeros(2 * TARGET_SR, dtype=np.float32), cmn])),
        ("cmn_snr5", "cmn", add_noise(cmn, 5.0)),
        ("eng_snr5", "eng", add_noise(eng, 5.0)),
        ("switch_cmn_then_eng", "mixed", np.concatenate([cmn_1s, eng_1s])),
        ("switch_eng_then_cmn", "mixed", np.concatenate([eng_1s, cmn_1s])),
    ]

    print("Loading model:", MODEL_ID)
    torch.set_num_threads(1)
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    print("Model loaded. Device: CPU")
    print()
    print(
        f"{'clip':<24} {'gold':<7} {'dur':>6} {'top1':<6} {'score':>8}  correct  top3"
    )
    print("-" * 88)

    rows = []
    n_ok = 0
    n_single = 0
    for name, gold, wav in clips:
        path = save(name, wav)
        dur = wav.shape[0] / float(TARGET_SR)
        inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        ranked = topk(logits, model.config.id2label, TOP_K)
        top1, score = ranked[0]
        if gold == "mixed":
            flag = "n/a"
        else:
            n_single += 1
            ok = top1 == gold
            n_ok += int(ok)
            flag = "YES" if ok else "NO"
        top3 = ", ".join(f"{a}:{b:.3f}" for a, b in ranked)
        line = f"{name:<24} {gold:<7} {dur:6.2f} {top1:<6} {score:8.4f}  {flag:<7}  {top3}"
        print(line)
        rows.append(line + "  file=" + str(path))

    print()
    if n_single:
        print("Single-language accuracy:", f"{n_ok}/{n_single} = {100.0 * n_ok / n_single:.1f}%")
    print("mixed clips have no single gold label; low score or flip is the point.")

    out = ROOT.parent / "docs" / "results" / "challenges_baseline.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "Baseline MMS-LID-126 on challenge set (no sliding window / reject yet).\n\n"
        + "\n".join(rows)
        + f"\n\nSingle-language accuracy: {n_ok}/{n_single}\n",
        encoding="utf-8",
    )
    print("Wrote", out)


if __name__ == "__main__":
    main()
