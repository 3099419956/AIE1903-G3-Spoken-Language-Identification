"""Shared audio/model/prediction helpers for the G3 LID project.

The entry scripts (predict.py, run_lid.py, run_third_lang.py,
eval_challenges.py, eval_improved.py, demo_server.py and eval_pipeline.py)
reuse the functions here so audio loading, model loading and prediction live
in exactly one place.

Model directory resolution order:
    1. explicit ``model_dir`` argument
    2. ``MMS_LID_MODEL_DIR`` environment variable
    3. ``<repo>/models/mms-lid-126``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

TARGET_SR = 16000
DEFAULT_TOP_K = 5
MAX_SECONDS = 8.0

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "mms-lid-126"


def resolve_model_dir(model_dir: Optional[str] = None) -> str:
    if model_dir:
        return str(model_dir)
    return os.environ.get("MMS_LID_MODEL_DIR") or str(DEFAULT_MODEL_DIR)


def _model_device(model):
    import torch

    return next(model.parameters()).device


def load_model(model_dir: Optional[str] = None, device: str = "auto"):
    """Load mms-lid-126 from a local directory, never from the network."""
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    resolved = resolve_model_dir(model_dir)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(
            "Model directory not found: " + resolved
            + "\nSet MMS_LID_MODEL_DIR or pass --model_dir to the local mms-lid-126 folder."
        )
    processor = AutoFeatureExtractor.from_pretrained(resolved, local_files_only=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(resolved, local_files_only=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, processor, device


def read_wav(path) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def to_16k_mono(waveform, sample_rate: int, max_seconds: Optional[float] = MAX_SECONDS) -> np.ndarray:
    """Convert an in-memory waveform to 16 kHz mono float32, capped to max_seconds."""
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim > 1:
        x = x[:, 0]
    if int(sample_rate) != TARGET_SR:
        import librosa

        x = librosa.resample(y=x, orig_sr=int(sample_rate), target_sr=TARGET_SR)
        x = np.asarray(x, dtype=np.float32)
    if max_seconds is not None:
        n = int(max_seconds * TARGET_SR)
        if x.shape[0] > n:
            x = x[:n]
    return x


def load_16k(path, max_seconds: Optional[float] = MAX_SECONDS) -> Tuple[np.ndarray, int]:
    wav, sr = read_wav(path)
    return to_16k_mono(wav, sr, max_seconds=max_seconds), sr


def topk_languages(logits, id2label: dict, k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
    import torch

    probs = torch.softmax(logits.float(), dim=-1)[0]
    values, indices = torch.topk(probs, k=k)
    return [(str(id2label[int(i)]), float(s)) for s, i in zip(values.tolist(), indices.tolist())]


def predict(model, processor, audio: np.ndarray, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
    """Whole-clip inference. Returns [(lang_code, probability), ...]."""
    import torch

    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {k: v.to(_model_device(model)) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return topk_languages(logits, model.config.id2label, top_k)


# ---- challenge-set helpers (used by eval_challenges.py) ----


def speech_start(x: np.ndarray, frame: int = 400, hop: int = 160, thr_ratio: float = 0.10) -> int:
    """First frame whose RMS is at least thr_ratio of the max RMS."""
    if x.shape[0] < frame:
        return 0
    n = 1 + (x.shape[0] - frame) // hop
    rms = np.array(
        [np.sqrt(np.mean(x[i * hop : i * hop + frame] ** 2)) for i in range(n)],
        dtype=np.float64,
    )
    thr = thr_ratio * float(np.max(rms))
    for i, v in enumerate(rms):
        if v >= thr:
            return int(i * hop)
    return 0


def crop_speech(x: np.ndarray, seconds: float, start: Optional[int] = None) -> np.ndarray:
    start = speech_start(x) if start is None else start
    n = int(seconds * TARGET_SR)
    y = x[start : start + n]
    if y.shape[0] < n:
        y = np.pad(y, (0, n - y.shape[0]))
    return y.astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng(0)
    power = float(np.mean(x ** 2) + 1e-12)
    noise = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    noise_power = float(np.mean(noise ** 2) + 1e-12)
    scale = np.sqrt(power / (noise_power * (10.0 ** (snr_db / 10.0))))
    y = x + (noise * scale)
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


# ---- sliding window + reject (used by eval_improved.py) ----


def window_slices(n: int, win: int, hop: int, start: int = 0) -> List[Tuple[int, int]]:
    if n - start <= win:
        return [(start, n)]
    starts = list(range(start, n - win + 1, hop))
    last = n - win
    if last - starts[-1] >= hop // 2:
        starts.append(last)
    return [(s, s + win) for s in starts]


def decode_top1(logits, id2label: dict) -> Tuple[str, float]:
    import torch

    probs = torch.softmax(logits.float(), dim=-1)[0]
    idx = int(torch.argmax(probs).item())
    return str(id2label[idx]), float(probs[idx].item())


def _logits(model, processor, wav: np.ndarray):
    import torch

    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {k: v.to(_model_device(model)) for k, v in inputs.items()}
    with torch.no_grad():
        return model(**inputs).logits


def sliding_predict(
    model,
    processor,
    id2label: dict,
    wav: np.ndarray,
    win_s: float = 1.0,
    hop_s: float = 0.5,
    min_score: float = 0.5,
) -> dict:
    """1 s sliding window, 0.5 s hop, drop windows below min_score.

    Falls back to one whole-clip forward pass when every window is dropped.
    Returns {"decision", "langs", "windows", "fallback"}.
    """
    win = int(win_s * TARGET_SR)
    hop = int(hop_s * TARGET_SR)
    n = int(wav.shape[0])

    if n <= win:
        lang, score = decode_top1(_logits(model, processor, wav), id2label)
        keep = score >= min_score
        windows = [
            {
                "start_s": 0.0,
                "end_s": round(n / TARGET_SR, 3),
                "lang": lang,
                "score": score,
                "keep": keep,
            }
        ]
        if keep:
            return {"decision": lang, "langs": [lang], "windows": windows, "fallback": False}
        return {"decision": "uncertain", "langs": [], "windows": windows, "fallback": False}

    onset = speech_start(wav)
    windows = []
    for start, end in window_slices(n, win, hop, start=onset):
        lang, score = decode_top1(_logits(model, processor, wav[start:end]), id2label)
        windows.append(
            {
                "start_s": round(start / TARGET_SR, 3),
                "end_s": round(end / TARGET_SR, 3),
                "lang": lang,
                "score": score,
                "keep": score >= min_score,
            }
        )

    kept = [w for w in windows if w["keep"]]
    fallback = False
    if not kept:
        lang, score = decode_top1(_logits(model, processor, wav), id2label)
        fallback = True
        windows.append(
            {
                "start_s": 0.0,
                "end_s": round(wav.shape[0] / TARGET_SR, 3),
                "lang": lang,
                "score": score,
                "keep": score >= min_score,
            }
        )
        if score >= min_score:
            decision, langs = lang, [lang]
        else:
            decision, langs = "uncertain", []
    else:
        langs = []
        for w in kept:
            if w["lang"] not in langs:
                langs.append(w["lang"])
        decision = "mixed:" + "+".join(langs) if len(langs) > 1 else langs[0]

    return {"decision": decision, "langs": langs, "windows": windows, "fallback": fallback}


def judge(gold: str, decision: str) -> str:
    if gold == "mixed":
        has_cmn = "cmn" in decision
        has_eng = "eng" in decision
        if has_cmn and has_eng:
            return "HIT"
        if has_cmn or has_eng:
            return "PARTIAL"
        return "MISS"
    if decision == "uncertain":
        return "REJECT"
    if decision == gold:
        return "HIT"
    return "MISS"
