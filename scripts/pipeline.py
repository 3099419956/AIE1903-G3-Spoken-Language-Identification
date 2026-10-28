"""Choose a light fix, then run the unchanged MMS-LID-126 model.

Clean speech of about 1 second or longer goes straight through.
Denoise and sliding windows run only when the checks say they are needed.
"""
from runtime import configure_cache

configure_cache()

import numpy as np
import torch

TARGET_SR = 16000
MAX_SECONDS = 8.0
MIN_SCORE = 0.50
SHORT_S = 1.0
SNR_GATE_DB = 15.0
FRAME = 400
HOP = 160
WIN_S = 1.0
HOP_S = 0.5
# Whole-clip top-2 looks like two languages, not one confident label.
SWITCH_TOP1_BELOW = 0.90
SWITCH_TOP2_AT_LEAST = 0.15
RELATED = {"cmn", "yue"}

_DENOISER = None


def to_mono_16k(wav: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
    x = np.asarray(wav, dtype=np.float32)
    channels = 1 if x.ndim == 1 else int(x.shape[1])
    if x.ndim > 1:
        x = np.mean(x, axis=1).astype(np.float32)
    note = f"{channels} ch, {int(sr)} Hz"
    if int(sr) != TARGET_SR:
        import librosa

        x = np.asarray(
            librosa.resample(y=x, orig_sr=int(sr), target_sr=TARGET_SR),
            dtype=np.float32,
        )
        note += " -> mono 16 kHz"
    else:
        note += " -> mono" if channels > 1 else " already mono 16 kHz"
    n = int(MAX_SECONDS * TARGET_SR)
    if x.shape[0] > n:
        x = x[:n]
        note += f", capped at {MAX_SECONDS:.0f}s"
    return x, note


def _frame_rms(x: np.ndarray) -> np.ndarray:
    n = int(x.shape[0])
    if n < FRAME:
        return np.array([float(np.sqrt(np.mean(x ** 2)))], dtype=np.float64)
    n_frames = 1 + (n - FRAME) // HOP
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * HOP : i * HOP + FRAME]
        rms[i] = float(np.sqrt(np.mean(seg ** 2)))
    return rms


def speech_bounds(x: np.ndarray) -> tuple[int, int] | None:
    """Cut only near-silent edges. Do not drop a quieter second language."""
    rms = _frame_rms(x)
    max_rms = float(np.max(rms)) if rms.size else 0.0
    if max_rms < 0.005:
        return None
    # Absolute floor: digital silence is ~0. A quiet language can sit well below
    # 10% of the louder language, and must not be treated as trailing silence.
    floor = 0.004
    idx = np.flatnonzero(rms >= floor)
    if idx.size == 0:
        return 0, int(x.shape[0])
    pad = int(0.05 * TARGET_SR)
    start = max(0, int(idx[0] * HOP) - pad)
    end = min(int(x.shape[0]), int(idx[-1] * HOP) + FRAME + pad)
    if end - start < int(0.2 * TARGET_SR):
        return None
    return start, end


def estimate_snr_db(x: np.ndarray) -> float:
    frame = FRAME
    hop = HOP
    if x.shape[0] < frame:
        return 99.0
    n = 1 + (x.shape[0] - frame) // hop
    power = np.array(
        [float(np.mean(x[i * hop : i * hop + frame] ** 2)) for i in range(n)],
        dtype=np.float64,
    )
    noise = max(float(np.percentile(power, 10)), 1e-12)
    speech = max(float(np.percentile(power, 90)) - noise, 1e-12)
    return float(10.0 * np.log10(speech / noise))


def maybe_gain(x: np.ndarray) -> tuple[np.ndarray, str]:
    """Boost only near-silent peaks. Laptop speech is already loud enough for MMS."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0.0:
        return x, "silent"
    if peak < 0.01:
        y = x * (0.2 / peak)
        return y.astype(np.float32), f"near-silent peak {peak:.4f} -> 0.20"
    if peak >= 0.99:
        return x, f"clipped peak {peak:.3f}, left unchanged"
    return x, "level ok, no gain"


def topk(model, processor, wav: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits.float(), dim=-1)[0]
    values, indices = torch.topk(probs, k=min(k, int(probs.shape[0])))
    return [
        (str(model.config.id2label[int(i)]), float(s))
        for s, i in zip(values.tolist(), indices.tolist())
    ]


def _windows(model, processor, wav: np.ndarray) -> list[dict]:
    win = int(WIN_S * TARGET_SR)
    hop = int(HOP_S * TARGET_SR)
    n = int(wav.shape[0])
    if n <= win:
        lang, score = topk(model, processor, wav, k=1)[0]
        return [{"start_s": 0.0, "end_s": n / TARGET_SR, "lang": lang, "score": score}]
    rows = []
    starts = list(range(0, n - win + 1, hop))
    last = n - win
    if last - starts[-1] >= hop // 2:
        starts.append(last)
    for start in starts:
        chunk = wav[start : start + win]
        lang, score = topk(model, processor, chunk, k=1)[0]
        rows.append(
            {
                "start_s": start / TARGET_SR,
                "end_s": (start + win) / TARGET_SR,
                "lang": lang,
                "score": score,
            }
        )
    return rows


def denoise_dns48(wav: np.ndarray) -> np.ndarray:
    global _DENOISER
    try:
        if _DENOISER is None:
            from denoiser.pretrained import dns48

            _DENOISER = dns48(pretrained=True)
            _DENOISER.eval()
        mix = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = _DENOISER(mix)
        y = out.squeeze().detach().cpu().numpy().astype(np.float32)
        return y[: wav.shape[0]]
    except Exception as exc:  # denoiser not installed or weights unavailable
        raise RuntimeError("denoiser unavailable: " + str(exc)) from exc


def _step(name: str, action: str, detail: str) -> dict:
    return {"name": name, "action": action, "detail": detail}


def run(model, processor, wav: np.ndarray, sr: int) -> dict:
    steps = []
    x, fmt = to_mono_16k(wav, sr)
    steps.append(_step("format", "done", fmt))

    bounds = speech_bounds(x)
    if bounds is None:
        steps.append(_step("speech", "stop", "no speech-like energy"))
        return {
            "decision": "no-speech",
            "langs": [],
            "top5": [],
            "duration_s": float(x.shape[0] / TARGET_SR),
            "snr_db": None,
            "steps": steps,
            "windows": [],
            "related_warning": False,
        }

    start, end = bounds
    trimmed = x[start:end]
    cut = (start > int(0.05 * TARGET_SR)) or (end < x.shape[0] - int(0.05 * TARGET_SR))
    steps.append(
        _step(
            "trim",
            "done" if cut else "skip",
            f"{start / TARGET_SR:.2f}s–{end / TARGET_SR:.2f}s kept",
        )
    )
    x = trimmed
    dur = float(x.shape[0] / TARGET_SR)
    snr_db = estimate_snr_db(x)
    noisy = snr_db < SNR_GATE_DB
    steps.append(_step("snr", "noisy" if noisy else "clean", f"{snr_db:.1f} dB"))
    if noisy:
        x, gain_note = maybe_gain(x)
    else:
        gain_note = "clean clip, level left unchanged"
    steps.append(_step("gain", "done" if "->" in gain_note or gain_note.startswith("clipped") else "skip", gain_note))

    if dur < SHORT_S:
        ranked = topk(model, processor, x)
        lang, score = ranked[0]
        steps.append(_step("path", "short", f"{dur:.2f}s, one pass"))
        steps.append(_step("denoise", "skip", "short clip is not denoised"))
        steps.append(_step("windows", "skip", "short clip is not sliced"))
        decision = lang if score >= MIN_SCORE else "uncertain"
        return _pack(decision, ranked, dur, snr_db, steps, [])

    ranked = topk(model, processor, x)
    lang, score = ranked[0]
    steps.append(_step("path", "whole", f"{dur:.2f}s, official whole-clip LID"))

    if noisy and score < MIN_SCORE:
        try:
            y = denoise_dns48(x)
        except RuntimeError as exc:
            steps.append(_step("denoise", "skip", str(exc)))
            y = None
        if y is not None:
            ranked2 = topk(model, processor, y)
            lang2, score2 = ranked2[0]
            steps.append(
                _step("denoise", "done", f"dns48 because noisy and score {score:.3f}; after {lang2} {score2:.3f}")
            )
            x = y
            ranked = ranked2
            lang, score = lang2, score2
    else:
        why = "clean" if not noisy else f"noisy but whole-clip score {score:.3f} is already high"
        steps.append(_step("denoise", "skip", why))

    second_lang, second_score = ranked[1]
    suspect = score < SWITCH_TOP1_BELOW and second_score >= SWITCH_TOP2_AT_LEAST and second_lang != lang
    windows = []
    decision = "uncertain" if score < MIN_SCORE else lang
    if suspect:
        windows = _windows(model, processor, x)
        kept = []
        for w in windows:
            if w["score"] >= MIN_SCORE and w["lang"] not in kept:
                kept.append(w["lang"])
        if len(kept) == 2:
            decision = "mixed:" + "+".join(kept)
            steps.append(_step("windows", "done", decision))
        else:
            steps.append(
                _step(
                    "windows",
                    "done",
                    "not exactly two confident languages; keep whole-clip",
                )
            )
    else:
        steps.append(_step("windows", "skip", "one confident language, no switch check"))

    return _pack(decision, ranked, dur, snr_db, steps, windows)


def _related_warning(ranked: list[tuple[str, float]]) -> bool:
    if len(ranked) < 2:
        return False
    a, sa = ranked[0]
    b, sb = ranked[1]
    return {a, b} <= RELATED and (sa - sb) < 0.30


def _pack(decision, ranked, dur, snr_db, steps, windows) -> dict:
    langs = []
    if decision.startswith("mixed:"):
        langs = decision.split(":", 1)[1].split("+")
    elif decision not in ("uncertain", "no-speech"):
        langs = [decision]
    return {
        "decision": decision,
        "langs": langs,
        "top5": [{"rank": i, "code": c, "score": s} for i, (c, s) in enumerate(ranked, start=1)],
        "duration_s": dur,
        "snr_db": snr_db,
        "steps": steps,
        "windows": windows,
        "related_warning": _related_warning(ranked),
    }
