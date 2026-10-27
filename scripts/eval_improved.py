"""Sliding-window LID + low-confidence reject on the same challenge set."""
import os
import sys
from collections import Counter
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
WIN_S = 1.0
HOP_S = 0.5
MIN_SCORE = 0.50

ROOT = Path(__file__).resolve().parent
CHALLENGE_DIR = ROOT / "challenges"
GOLD = {
    "cmn_clean": "cmn",
    "eng_clean": "eng",
    "cmn_0.5s": "cmn",
    "cmn_1.0s": "cmn",
    "eng_0.5s": "eng",
    "eng_1.0s": "eng",
    "cmn_lead_silence_2s": "cmn",
    "cmn_snr5": "cmn",
    "eng_snr5": "eng",
    "switch_cmn_then_eng": "mixed",
    "switch_eng_then_cmn": "mixed",
}


def load_16k(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    if int(sr) != TARGET_SR:
        raise ValueError("Expected 16 kHz: " + str(path))
    return np.asarray(x, dtype=np.float32)


def speech_start(x: np.ndarray) -> int:
    """First frame whose RMS is at least 10% of max RMS."""
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


def window_slices(n: int, win: int, hop: int, start: int = 0) -> list[tuple[int, int]]:
    if n - start <= win:
        return [(start, n)]
    starts = list(range(start, n - win + 1, hop))
    last = n - win
    if last - starts[-1] >= hop // 2:
        starts.append(last)
    return [(s, s + win) for s in starts]


def predict_logits(model, processor, wav: np.ndarray) -> torch.Tensor:
    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        return model(**inputs).logits


def decode_top1(logits: torch.Tensor, id2label: dict) -> tuple[str, float]:
    probs = torch.softmax(logits.float(), dim=-1)[0]
    idx = int(torch.argmax(probs).item())
    return str(id2label[idx]), float(probs[idx].item())


def pack(decision: str, langs: list[str], windows: list[dict], fallback: bool) -> dict:
    return {"decision": decision, "langs": langs, "windows": windows, "fallback": fallback}


def sliding_predict(model, processor, id2label: dict, wav: np.ndarray) -> dict:
    win = int(WIN_S * TARGET_SR)
    hop = int(HOP_S * TARGET_SR)
    n = int(wav.shape[0])
    if n <= win:
        lang, score = decode_top1(predict_logits(model, processor, wav), id2label)
        keep = score >= MIN_SCORE
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
            return pack(lang, [lang], windows, False)
        return pack("uncertain", [], windows, False)
    onset = speech_start(wav)
    spans = window_slices(n, win, hop, start=onset)
    windows = []
    for start, end in spans:
        lang, score = decode_top1(predict_logits(model, processor, wav[start:end]), id2label)
        windows.append(
            {
                "start_s": round(start / TARGET_SR, 3),
                "end_s": round(end / TARGET_SR, 3),
                "lang": lang,
                "score": score,
                "keep": score >= MIN_SCORE,
            }
        )
    kept = [w for w in windows if w["keep"]]
    fallback = False
    if not kept:
        # 1 s noisy windows can all fall below the gate; whole-clip pooling is more stable.
        lang, score = decode_top1(predict_logits(model, processor, wav), id2label)
        fallback = True
        windows.append(
            {
                "start_s": 0.0,
                "end_s": round(wav.shape[0] / TARGET_SR, 3),
                "lang": lang,
                "score": score,
                "keep": score >= MIN_SCORE,
            }
        )
        if score >= MIN_SCORE:
            decision = lang
            langs = [lang]
        else:
            decision = "uncertain"
            langs = []
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


def main() -> None:
    missing = [name for name in GOLD if not (CHALLENGE_DIR / (name + ".wav")).is_file()]
    if missing:
        print("Missing challenge wavs in", CHALLENGE_DIR)
        print("Run python eval_challenges.py first.")
        sys.exit(1)
    torch.set_num_threads(1)
    print("Loading model:", MODEL_ID)
    print(f"Window {WIN_S}s, hop {HOP_S}s, reject if score < {MIN_SCORE}")
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    print("Model loaded. Device: CPU")
    print()

    lines = [
        f"Improved LID: window={WIN_S}s hop={HOP_S}s min_score={MIN_SCORE}",
        "HIT=correct, REJECT=uncertain (no false language), PARTIAL=mixed but missing a language, MISS=wrong",
        "",
    ]
    header = f"{'clip':<24} {'gold':<7} {'baseline_note':<12} {'improved':<22} {'judge':<8} windows"
    print(header)
    print("-" * 110)
    lines.append(header)

    counts = Counter()
    for name, gold in GOLD.items():
        wav = load_16k(CHALLENGE_DIR / (name + ".wav"))
        out = sliding_predict(model, processor, model.config.id2label, wav)
        judge_tag = judge(gold, out["decision"])
        counts[judge_tag] += 1
        win_txt = " | ".join(
            f"{w['start_s']:.1f}-{w['end_s']:.1f}:{w['lang']}@{w['score']:.2f}"
            + ("*" if w["keep"] else "")
            for w in out["windows"]
        )
        note = "fallback" if out["fallback"] else "window"
        row = f"{name:<24} {gold:<7} {note:<12} {out['decision']:<22} {judge_tag:<8} {win_txt}"
        print(row)
        lines.append(row)

    print()
    summary = (
        f"Counts: HIT={counts['HIT']} REJECT={counts['REJECT']} "
        f"PARTIAL={counts['PARTIAL']} MISS={counts['MISS']}"
    )
    print(summary)
    print("Star * means the window passed the confidence gate.")
    lines.extend(["", summary])
    out_path = ROOT.parent / "docs" / "results" / "challenges_improved.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
