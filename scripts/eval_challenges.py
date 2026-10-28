"""Build a challenge set and run baseline MMS-LID-126 (no sliding window / reject)."""

import argparse
import sys
from pathlib import Path

from runtime import configure_cache

configure_cache()

import numpy as np
import soundfile as sf

from lid import TARGET_SR, add_noise, crop_speech, load_16k, load_model, predict, resolve_model_dir

TOP_K = 3
RNG = np.random.default_rng(0)

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "samples"
CHALLENGE_DIR = ROOT / "challenges"
CMN_WAV = SAMPLE_DIR / "aishell_cmn_BAC009S0770W0259.wav"
ENG_WAV = SAMPLE_DIR / "librispeech_eng_6930-75918-0000.wav"


def save(name: str, x: np.ndarray) -> Path:
    CHALLENGE_DIR.mkdir(parents=True, exist_ok=True)
    path = CHALLENGE_DIR / (name + ".wav")
    sf.write(str(path), x, TARGET_SR)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline LID on the challenge set")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    args = ap.parse_args()

    missing = [p.name for p in (CMN_WAV, ENG_WAV) if not p.is_file()]
    if missing:
        print("Missing sample wavs in", SAMPLE_DIR)
        print("Need:", ", ".join(missing))
        print("Rebuild them with run_lid.py after setting AISHELL_CACHE and LIBRI_PARQUET.")
        sys.exit(1)

    cmn = load_16k(CMN_WAV)[0]
    eng = load_16k(ENG_WAV)[0]
    cmn_1s = crop_speech(cmn, 1.0)
    eng_1s = crop_speech(eng, 1.0)

    clips = [
        ("cmn_clean", "cmn", cmn),
        ("eng_clean", "eng", eng),
        ("cmn_0.5s", "cmn", crop_speech(cmn, 0.5)),
        ("cmn_1.0s", "cmn", cmn_1s),
        ("eng_0.5s", "eng", crop_speech(eng, 0.5)),
        ("eng_1.0s", "eng", eng_1s),
        ("cmn_lead_silence_2s", "cmn", np.concatenate([np.zeros(2 * TARGET_SR, dtype=np.float32), cmn])),
        ("cmn_snr5", "cmn", add_noise(cmn, 5.0, rng=RNG)),
        ("eng_snr5", "eng", add_noise(eng, 5.0, rng=RNG)),
        ("switch_cmn_then_eng", "mixed", np.concatenate([cmn_1s, eng_1s])),
        ("switch_eng_then_cmn", "mixed", np.concatenate([eng_1s, cmn_1s])),
    ]

    print("Model dir:", resolve_model_dir(args.model_dir))
    model, processor, device = load_model(args.model_dir, device="cpu")
    print("Model loaded. Device:", device)
    print()
    print(f"{'clip':<24} {'gold':<7} {'dur':>6} {'top1':<6} {'score':>8}  correct  top3")
    print("-" * 88)

    rows = []
    n_ok = 0
    n_single = 0
    for name, gold, wav in clips:
        path = save(name, wav)
        dur = wav.shape[0] / float(TARGET_SR)
        ranked = predict(model, processor, wav, top_k=args.top_k)
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
