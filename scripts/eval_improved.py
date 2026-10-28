"""Sliding-window LID + low-confidence reject on the same challenge set."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from runtime import configure_cache

configure_cache()

from lid import judge, load_16k, load_model, sliding_predict

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Sliding-window LID + reject")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--win_s", type=float, default=1.0)
    ap.add_argument("--hop_s", type=float, default=0.5)
    ap.add_argument("--min_score", type=float, default=0.50)
    args = ap.parse_args()

    missing = [name for name in GOLD if not (CHALLENGE_DIR / (name + ".wav")).is_file()]
    if missing:
        print("Missing challenge wavs in", CHALLENGE_DIR)
        print("Run python eval_challenges.py first.")
        sys.exit(1)

    model, processor, device = load_model(args.model_dir, device="cpu")
    print("Model loaded. Device:", device)
    print(f"Window {args.win_s}s, hop {args.hop_s}s, reject if score < {args.min_score}")
    print()

    lines = [
        f"Improved LID: window={args.win_s}s hop={args.hop_s}s min_score={args.min_score}",
        "HIT=correct, REJECT=uncertain (no false language), PARTIAL=mixed but missing a language, MISS=wrong",
        "",
    ]
    header = f"{'clip':<24} {'gold':<7} {'baseline_note':<12} {'improved':<22} {'judge':<8} windows"
    print(header)
    print("-" * 110)
    lines.append(header)

    counts = Counter()
    for name, gold in GOLD.items():
        wav = load_16k(CHALLENGE_DIR / (name + ".wav"), max_seconds=None)[0]
        out = sliding_predict(
            model,
            processor,
            model.config.id2label,
            wav,
            win_s=args.win_s,
            hop_s=args.hop_s,
            min_score=args.min_score,
        )
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

    out = ROOT.parent / "docs" / "results" / "challenges_improved.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
