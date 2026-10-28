"""Entry script: one wav file in, MMS-LID-126 Top-K out.

By default this is the official whole-clip inference. Pass ``--pipeline`` to run
the same five checks the web demo uses (silence, trim, denoise, switch detection).

The model is loaded from a local directory, never from the network::

    python predict.py path\\to\\audio.wav
    python predict.py path\\to\\audio.wav --model_dir D:\\models\\mms-lid-126
    python predict.py path\\to\\audio.wav --pipeline

Set ``MMS_LID_MODEL_DIR`` once to avoid passing ``--model_dir`` every time.
"""

import argparse
import sys
from pathlib import Path

from runtime import configure_cache

configure_cache()

from lid import DEFAULT_TOP_K, MAX_SECONDS, TARGET_SR, load_16k, load_model, predict, read_wav, resolve_model_dir


def _print_topk(ranked, top_k: int) -> None:
    print("Top-1:", ranked[0][0], "score:", f"{ranked[0][1]:.6f}")
    print("Top-%d:" % top_k)
    for n, (lang, score) in enumerate(ranked, start=1):
        print(f"  {n}. {lang}  {score:.6f}")


def _print_pipeline(out: dict) -> None:
    print("Decision:", out["decision"])
    print("Duration used (seconds):", round(out["duration_s"], 3))
    print("SNR (dB):", out["snr_db"])
    print("Steps:")
    for s in out["steps"]:
        print(f"  {s['name']:<8} {s['action']:<6} {s['detail']}")
    if out["top5"]:
        print("Top-%d:" % len(out["top5"]))
        for row in out["top5"]:
            print(f"  {row['rank']}. {row['code']}  {row['score']:.6f}")
    if out["windows"]:
        print("Windows:")
        for w in out["windows"]:
            print(f"  {w['start_s']:.1f}-{w['end_s']:.1f}s  {w['lang']}  {w['score']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MMS LID-126 inference")
    ap.add_argument("input", help="path to a wav/flac/mp3 file")
    ap.add_argument("--model_dir", default=None, help="local model directory")
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--max_seconds", type=float, default=MAX_SECONDS)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--pipeline", action="store_true", help="run the five-check pipeline (same as the web demo)")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print("File not found:", path)
        sys.exit(1)

    print("Model dir:", resolve_model_dir(args.model_dir))
    print("File:", path)
    print("Loading model...")
    model, processor, device = load_model(args.model_dir, args.device)
    print("Device:", device)

    if args.pipeline:
        from pipeline import run as run_pipeline

        raw, sr = read_wav(path)
        out = run_pipeline(model, processor, raw, sr)
        _print_pipeline(out)
        return

    wav, orig_sr = load_16k(path, max_seconds=args.max_seconds)
    dur = wav.shape[0] / float(TARGET_SR)
    print("Original sample rate:", orig_sr)
    print("Duration used (seconds):", round(dur, 3), "(cap", args.max_seconds, "s)")
    ranked = predict(model, processor, wav, top_k=args.top_k)
    _print_topk(ranked, args.top_k)


if __name__ == "__main__":
    main()
