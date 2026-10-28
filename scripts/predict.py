"""Entry script: one wav file in, MMS-LID-126 Top-K out. Official whole-clip inference.

The model is loaded from a local directory, never from the network::

    python predict.py path\\to\\audio.wav
    python predict.py path\\to\\audio.wav --model_dir D:\\models\\mms-lid-126

Set ``MMS_LID_MODEL_DIR`` once to avoid passing ``--model_dir`` every time.
"""

import argparse
import sys
from pathlib import Path

from runtime import configure_cache

configure_cache()

from lid import DEFAULT_TOP_K, MAX_SECONDS, TARGET_SR, load_16k, load_model, predict, resolve_model_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="MMS LID-126 inference")
    ap.add_argument("input", help="path to a wav/flac/mp3 file")
    ap.add_argument("--model_dir", default=None, help="local model directory")
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--max_seconds", type=float, default=MAX_SECONDS)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print("File not found:", path)
        sys.exit(1)

    wav, orig_sr = load_16k(path, max_seconds=args.max_seconds)
    dur = wav.shape[0] / float(TARGET_SR)

    print("Model dir:", resolve_model_dir(args.model_dir))
    print("File:", path)
    print("Original sample rate:", orig_sr)
    print("Duration used (seconds):", round(dur, 3), "(cap", args.max_seconds, "s)")
    print("Loading model...")
    model, processor, device = load_model(args.model_dir, args.device)
    print("Device:", device)

    ranked = predict(model, processor, wav, top_k=args.top_k)
    print("Top-1:", ranked[0][0], "score:", f"{ranked[0][1]:.6f}")
    print("Top-%d:" % args.top_k)
    for n, (lang, score) in enumerate(ranked, start=1):
        print(f"  {n}. {lang}  {score:.6f}")


if __name__ == "__main__":
    main()
