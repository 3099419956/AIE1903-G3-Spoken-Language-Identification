"""Command-line inference for the frozen MMS language identification model."""

import json
from pathlib import Path

from runtime import configure_cache

configure_cache()

import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from cli_options import build_predict_parser
from pipeline import run as run_pipeline
from pipeline import to_mono_16k, topk


def predict_whole(model, processor, path: Path, model_id: str, top_k: int) -> dict:
    wav, source_sr = sf.read(str(path), dtype="float32")
    x, format_note = to_mono_16k(wav, int(source_sr))
    ranked = topk(model, processor, x, k=top_k)
    lang, score = ranked[0]
    return {
        "file": str(path),
        "model": model_id,
        "mode": "whole",
        "source_sample_rate": int(source_sr),
        "duration_s": x.shape[0] / 16000,
        "decision": lang,
        "top1_score": score,
        "top5": [{"rank": i, "code": code, "score": s} for i, (code, s) in enumerate(ranked, 1)],
        "steps": [{"name": "format", "action": "done", "detail": format_note}],
    }


def predict_pipeline(model, processor, path: Path, model_id: str, top_k: int) -> dict:
    wav, source_sr = sf.read(str(path), dtype="float32")
    out = run_pipeline(model, processor, wav, int(source_sr))
    out["file"] = str(path)
    out["model"] = model_id
    out["mode"] = "pipeline"
    out["source_sample_rate"] = int(source_sr)
    out["top5"] = out.get("top5", [])[:top_k]
    return out


def print_result(result: dict) -> None:
    print("File:", result["file"])
    print("Model:", result["model"])
    print("Mode:", result["mode"])
    print("Original sample rate:", result["source_sample_rate"])
    print("Duration used (seconds):", round(result["duration_s"], 3))
    print("Decision:", result["decision"])

    if result.get("snr_db") is not None:
        print("Estimated SNR (dB):", round(float(result["snr_db"]), 2))

    steps = result.get("steps") or []
    if steps:
        print("Pipeline:")
        for step in steps:
            print(f"  {step['name']:<10} {step['action']:<7} {step['detail']}")

    print(f"Top-{len(result.get('top5') or [])}:")
    for row in result.get("top5") or []:
        print(f"  {row['rank']}. {row['code']:<7} {row['score']:.6f}")


def main() -> None:
    args = build_predict_parser().parse_args()
    torch.set_num_threads(args.threads)

    existing = [path for path in args.audio if path.is_file()]
    missing = [path for path in args.audio if not path.is_file()]
    for path in missing:
        print("File not found:", path)

    results: list[dict] = []
    if existing:
        print(f"Loading {args.model} on CPU with {args.threads} thread(s)...", flush=True)
        processor = AutoFeatureExtractor.from_pretrained(args.model)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model)
        model.eval()

        predictor = predict_pipeline if args.mode == "pipeline" else predict_whole
        for path in existing:
            result = predictor(model, processor, path, args.model, args.top_k)
            results.append(result)
            if not args.as_json and args.output is None:
                print_result(result)
                print()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("Wrote", args.output)
    elif args.as_json:
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
