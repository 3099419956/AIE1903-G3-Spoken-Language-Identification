"""Score the webpage pipeline on the challenge wavs. Does not change the model."""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", r"C:\aie1903\hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"C:\aie1903\hf\hub")

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from pipeline import TARGET_SR, run

ROOT = Path(__file__).resolve().parent
CHAL = ROOT / "challenges"
LOG = ROOT.parent / "docs" / "results" / "pipeline_eval.txt"
MODEL_ID = "facebook/mms-lid-126"

CASES = [
    ("cmn_clean.wav", "cmn"),
    ("eng_clean.wav", "eng"),
    ("cmn_0.5s.wav", "uncertain"),
    ("cmn_1.0s.wav", "cmn"),
    ("eng_0.5s.wav", "eng"),
    ("eng_1.0s.wav", "eng"),
    ("cmn_lead_silence_2s.wav", "cmn"),
    ("cmn_snr5.wav", "cmn"),
    ("eng_snr5.wav", "eng"),
    ("switch_cmn_then_eng.wav", "mixed"),
    ("switch_eng_then_cmn.wav", "mixed"),
    ("live_20261021_231952.wav", "yue-or-cmn"),
]


def load(path: Path):
    x, sr = sf.read(str(path), dtype="float32")
    return np.asarray(x, dtype=np.float32), int(sr)


def judge(gold: str, decision: str) -> str:
    if gold == "uncertain":
        return "OK" if decision == "uncertain" else "MISS"
    if gold == "mixed":
        return "OK" if "cmn" in decision and "eng" in decision else "MISS"
    if gold == "yue-or-cmn":
        return "SEE"
    return "OK" if decision == gold else "MISS"


def main() -> None:
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    silence = np.zeros(TARGET_SR * 2, dtype=np.float32)
    extra = [("silence_2s", "no-speech", silence, TARGET_SR)]

    lines = ["pipeline vs expected behavior (model weights unchanged)", ""]
    ok = 0
    scored = 0
    for name, gold in CASES:
        path = CHAL / name
        if not path.is_file():
            path = ROOT / "recordings" / name
        if not path.is_file():
            lines.append(f"MISSING {name}")
            continue
        wav, sr = load(path)
        out = run(model, processor, wav, sr)
        mark = judge(gold, out["decision"])
        if mark != "SEE":
            scored += 1
            ok += int(mark == "OK")
        top = out["top5"][0] if out["top5"] else ("", 0.0)
        actions = " | ".join(f"{s['name']}:{s['action']}" for s in out["steps"])
        row = (
            f"{name:<28} gold={gold:<12} got={out['decision']:<16} {mark:<4} "
            f"snr={out['snr_db']} top={top} :: {actions}"
        )
        print(row, flush=True)
        lines.append(row)

    wav, sr = extra[0][2], extra[0][3]
    out = run(model, processor, wav, sr)
    mark = "OK" if out["decision"] == "no-speech" else "MISS"
    scored += 1
    ok += int(mark == "OK")
    row = f"{'silence_2s':<28} gold={'no-speech':<12} got={out['decision']:<16} {mark}"
    print(row, flush=True)
    lines.append(row)
    lines.append("")
    lines.append(f"scored OK {ok}/{scored} (live recording is SEE, not scored)")
    print(lines[-1], flush=True)
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
