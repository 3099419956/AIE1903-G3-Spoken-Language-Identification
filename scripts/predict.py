"""Entry script: one wav file in, MMS-LID-126 Top-5 out. Official whole-clip inference."""
import sys
from pathlib import Path

from runtime import configure_cache

configure_cache()

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

MODEL_ID = "facebook/mms-lid-126"
TARGET_SR = 16000
MAX_SECONDS = 8.0
TOP_K = 5


def load_16k(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    x = np.asarray(wav, dtype=np.float32)
    orig_sr = int(sr)
    if orig_sr != TARGET_SR:
        import librosa

        x = np.asarray(librosa.resample(y=x, orig_sr=orig_sr, target_sr=TARGET_SR), dtype=np.float32)
    n = int(MAX_SECONDS * TARGET_SR)
    if x.shape[0] > n:
        x = x[:n]
    return x, orig_sr


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python predict.py path\\to\\audio.wav")
        print("Example: python predict.py samples\\aishell_cmn_BAC009S0770W0259.wav")
        print("Output: ISO 639-3 language codes (cmn/eng/kor/...), not a transcript.")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print("File not found:", path)
        sys.exit(1)

    torch.set_num_threads(1)
    wav, orig_sr = load_16k(path)
    dur = wav.shape[0] / float(TARGET_SR)

    print("Model:", MODEL_ID)
    print("File:", path)
    print("Original sample rate:", orig_sr)
    print("Duration used (seconds):", round(dur, 3), "(cap", MAX_SECONDS, "s)")
    print("Loading model...")
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits.float(), dim=-1)[0]
    values, indices = torch.topk(probs, k=TOP_K)
    ranked = [(str(model.config.id2label[int(i)]), float(s)) for s, i in zip(values.tolist(), indices.tolist())]

    print("Top-1:", ranked[0][0], "score:", f"{ranked[0][1]:.6f}")
    print("Top-5:")
    for n, (lang, score) in enumerate(ranked, start=1):
        print(f"  {n}. {lang}  {score:.6f}")


if __name__ == "__main__":
    main()
