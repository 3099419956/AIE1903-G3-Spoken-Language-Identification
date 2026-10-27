"""Local HTML demo: dataset clips + your own recording -> MMS-LID-126 Top-5."""
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from runtime import configure_cache

configure_cache()

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from http_form import MAX_UPLOAD_BYTES, parse_multipart_form, safe_wav_name
from pipeline import run as run_pipeline

MODEL_ID = "facebook/mms-lid-126"
TARGET_SR = 16000
MAX_SECONDS = 8.0
TOP_K = 5
HOST = "127.0.0.1"
PORT = 8765
ROOT = Path(__file__).resolve().parent
HTML = ROOT / "demo" / "index.html"
SAMPLE_DIR = ROOT / "samples"
RECORD_DIR = ROOT / "recordings"

SAMPLES = {
    "cmn": SAMPLE_DIR / "aishell_cmn_BAC009S0770W0259.wav",
    "eng": SAMPLE_DIR / "librispeech_eng_6930-75918-0000.wav",
    "kor": SAMPLE_DIR / "minds14_kor_602bef265f67b421554f65e7.wav",
}

CASE_DIR = ROOT / "challenges"
CASES = {
    "clean": CASE_DIR / "cmn_clean.wav",
    "noisy": CASE_DIR / "cmn_snr5.wav",
    "short": CASE_DIR / "cmn_0.5s.wav",
    "silence": CASE_DIR / "cmn_lead_silence_2s.wav",
    "switch": CASE_DIR / "switch_cmn_then_eng.wav",
}

# Short English names for codes we actually show in the demo.
NAMES = {
    "cmn": "Mandarin Chinese",
    "eng": "English",
    "kor": "Korean",
    "jpn": "Japanese",
    "yue": "Yue Chinese",
    "bod": "Tibetan",
    "mya": "Burmese",
    "vie": "Vietnamese",
    "tha": "Thai",
    "lao": "Lao",
    "khm": "Khmer",
    "ind": "Indonesian",
    "zlm": "Malay",
    "jav": "Javanese",
    "fra": "French",
    "spa": "Spanish",
    "deu": "German",
    "rus": "Russian",
    "ara": "Arabic",
    "hin": "Hindi",
    "fas": "Persian",
    "cym": "Welsh",
    "hye": "Armenian",
}

PROCESSOR = None
MODEL = None
INFER_LOCK = threading.Lock()


def to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x[:, 0]
    if int(sr) != TARGET_SR:
        import librosa

        x = np.asarray(librosa.resample(y=x, orig_sr=int(sr), target_sr=TARGET_SR), dtype=np.float32)
    n = int(MAX_SECONDS * TARGET_SR)
    if x.shape[0] > n:
        x = x[:n]
    return x


def predict_array(wav: np.ndarray, sr: int, source: str) -> dict:
    with INFER_LOCK:
        out = run_pipeline(MODEL, PROCESSOR, wav, sr)
    top5 = []
    for row in out["top5"]:
        code = row["code"]
        top5.append(
            {
                "rank": row["rank"],
                "code": code,
                "name": NAMES.get(code, code),
                "score": row["score"],
            }
        )
    return {
        "source": source,
        "duration_s": out["duration_s"],
        "snr_db": out["snr_db"],
        "decision": out["decision"],
        "steps": out["steps"],
        "related_warning": out["related_warning"],
        "windows": out["windows"],
        "top5": top5,
    }


def read_wav_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    return np.asarray(wav, dtype=np.float32), int(sr)


def list_recordings() -> list[str]:
    if not RECORD_DIR.is_dir():
        return []
    names = [p.name for p in RECORD_DIR.glob("*.wav") if p.is_file()]
    names.sort(reverse=True)
    return names[:40]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, raw, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            return
        if path.startswith("/sample-audio/"):
            key = path.rsplit("/", 1)[-1]
            wav_path = SAMPLES.get(key)
            if wav_path is None or not wav_path.is_file():
                self._json(404, {"error": "unknown sample"})
                return
            self._send(200, wav_path.read_bytes(), "audio/wav")
            return
        if path == "/api/recordings":
            self._json(200, {"files": list_recordings()})
            return
        if path.startswith("/recording-audio/"):
            name = safe_wav_name(path.rsplit("/", 1)[-1])
            wav_path = RECORD_DIR / name
            if wav_path.parent != RECORD_DIR or not wav_path.is_file():
                self._json(404, {"error": "unknown recording"})
                return
            self._send(200, wav_path.read_bytes(), "audio/wav")
            return
        if path.startswith("/case-audio/"):
            key = path.rsplit("/", 1)[-1]
            wav_path = CASES.get(key)
            if wav_path is None or not wav_path.is_file():
                self._json(404, {"error": "unknown case"})
                return
            self._send(200, wav_path.read_bytes(), "audio/wav")
            return
        if path.startswith("/api/case/"):
            key = path.rsplit("/", 1)[-1]
            wav_path = CASES.get(key)
            if wav_path is None or not wav_path.is_file():
                self._json(404, {"error": "unknown case"})
                return
            wav, sr = sf.read(str(wav_path), dtype="float32")
            try:
                out = predict_array(np.asarray(wav, dtype=np.float32), int(sr), "case: " + key)
            except Exception as exc:
                self._json(500, {"error": str(exc)})
                return
            self._json(200, out)
            return
        if path.startswith("/api/sample/"):
            key = path.rsplit("/", 1)[-1]
            wav_path = SAMPLES.get(key)
            if wav_path is None or not wav_path.is_file():
                self._json(404, {"error": "unknown sample"})
                return
            wav, sr = sf.read(str(wav_path), dtype="float32")
            try:
                out = predict_array(wav, sr, "dataset: " + wav_path.name)
            except Exception as exc:
                self._json(500, {"error": str(exc)})
                return
            self._json(200, out)
            return
        self._json(404, {"error": "not found"})

    def _read_audio_form(self):
        content_type = self.headers.get("Content-Type", "")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length header") from exc
        if length <= 0:
            raise ValueError("missing request body")
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("audio upload is too large")
        body = self.rfile.read(length)
        return parse_multipart_form(content_type, body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/save":
            try:
                raw, filename, _form = self._read_audio_form()
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            if not raw:
                self._json(400, {"error": "missing audio"})
                return
            RECORD_DIR.mkdir(parents=True, exist_ok=True)
            name = safe_wav_name(filename)
            dest = RECORD_DIR / name
            dest.write_bytes(raw)
            print("Saved recording:", dest, flush=True)
            self._json(200, {"filename": name})
            return
        if path != "/api/predict":
            self._json(404, {"error": "not found"})
            return
        try:
            raw, filename, form = self._read_audio_form()
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        if not raw:
            self._json(400, {"error": "missing audio"})
            return
        source = "your recording"
        if "source" in form:
            source = str(form.get("source") or source)
        try:
            wav, sr = read_wav_bytes(raw)
            out = predict_array(wav, sr, source)
        except Exception as exc:
            self._json(400, {"error": "could not read audio: " + str(exc)})
            return
        self._json(200, out)


def main() -> None:
    global PROCESSOR, MODEL
    n_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(n_threads)
    print("Loading", MODEL_ID, "once (CPU, ~30s, threads=%d)..." % n_threads, flush=True)
    PROCESSOR = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    MODEL = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
    MODEL.eval()
    print("Model ready.")
    missing = [k for k, p in SAMPLES.items() if not p.is_file()]
    if missing:
        print("Missing sample wavs:", missing)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("Open in Chrome: http://127.0.0.1:%d/" % PORT, flush=True)
    print("Leave this window open. Ctrl+C to stop.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
