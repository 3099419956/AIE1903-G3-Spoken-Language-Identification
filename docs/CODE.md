# LID code — AIE1903 Group 3

Code for Task 4 (Spoken Language Identification).  
Suggested checkpoint: [`facebook/mms-lid-126`](https://huggingface.co/facebook/mms-lid-126).

This folder is **inference + a wrapper**. It labels a language (`cmn` / `eng` / `kor`, or `mixed:…` / `uncertain`). It does **not** transcribe speech.

Why the *task* matters belongs in the **slides** (evaluation item 1), not here. This README covers evaluation item 5: runnable code, what each file does, complexity, and who did what.

## Layout

```text
scripts/
  predict.py              command line: whole-clip or full pipeline, text/JSON
  cli_options.py          shared command-line arguments and validation
  pipeline.py             five checks around the frozen model
  demo_server.py          local HTML page plus configurable HTTP server
  demo/index.html         the page served at http://127.0.0.1:8765/
  eval_pipeline.py        score those five checks on local challenge wavs
  run_lid.py              whole-clip LID on AISHELL + LibriSpeech
  eval_challenges.py      build challenge wavs; whole-clip baseline
  eval_improved.py        earlier sliding window + reject + fallback
  run_third_lang.py       same model on one Korean clip (plus cmn/eng)
  samples/                three real 16 kHz wavs (not committed)
  challenges/             crops / noise / silence / 1 s+1 s concat (not committed)
```

Every script finds samples, challenges, and logs from its own folder.
`scripts/samples/` and `scripts/challenges/` hold the wavs. Result text goes to `docs/results/`.
`run_lid.py` does not contain a personal dataset path. Set `AISHELL_CACHE` and `LIBRI_PARQUET` only when rebuilding the two original clips.
Do not put API tokens in this folder.

## Environment

- Conda env `speech_processing`, Python 3.10, CPU PyTorch
- Standard Hugging Face / PyTorch caches by default
- Optional shared cache root: set `AIE1903_CACHE_DIR`
- Tested versions: `torch 2.14.0+cpu`, `transformers 5.20.0`
- Packages used by the scripts: `torch`, `transformers`, `soundfile`, `numpy`, `librosa`, `pyarrow`, `huggingface_hub`
- Optional: `denoiser` (Facebook dns48). `pipeline.py` imports it only when a clip is noisy and the whole-clip score is below 0.5. It is not in `requirements.txt`.

Run the scripts from `scripts/`.

## How to run

HTML demo (dataset clips + microphone / upload):

```text
python demo_server.py
python demo_server.py --port 9000 --threads 2 --open-browser
```

Then open Chrome at `http://127.0.0.1:8765/` . Leave the terminal open. First load of the model is ~30 s on CPU.
`GET /api/health` reports model readiness, CPU threads, and available demo clips.

The Record button shows a red timer. In this repository, Stop writes a wav under `scripts/recordings/`. Pick that file later and click Identify. Live wavs are gitignored. CPU inference often takes 10–20 s; the yellow result box updates when it finishes.

Command-line entry:

```text
python predict.py samples\aishell_cmn_BAC009S0770W0259.wav
python predict.py --mode pipeline --json samples\librispeech_eng_6930-75918-0000.wav
python predict.py --mode pipeline --top-k 3 --threads 2 samples\a.wav samples\b.wav
python predict.py --mode pipeline --output outputs\predictions.json samples\a.wav
```

The default `whole` mode reports the official full-clip label. `pipeline` mode
also applies silence checks, confidence rejection, optional denoising, and mixed
speech detection. `--output` always writes JSON.

Reproducing the tables:

```text
python run_lid.py
python eval_challenges.py
python eval_improved.py
python eval_pipeline.py
python run_third_lang.py
```

| Script | Reads | Writes |
|--------|--------|--------|
| `predict.py` | one or more audio paths; `--mode whole` or `--mode pipeline` | text, JSON, or a JSON output file |
| `cli_options.py` | no data | validated arguments shared by the CLI and demo server |
| `run_lid.py` | `AISHELL_CACHE` and `LIBRI_PARQUET` environment variables, only to rebuild the two clips | `scripts/samples/aishell_*.wav` and `librispeech_*.wav` |
| `eval_challenges.py` | those two samples in `scripts/samples/` | `scripts/challenges/*.wav` and `docs/results/challenges_baseline.txt` |
| `eval_improved.py` | `scripts/challenges/*.wav` | `docs/results/challenges_improved.txt` |
| `eval_pipeline.py` | `scripts/challenges/*.wav`. A missing file is skipped | `docs/results/pipeline_eval.txt` |
| `run_third_lang.py` | the two samples in `scripts/samples/`, plus one downloaded MINDS-14 `ko-KR` row | `scripts/samples/minds14_kor_*.wav` and `docs/results/third_lang.txt` |

If `AIE1903_CACHE_DIR` is unset, the scripts use the standard user caches. To reuse a shared cache, set it before running:

```powershell
$env:AIE1903_CACHE_DIR = "D:\aie1903-cache"
```

Keep clips short (scripts cap at 8 s). `eval_improved.py` must run after `eval_challenges.py`. The `file=` column in the committed logs shows the folder used on the day of that run. Current scripts write under `scripts/` and `docs/results/`.

Run the standard-library checks before opening the demo:

```text
python -m unittest discover -s tests -v
python -m compileall -q scripts
```

## Wrapper (our code, not a new model)

`mms-lid-126` mean-pools time, so one forward pass → one label. The webpage does not change the weights. `pipeline.py` only decides what to feed it:

1. Mono, 16 kHz (cap 8 s)
2. Almost no energy → `no-speech`
3. Trim silent edges only
4. One official whole-clip label
5. Extra work only when that label is weak: dns48 if noisy and score < 0.5; 1 s slices if the top two scores look like two languages. `mixed` requires exactly two confident languages.

`eval_improved.py` is the earlier challenge-set experiment (window, reject, whole-clip fallback). The webpage does not use that fallback. On this machine `eval_pipeline.py` scored OK 12/12; the live mic row is recorded as SEE and is not part of the 12. A noisy flag does not by itself turn on denoising. Concat test is **1 s Mandarin + 1 s English**, not arbitrary code-switching. Logs: `docs/results/`.

## Complexity

- Frozen model: the course-suggested Wav2Vec2 language classification checkpoint with 126 labels.
- The paper fine-tunes an MMS 1B pretrained model. We only run inference with the released checkpoint and do not train its weights.
- Our code: inference, HTTP demo, and evaluation modules under `scripts/`; no training loop.
- `demo_server.py` and `predict.py` accept a validated `--threads` value. The remaining evaluation scripts use 1 thread.
- Runtime: one 4 s clip is often 10–20 s on CPU. `eval_improved.py` runs many 1 s forwards.
- We do not fine-tune the encoder or load the paper's 1B ASR model.

## Group contribution

Evaluation item 5 asks for this. Fan Rui owns the model pipeline and evaluation.
Zhu Zhiming owns the command-line and web-demo implementation. The slide file is
not in this repository.

| Member | Code in this repository | Presentation |
|--------|-------------------------|--------------|
| Fan Rui | `pipeline.py`, model inference, data preparation, and main evaluation scripts | paper difficulties, the preprocessing checks, and the plan to train only a small head |
| Han Sihao | no source file | paper and model: MMS, mean-pool, why 126 |
| Qian Chen | no source file | task: input, output, and why language identification comes first |
| Zhu Zhiming | `predict.py`, `cli_options.py`, demo-server options and health endpoint, frontend status, and CLI tests | implementation, command line, and the web demo |
