# LID code — AIE1903 Group 3

Code for Task 4 (Spoken Language Identification).  
Suggested checkpoint: [`facebook/mms-lid-126`](https://huggingface.co/facebook/mms-lid-126).

This folder is **inference + a wrapper**. It labels a language (`cmn` / `eng` / `kor`, or `mixed:…` / `uncertain`). It does **not** transcribe speech.

Why the *task* matters belongs in the **slides** (evaluation item 1), not here. This README covers evaluation item 5: runnable code, what each file does, complexity, and who did what.

## Layout

```text
scripts/
  predict.py              command-line entry: any wav -> Top-5
  pipeline.py             five checks around the frozen model
  demo_server.py          local HTML page (upload / record / dataset clips)
  demo/index.html         the page served at http://127.0.0.1:8765/
  eval_pipeline.py        score those five checks on local challenge wavs
  run_lid.py              whole-clip LID on AISHELL + LibriSpeech
  eval_challenges.py      build challenge wavs; whole-clip baseline
  eval_improved.py        earlier sliding window + reject + fallback
  run_third_lang.py       same model on one Korean clip (plus cmn/eng)
  samples/                three real 16 kHz wavs (not committed)
  challenges/             crops / noise / silence / 1 s+1 s concat (not committed)
```

Paths inside the scripts are Windows absolute paths (`C:\aie1903\...`). Do not put API tokens in this folder.

## Environment

- Conda env `speech_processing`, Python 3.10, CPU PyTorch
- Hugging Face cache: `C:\aie1903\hf`
- Interpreter we used: `D:\Anaconda3\envs\speech_processing\python.exe`
- Packages: `torch`, `transformers`, `soundfile`, `numpy`, `librosa`, `pyarrow`, `huggingface_hub`
- Optional: `denoiser` (Facebook dns48). The webpage imports it only when a clip is both noisy and the whole-clip score is below 0.5.

Working directory:

```text
C:\aie1903\lid_project
```

## How to run

HTML demo (dataset clips + microphone / upload):

```text
python demo_server.py
```

Then open Chrome at `http://127.0.0.1:8765/` . Leave the terminal open. First load of the model is ~30 s on CPU.

The Record button shows a red timer. Stop writes a wav under `scripts/recordings/` (or `recordings/` next to `demo_server.py`). Pick that file later and click Identify. Live wavs are gitignored. CPU inference often takes 10–20 s; the yellow result box updates when it finishes.

Command-line entry (same official whole-clip LID):

```text
python predict.py samples\aishell_cmn_BAC009S0770W0259.wav
python predict.py samples\librispeech_eng_6930-75918-0000.wav
python predict.py samples\minds14_kor_602bef265f67b421554f65e7.wav
```

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
| `predict.py` | any wav path you pass | prints Top-5 (does not write a new wav) |
| `run_lid.py` | AISHELL wav under `C:\d` (id `BAC009S0770W0259`); LibriSpeech parquet `C:\aie1903\assignment1\data\huggingface_librispeech_test_clean\all\test.clean\0000.parquet` | `samples/aishell_*.wav`, `samples/librispeech_*.wav` |
| `eval_challenges.py` | those two samples | `challenges/*.wav`, `challenges_baseline.txt` |
| `eval_improved.py` | `challenges/*.wav` | `challenges_improved.txt` |
| `eval_pipeline.py` | `scripts/challenges/*.wav` (skipped if that folder is empty) | `docs/results/pipeline_eval.txt` |
| `run_third_lang.py` | the two samples + downloads one MINDS-14 `ko-KR` row | `samples/minds14_kor_*.wav`, `third_lang.txt` |

First use of the model downloads weights into `C:\aie1903\hf`. Keep clips short (scripts cap at 8 s). `eval_improved.py` must run after `eval_challenges.py`.

## Wrapper (our code, not a new model)

`mms-lid-126` mean-pools time, so one forward pass → one label. The webpage does not change the weights. `pipeline.py` only decides what to feed it:

1. Mono, 16 kHz (cap 8 s)
2. Almost no energy → `no-speech`
3. Trim silent edges only
4. One official whole-clip label
5. Extra work only when that label is weak: dns48 if noisy and score < 0.5; 1 s slices if the top two scores look like two languages. `mixed` requires exactly two confident languages.

`eval_improved.py` is the earlier challenge-set experiment (window, reject, whole-clip fallback). The webpage does not use that fallback. On this machine `eval_pipeline.py` scored OK 12/12; the live mic row is recorded as SEE and is not part of the 12. A noisy flag does not by itself turn on denoising. Concat test is **1 s Mandarin + 1 s English**, not arbitrary code-switching. Logs: `docs/results/`.

## Complexity

- Frozen model: Wav2Vec2 sequence classifier, ~1B parameters; 1280 → 1024 projector, then 126-way linear head.
- Our code: a few hundred lines of Python, no training loop.
- Runtime: one 4 s clip is tens of seconds on CPU; `eval_improved.py` is many 1 s forwards (a few minutes).
- We do not fine-tune the 1B encoder on CPU.

## Group contribution

Fill the empty rows before submit. Item 5 asks for this in the README.

| Member | Code / files |
|--------|----------------|
| Fan Rui | `predict.py`, `pipeline.py`, `demo_server.py`, `demo/index.html`, `eval_pipeline.py`, `run_lid.py`, `eval_challenges.py`, `eval_improved.py`, `run_third_lang.py`, this README |
| Han Sihao | |
| Qian Chen | |
| Zhu Zhiming | |
