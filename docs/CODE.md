# LID code — AIE1903 Group 3

Code for Task 4 (Spoken Language Identification).  
Suggested checkpoint: [`facebook/mms-lid-126`](https://huggingface.co/facebook/mms-lid-126).

This folder is **inference + a wrapper**. It labels a language (`cmn` / `eng` / `kor`, or `mixed:…` / `uncertain`). It does **not** transcribe speech.

Why the *task* matters belongs in the **slides** (evaluation item 1), not here. This README covers evaluation item 5: runnable code, what each file does, complexity, and who did what.

## Layout

```text
lid_project/
  predict.py              command-line entry: any wav -> Top-5
  demo_server.py          local HTML page (upload / record / dataset clips)
  demo/index.html         the page served at http://127.0.0.1:8765/
  run_lid.py              whole-clip LID on AISHELL + LibriSpeech
  eval_challenges.py      build challenge wavs; whole-clip baseline
  eval_improved.py        sliding window + reject + whole-clip fallback
  run_third_lang.py       same model on one Korean clip (plus cmn/eng)
  samples/                three real 16 kHz wavs (not TTS)
  challenges/             crops / noise / silence / 1 s+1 s concat
  challenges_baseline.txt
  challenges_improved.txt
  third_lang.txt
```

Paths inside the scripts are Windows absolute paths (`C:\aie1903\...`). Do not put API tokens in this folder.

## Environment

- Conda env `speech_processing`, Python 3.10, CPU PyTorch
- Hugging Face cache: `C:\aie1903\hf`
- Interpreter we used: `D:\Anaconda3\envs\speech_processing\python.exe`
- Packages: `torch`, `transformers`, `soundfile`, `numpy`, `librosa`, `pyarrow`, `huggingface_hub`

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
python run_third_lang.py
```

| Script | Reads | Writes |
|--------|--------|--------|
| `predict.py` | any wav path you pass | prints Top-5 (does not write a new wav) |
| `run_lid.py` | AISHELL wav under `C:\d` (id `BAC009S0770W0259`); LibriSpeech parquet `C:\aie1903\assignment1\data\huggingface_librispeech_test_clean\all\test.clean\0000.parquet` | `samples/aishell_*.wav`, `samples/librispeech_*.wav` |
| `eval_challenges.py` | those two samples | `challenges/*.wav`, `challenges_baseline.txt` |
| `eval_improved.py` | `challenges/*.wav` | `challenges_improved.txt` |
| `run_third_lang.py` | the two samples + downloads one MINDS-14 `ko-KR` row | `samples/minds14_kor_*.wav`, `third_lang.txt` |

First use of the model downloads weights into `C:\aie1903\hf`. Keep clips short (scripts cap at 8 s). `eval_improved.py` must run after `eval_challenges.py`.

## Wrapper (our code, not a new model)

`mms-lid-126` mean-pools time, so one forward pass → one label. Extra logic is in `eval_improved.py`:

1. Energy onset (skip leading silence)
2. 1.0 s window, 0.5 s hop
3. Drop a window if top-1 score < 0.5 (`uncertain`)
4. If kept windows disagree → `mixed:lang1+lang2`
5. If every window is dropped → one whole-clip forward (helps noisy Mandarin)

Expected numbers on this machine are in the `*.txt` logs. Baseline single-language 8/9; wrapper HIT 10 / REJECT 1 / MISS 0. Concat test is **1 s Mandarin + 1 s English**, not arbitrary code-switching.

## Complexity

- Frozen model: Wav2Vec2 sequence classifier, ~1B parameters; 1280 → 1024 projector, then 126-way linear head.
- Our code: a few hundred lines of Python, no training loop.
- Runtime: one 4 s clip is tens of seconds on CPU; `eval_improved.py` is many 1 s forwards (a few minutes).
- We do not fine-tune the 1B encoder on CPU.

## Group contribution

Fill the empty rows before submit. Item 5 asks for this in the README.

| Member | Code / files |
|--------|----------------|
| Fan Rui | `predict.py`, `demo_server.py`, `demo/index.html`, `run_lid.py`, `eval_challenges.py`, `eval_improved.py`, `run_third_lang.py`, this README |
| Han Sihao | |
| Qian Chen | |
| Zhu Zhiming | |
