# Scripts

Runnable entry points for Group 3 LID. Run these from this `scripts/` folder
in the conda env `speech_processing`. Hugging Face cache: `C:\aie1903\hf`.

Fan Rui's day-to-day working copy is still `C:\aie1903\lid_project`.
This folder is the same code, laid out for the group repo.

```text
python demo_server.py
python predict.py samples\aishell_cmn_BAC009S0770W0259.wav
python run_lid.py
python eval_challenges.py
python eval_improved.py
python eval_pipeline.py
python run_third_lang.py
```

| File | Role |
|------|------|
| `predict.py` | CLI: one wav → Top-5 language scores |
| `pipeline.py` | five checks around the frozen model; the webpage calls this |
| `demo_server.py` | local page at http://127.0.0.1:8765/ (record timer, save wav, Identify) |
| `demo/index.html` | dataset clips, five case buttons, step list, microphone |
| `recordings/` | live wavs written on Stop (gitignored) |
| `run_lid.py` | whole-clip LID on AISHELL + LibriSpeech |
| `eval_challenges.py` | short / noise / concat baseline |
| `eval_improved.py` | earlier sliding-window experiment |
| `eval_pipeline.py` | score the five webpage checks; looks for wavs in `challenges/` next to this script |
| `run_third_lang.py` | same model on one Korean clip |

The webpage checks, in order: mono 16 kHz; stop if almost silent; trim silent edges only; one official whole-clip label. Denoise (dns48) runs only when the clip is noisy and the top score is below 0.5. One-second slices run only when the top two scores look like two languages. `pip install denoiser` is needed only for that rare denoise branch. Model weights stay frozen.

Do not commit `samples/*.wav` or `challenges/*.wav` (gitignored).
Local copies live here and under `data/raw/`. Metric logs are in `docs/results/`.
How-to-run notes: `docs/CODE.md`.
