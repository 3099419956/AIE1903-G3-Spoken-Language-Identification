# Scripts

Runnable entry points for Group 3 LID. Run these from this `scripts/` folder
in the conda env `speech_processing`. The scripts use standard user caches by
default; set `AIE1903_CACHE_DIR` to reuse a shared cache.

Paths are next to these scripts. Wavs go in `samples/` and `challenges/`.
Result text goes in `docs/results/`.

```text
python demo_server.py
python demo_server.py --port 9000 --threads 2 --open-browser
python predict.py samples\aishell_cmn_BAC009S0770W0259.wav
python predict.py --mode pipeline --json samples\librispeech_eng_6930-75918-0000.wav
python run_lid.py
python eval_challenges.py
python eval_improved.py
python eval_pipeline.py
python run_third_lang.py
```

| File | Role |
|------|------|
| `predict.py` | CLI: whole-clip or full pipeline, multiple files, text or JSON |
| `cli_options.py` | shared validated arguments for the CLI and demo server |
| `pipeline.py` | five checks around the frozen model; `demo_server.py` and `eval_pipeline.py` call this |
| `demo_server.py` | local page and `/api/health`; supports host, port, threads, and browser launch |
| `demo/index.html` | dataset clips, five case buttons, step list, microphone |
| `recordings/` | wavs written when Stop is clicked (gitignored). This repository still saves on Stop |
| `run_lid.py` | whole-clip LID on AISHELL + LibriSpeech |
| `eval_challenges.py` | short / noise / concat baseline |
| `eval_improved.py` | earlier sliding-window experiment |
| `eval_pipeline.py` | score the five webpage checks; looks for wavs in `challenges/` next to this script |
| `run_third_lang.py` | same model on one Korean clip |
| `runtime.py` | shared environment and cache configuration |
| `http_form.py` | standard-library multipart parser for browser uploads |

The webpage checks, in order: mono 16 kHz; stop if almost silent; trim silent edges only; one official whole-clip label. Denoise (dns48) runs only when the clip is noisy and the top score is below 0.5. One-second slices run only when the top two scores look like two languages. `pip install denoiser` is needed only for that rare denoise branch. Model weights stay frozen.

Do not commit `samples/*.wav` or `challenges/*.wav` (gitignored).
Put those wavs next to the scripts. `data/raw/` is not read.
Metric logs are in `docs/results/`. How-to-run notes: `docs/CODE.md`.
