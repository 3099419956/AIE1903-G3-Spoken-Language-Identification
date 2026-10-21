# Contributing

## Branching

Use `main` for reviewed, working code. Create a short-lived branch for each task:

```text
feat/<name>-baseline-inference
fix/<name>-audio-resampling
docs/<name>-experiment-notes
```

Keep commits focused and use clear messages:

```text
feat: add MMS baseline inference
fix: resample audio to 16 kHz
docs: summarize first evaluation results
```

## Pull requests

1. Pull the latest `main` before starting work.
2. Describe what changed, why it changed, and how it was checked.
3. Include the command or notebook used for verification.
4. Request at least one teammate to review before merging.
5. Do not merge generated artifacts or unrelated formatting changes.

## Data and artifacts

- Commit lightweight documentation, code, configs, and small test fixtures.
- Do not commit raw audio, private data, model weights, checkpoints, or large
  experiment outputs.
- Record dataset provenance and licensing in `data/README.md`.
- Put aggregate metrics and final figures in the report; keep local full
  outputs under `outputs/`.

## Reproducibility

- Fix random seeds where supported.
- Record package versions, model names, and evaluation settings.
- Prefer scripts or parameterized notebooks over unexplained manual steps.
- Document any sample exclusions or preprocessing decisions.
