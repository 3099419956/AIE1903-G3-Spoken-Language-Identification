# AIE1903 G3 Spoken Language Identification

Course midterm group project for identifying the language spoken in an audio
recording. The starting point is Meta's MMS language identification model,
`facebook/mms-lid-126`.

> Status: initial repository scaffold. The dataset, experiments, evaluation
> results, and final report are not yet complete.

## Project goal

Build and evaluate a reproducible spoken language identification pipeline:

1. Load and validate audio clips.
2. Run language identification with an MMS-based baseline.
3. Evaluate predictions with suitable classification metrics.
4. Compare at least one extension, such as preprocessing, domain adaptation,
   a different feature extractor, or a stronger classifier.
5. Present the method, results, limitations, and error analysis in a short
   group report.

## Suggested references

- Paper: [Scaling Speech Technology to 1,000+ Languages](https://jmlr.org/papers/v25/23-1318.html)
- Docs: [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- Model: [facebook/mms-lid-126](https://huggingface.co/facebook/mms-lid-126)

## Repository layout

```text
.
├── data/        # Dataset documentation and local data placeholders
├── docs/        # Project notes, reports, and presentation material
├── notebooks/   # Exploratory analysis and reproducible experiments
├── outputs/     # Generated predictions and figures (not committed)
├── scripts/     # Runnable training, inference, and evaluation entry points
└── src/         # Reusable Python package code
```

## Environment

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The model download can be large. Keep the Hugging Face cache outside this
repository if disk space is limited:

```powershell
$env:HF_HOME = "D:\hf-cache"
```

## Collaboration

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
Keep raw audio, downloaded model weights, generated outputs, credentials, and
personal information out of Git.

## Initial milestones

- [ ] Agree on the dataset, language coverage, and train/validation/test split.
- [ ] Reproduce `facebook/mms-lid-126` baseline predictions.
- [ ] Define the primary metric and report per-language results.
- [ ] Implement one extension or controlled comparison.
- [ ] Add an error analysis and document limitations.
- [ ] Prepare the final report and presentation.
