# Data

Store dataset documentation and small metadata files here. Raw and processed
audio are intentionally excluded from Git.

Clips used for the midterm demo (wav files stay on disk, not in Git):

| Clip | Source | Language | Notes |
|------|--------|----------|--------|
| AISHELL-1 `BAC009S0770W0259` | ModelScope `speech_asr_aishell1_testsets` (test) | Mandarin `cmn` | 16 kHz, ~4.2 s |
| LibriSpeech `6930-75918-0000` | Hugging Face `openslr/librispeech_asr` `all/test.clean` | English `eng` | 16 kHz, ~3.5 s |
| MINDS-14 one `ko-KR` row | Hugging Face `PolyAI/minds14` | Korean `kor` | 16 kHz, ~5.2 s |

No train/validation split: we only run the official `facebook/mms-lid-126` checkpoint. `scripts/eval_challenges.py` writes crops (0.5 s, SNR=5, 1 s+1 s concat) to `scripts/challenges/`.

`demo_server.py` reads `scripts/samples/` and `scripts/challenges/` next to itself. Those wavs are gitignored. `data/raw/` is not read by the current scripts.

Local working directories:

```text
data/raw/        # original downloaded data (ignored)
data/processed/  # generated manifests/features (ignored)
data/external/   # third-party resources (ignored)
```
