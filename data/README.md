# Data

Store dataset documentation and small metadata files here. Raw and processed
audio are intentionally excluded from Git.

Add the following information before experiments begin:

- Dataset name and source URL
- License and permitted use
- Language coverage and class distribution
- Audio format, sample rate, and clip duration
- Train/validation/test split strategy
- Known quality issues and exclusions

Local working directories:

```text
data/raw/        # original downloaded data (ignored)
data/processed/  # generated manifests/features (ignored)
data/external/   # third-party resources (ignored)
```
