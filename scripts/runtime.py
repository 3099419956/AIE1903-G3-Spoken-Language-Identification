"""Shared runtime settings for command-line and web entry points."""

import os
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent


def configure_cache() -> str:
    """Use standard caches unless the team opts into a shared cache directory."""
    cache_root = os.environ.get("AIE1903_CACHE_DIR", "").strip()
    if cache_root:
        root = Path(cache_root).expanduser()
        os.environ.setdefault("HF_HOME", str(root / "huggingface"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "huggingface" / "hub"))
        os.environ.setdefault("TORCH_HOME", str(root / "torch"))

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return os.environ.get("HF_HOME", "default Hugging Face cache")
