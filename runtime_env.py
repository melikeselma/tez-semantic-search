import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache" / "huggingface"


def ensure_model_cache_dirs():
    cache_dirs = {
        "HF_HOME": CACHE_ROOT,
        "HF_HUB_CACHE": CACHE_ROOT / "hub",
        "HUGGINGFACE_HUB_CACHE": CACHE_ROOT / "hub",
        "TRANSFORMERS_CACHE": CACHE_ROOT / "transformers",
        "SENTENCE_TRANSFORMERS_HOME": CACHE_ROOT / "sentence_transformers",
    }

    for env_name, path in cache_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(path))
