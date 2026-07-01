"""Persistent user settings for the desktop app, stored at ~/.attest/config.json.

Seeded from the same ATTEST_* environment variables / .env the CLI uses, so the
app works out of the box if you've already configured the CLI, and remembers your
choices (models, pipeline toggles, theme) across launches.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("ATTEST_HOME", Path.home() / ".attest"))
CONFIG_PATH = CONFIG_DIR / "config.json"


def defaults() -> dict:
    return {
        "theme": "dark",  # "dark" | "light"
        "provider": {
            "base_url": os.environ.get("ATTEST_BASE_URL", ""),
            "api_key": os.environ.get("ATTEST_API_KEY", ""),
        },
        "models": {
            "generator": os.environ.get("ATTEST_MODEL", ""),
            "judge": os.environ.get("ATTEST_JUDGE_MODEL", "openai/gpt-4o-mini"),
            "vision": os.environ.get("ATTEST_VISION_MODEL", "openai/gpt-4o-mini"),
            "embedder": "local",  # "local" | "mock"
        },
        "pipeline": {
            "lexical": True,   # BM25 keyword fusion (recall)
            "expand": True,    # HyDE query expansion (recall)
            "rerank": False,   # cross-encoder precision (slow on CPU; great on GPU)
            "verify": True,    # judge-check that cited passages support the answer
            "k": 8,
            "allow_uncited": False,  # if abstained, also show the model's own answer (labelled)
        },
        "index_path": "",  # the active index directory
    }


def load_config() -> dict:
    cfg = defaults()
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            saved = {}
        for key, value in saved.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
