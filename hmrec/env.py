"""Minimal ``.env`` loader. Values are passed on to child processes and never printed."""

from __future__ import annotations

import os
from pathlib import Path


def read_env(path: Path | str = ".env") -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines (``#`` comments, optional quotes). Missing file -> empty dict."""
    p = Path(path)
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            out[key.strip()] = value
    return out


def kaggle_env(path: Path | str = ".env") -> dict[str, str]:
    """Environment for the ``kaggle`` CLI: current env plus any ``KAGGLE_*`` keys from ``.env``."""
    env = dict(os.environ)
    env.update({k: v for k, v in read_env(path).items() if k.startswith("KAGGLE_")})
    return env


def has_kaggle_credentials(path: Path | str = ".env") -> bool:
    """True if a usable credential is configured (checks key names only, never exposes values)."""
    env = kaggle_env(path)
    return bool(env.get("KAGGLE_API_TOKEN")) or (
        bool(env.get("KAGGLE_USERNAME")) and bool(env.get("KAGGLE_KEY"))
    )
