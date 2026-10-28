"""Load config.yaml and resolve project paths."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]          # project root (folder with config.yaml)
DEFAULT_CONFIG = ROOT / "config.yaml"

log = logging.getLogger(__name__)


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML config. The file location is remembered under cfg['_path']."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = str(path)
    return cfg


def resolve(path: str | Path) -> Path:
    """Relative paths are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def dir_path(cfg: dict, key: str, create: bool = False) -> Path:
    """paths.<key> as an absolute Path (optionally created)."""
    p = resolve(cfg["paths"][key])
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def config_hash(cfg: dict) -> str:
    """Stable short hash of the config (without private keys)."""
    public = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(public, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def git_commit() -> str | None:
    """Current git commit if the project is a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # git missing or not a repo
        return None


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
