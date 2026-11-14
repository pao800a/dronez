"""Checkpoints: ``model.zip`` and ``vecnormalize.pkl`` are **always** written together.

``checkpoints/step_<012d>/{model.zip, vecnormalize.pkl, ckpt.json}`` and ``checkpoints/best/``. A
directory is built under a temporary name and renamed into place, so a crash never leaves a
half-written checkpoint (and never a model without its normalization statistics).
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from dronez.rl.run_dir import utc_now
from dronez.utils import paths

MODEL_FILE = "model.zip"
VECNORM_FILE = "vecnormalize.pkl"
META_FILE = "ckpt.json"


def save_checkpoint(
    model: PPO, vecnorm: VecNormalize | None, path: Path, meta: dict[str, Any] | None = None
) -> Path:
    """Write ``path/{model.zip, vecnormalize.pkl, ckpt.json}`` atomically (tmp dir + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp-{path.name}-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    model.save(str(tmp / MODEL_FILE))
    if vecnorm is not None:
        vecnorm.save(str(tmp / VECNORM_FILE))
    body = {"created_at": utc_now(), **(meta or {})}
    (tmp / META_FILE).write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    old = None
    if path.exists():
        old = path.with_name(f".old-{path.name}-{os.getpid()}")
        os.replace(path, old)
    os.replace(tmp, path)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    return path


def load_vecnormalize(path: Path, venv: VecEnv | None = None) -> VecNormalize:
    """The pickled ``VecNormalize`` (with ``venv`` attached when given)."""
    file = Path(path) / VECNORM_FILE if Path(path).is_dir() else Path(path)
    if venv is not None:
        return VecNormalize.load(str(file), venv)
    with file.open("rb") as fh:
        vn: VecNormalize = pickle.load(fh)
    return vn


def load_checkpoint(
    path: Path | str, venv: VecEnv | None = None, device: str = "cpu"
) -> tuple[PPO, VecNormalize]:
    """``(PPO, VecNormalize)`` from a checkpoint directory. With ``venv`` both are bound to it."""
    path = Path(path)
    model = PPO.load(str(path / MODEL_FILE), env=venv, device=device)
    return model, load_vecnormalize(path, venv)


def read_meta(path: Path) -> dict[str, Any]:
    f = Path(path) / META_FILE
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def list_checkpoints(run_root: Path) -> list[Path]:
    """``step_*`` directories (complete ones only), oldest first."""
    d = Path(run_root) / "checkpoints"
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.glob("step_*") if (p / MODEL_FILE).exists() and (p / VECNORM_FILE).exists()
    )


def latest_checkpoint(run_root: Path) -> Path | None:
    cks = list_checkpoints(run_root)
    return cks[-1] if cks else None


def resolve_checkpoint(run_id: str, ckpt: str | None = None) -> Path:
    """``runs/<run_id>/checkpoints/<ckpt>``; ``ckpt`` = ``best`` | ``latest`` | ``step_...`` |
    ``None`` (best if present, else latest)."""
    root = paths.runs_dir(run_id, create=False)
    if ckpt in (None, "best"):
        best = root / "checkpoints" / "best"
        if (best / MODEL_FILE).exists():
            return best
        if ckpt == "best":
            raise FileNotFoundError(f"no best checkpoint in {root}")
    if ckpt in (None, "latest"):
        latest = latest_checkpoint(root)
        if latest is None:
            raise FileNotFoundError(f"no checkpoints in {root}")
        return latest
    p = root / "checkpoints" / str(ckpt)
    if not (p / MODEL_FILE).exists():
        raise FileNotFoundError(f"checkpoint {p} not found")
    return p


def prune_checkpoints(run_root: Path, keep: int = 5) -> None:
    for p in list_checkpoints(run_root)[:-keep]:
        shutil.rmtree(p, ignore_errors=True)
