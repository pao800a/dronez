"""YAML config loading and stable hashing.

Typed config schemas (drone, sim, train, eval) are pydantic models defined
next to the code that consumes them; this module only provides the generic
loading, merging and hashing helpers they share.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from dronez.utils.paths import configs_dir

M = TypeVar("M", bound=BaseModel)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping; an empty file yields ``{}``."""
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_config_path(path_or_name: str | Path, kind: str) -> Path:
    """Accept a path, or a bare name looked up as ``configs/<kind>/<name>.yaml``."""
    p = Path(path_or_name)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    candidate = configs_dir() / kind / f"{p.stem}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"config {path_or_name!r} not found (also tried {candidate})")


def load_model[M: BaseModel](model_cls: type[M], path_or_name: str | Path, kind: str) -> M:
    """Load and validate a YAML config into a pydantic model."""
    return model_cls.model_validate(load_yaml(resolve_config_path(path_or_name, kind)))


def stable_hash(obj: Any, length: int = 10) -> str:
    """Deterministic short SHA-256 of a JSON-serialisable object or pydantic model."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]
