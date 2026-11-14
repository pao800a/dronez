"""Locating and loading cached CAD artifacts. No CAD kernel (build123d) imports here.

``sim`` reads the CAD outputs only through this module and ``cad.schema``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dronez.cad.schema import DroneConfig, MassProperties, model_hash
from dronez.utils.paths import models_dir

MANIFEST_NAME = "manifest.json"
MASS_PROPS_NAME = "mass_props.json"
MESHES_DIR = "meshes"


class MissingModelError(FileNotFoundError):
    """The cached model for this config does not exist (run ``dronez-build-model``)."""


@dataclass(frozen=True)
class ModelArtifacts:
    root: Path
    mass_props: MassProperties
    meshes: dict[str, Path]  # part name -> STL (metres, frame C)
    manifest: dict[str, Any]


def model_dir(cfg: DroneConfig) -> Path:
    """``$DRONEZ_DATA_DIR/models/<name>-<hash>/`` (not created)."""
    return models_dir(create=True) / f"{cfg.name}-{model_hash(cfg)}"


def is_valid_cache(root: Path, cfg: DroneConfig) -> bool:
    """A cache dir is valid only if its manifest exists and carries the current hash."""
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(data.get("model_hash") == model_hash(cfg))


def read_artifacts(root: Path) -> ModelArtifacts:
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    mass_props = MassProperties.model_validate_json(
        (root / MASS_PROPS_NAME).read_text(encoding="utf-8")
    )
    meshes = {p.stem: p for p in sorted((root / MESHES_DIR).glob("*.stl"))}
    return ModelArtifacts(root=root, mass_props=mass_props, meshes=meshes, manifest=manifest)


def load_artifacts(cfg: DroneConfig) -> ModelArtifacts:
    root = model_dir(cfg)
    if not is_valid_cache(root, cfg):
        raise MissingModelError(
            f"no built model for {cfg.name!r} (hash {model_hash(cfg)}) in {root}; "
            f"run `uv run dronez-build-model --drone {cfg.name}`"
        )
    return read_artifacts(root)
