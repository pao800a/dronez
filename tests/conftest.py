"""Fixtures shared by every test directory (env/RL tests live in unit/, physics/, integration/)."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import pytest

from dronez.cad.artifacts import MissingModelError, load_artifacts
from dronez.cad.schema import load_drone_config

DRONE = "default_5in"


@pytest.fixture(scope="session")
def api() -> ModuleType:
    """``tests/env_api.py``: the single mapping from ADR-0006 names to the Phase 4 code."""
    return importlib.import_module("env_api")


@pytest.fixture(scope="session")
def model_cache() -> None:
    """Skip unless the CAD cache exists (CI builds it before the fast tests)."""
    try:
        load_artifacts(load_drone_config(DRONE))
    except MissingModelError as exc:
        pytest.skip(f"CAD cache missing ({exc}); run `uv run dronez-build-model --drone {DRONE}`")


@pytest.fixture
def mk(api: ModuleType, model_cache: None) -> Iterator[Callable[..., Any]]:
    """Env factory ``mk(env_id="DronezAttitude-v0", **env_cfg_overrides)``; closes what it made."""
    made: list[Any] = []

    def _mk(env_id: str = "DronezAttitude-v0", **cfg: Any) -> Any:
        env = api.make(env_id, **cfg)
        made.append(env)
        return env

    yield _mk
    for env in made:
        env.close()
