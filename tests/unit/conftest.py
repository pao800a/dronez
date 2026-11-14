"""Shared fixtures for the sim unit tests: a synthetic (CAD-free) model, no cache needed."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from dronez.cad.artifacts import ModelArtifacts
from dronez.cad.schema import DroneConfig, MassProperties, load_drone_config

# Values of the default_5in CAD build (mass_props.json); only used to make a realistic model.
_COM_C = (5.9e-5, 0.0, 0.0247413)
_MASS = 0.5143276
_MOMENTS = (1.70864e-3, 1.89404e-3, 3.21019e-3)
_QUAT = (0.9999869, 0.0, 0.0051227, 0.0)


@pytest.fixture(scope="session")
def sim_cfg_drone() -> DroneConfig:
    return load_drone_config("default_5in")


@pytest.fixture(scope="session")
def sim_art(sim_cfg_drone: DroneConfig, tmp_path_factory: pytest.TempPathFactory) -> ModelArtifacts:
    """Mass properties as built by the CAD stage, but without meshes (no build123d, no cache)."""
    cfg = sim_cfg_drone
    com = np.array(_COM_C)
    hubs = [tuple(float(v) for v in np.array(p) - com) for p in cfg.motor_positions_c_m]
    mp = MassProperties(
        drone=cfg.name,
        model_hash="synthetic",
        cad_version="test",
        mass_kg=_MASS,
        com_c_m=_COM_C,
        inertia_com_kg_m2=(
            (1.7088e-3, 0.0, 1.5383e-5),
            (0.0, 1.89404e-3, 0.0),
            (1.5383e-5, 0.0, 3.21003e-3),
        ),
        principal_moments_kg_m2=_MOMENTS,
        principal_axes_quat_wxyz=_QUAT,
        parts=[],
        motor_hubs_b_m=(hubs[0], hubs[1], hubs[2], hubs[3]),  # type: ignore[arg-type]
        target_auw_kg=0.55,
        auw_error_frac=-0.065,
        overrides_applied=[],
        source="cad",
    )
    return ModelArtifacts(
        root=tmp_path_factory.mktemp("synthetic_model"), mass_props=mp, meshes={}, manifest={}
    )


@pytest.fixture
def synthetic_env(
    sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts
) -> Iterator[Callable[..., Any]]:
    """``synthetic_env(**env_cfg)``: ``DronezAttitude-v0`` on the synthetic model (no CAD cache),
    ``info_level='full'`` as the eval runner requires. Closes what it made."""
    from dronez.envs import AttitudeEnv, EnvConfig
    from dronez.sim import SimConfig

    made: list[Any] = []

    def _mk(**cfg: Any) -> Any:
        cfg.setdefault("info_level", "full")
        env = AttitudeEnv(EnvConfig(**cfg), SimConfig(), sim_cfg_drone, artifacts=sim_art)
        made.append(env)
        return env

    yield _mk
    for env in made:
        env.close()
