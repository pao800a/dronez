"""MP4 rendering of a baseline episode. Skips where no OpenGL backend (EGL/OSMesa/WGL) works."""

from __future__ import annotations

import warnings

import pytest

from dronez.control import CascadedPID, NominalVehicle
from dronez.eval.runner import SimEnv, run_episode
from dronez.eval.scenarios import Scenario, make_plan
from dronez.eval.video import render_video
from dronez.sim import DroneSim

pytestmark = pytest.mark.render


def test_render_recovery_video(rig, tmp_path) -> None:  # type: ignore[no-untyped-def]
    sc = Scenario(name="r", kind="recovery", n_episodes=1, duration_s=0.4, start_height_m=1.5,
                  tilt_deg=30.0)  # fmt: skip
    env = SimEnv(DroneSim(rig.cfg, rig.sim_cfg))
    traj = run_episode(
        env, CascadedPID(NominalVehicle.from_drone(rig.cfg, rig.art)), make_plan(sc, 1)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = render_video(traj.qpos, tmp_path / "v.mp4", fps=25, size=(160, 120))
    if out is None:
        pytest.skip(f"no usable OpenGL backend: {[str(w.message) for w in caught]}")
    assert out.exists()
    assert out.stat().st_size > 1000
