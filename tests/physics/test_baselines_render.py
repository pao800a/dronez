"""MP4 rendering of a baseline episode. Skips where no OpenGL backend (EGL/OSMesa/WGL) works."""

from __future__ import annotations

import warnings

import pytest

import dronez.envs  # noqa: F401 - registers the env ids
from dronez.envs import AttitudeEnv, EnvConfig
from dronez.eval.controllers import ControllerSpec
from dronez.eval.runner import run_episode
from dronez.eval.scenarios import Scenario, make_plan
from dronez.eval.video import render_video

pytestmark = pytest.mark.render


def test_render_recovery_video(rig, tmp_path) -> None:  # type: ignore[no-untyped-def]
    sc = Scenario(name="r", kind="recovery", n_episodes=1, duration_s=0.4, start_height_m=1.5,
                  tilt_deg=30.0)  # fmt: skip
    env = AttitudeEnv(EnvConfig(info_level="full"), rig.sim_cfg, rig.cfg, artifacts=rig.art)
    traj = run_episode(env, ControllerSpec("pid")(), make_plan(sc, 1))
    env.close()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = render_video(traj.qpos, tmp_path / "v.mp4", fps=25, size=(160, 120))
    if out is None:
        pytest.skip(f"no usable OpenGL backend: {[str(w.message) for w in caught]}")
    assert out.exists()
    assert out.stat().st_size > 1000
