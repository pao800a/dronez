"""Eval harness: controllers see ``Sensed`` only; suite aggregation, G3 gate and report files."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from dronez.cad.artifacts import ModelArtifacts
from dronez.cad.schema import DroneConfig
from dronez.control import CascadedPID, NominalVehicle
from dronez.eval.report import write_report
from dronez.eval.runner import run_episode
from dronez.eval.scenarios import EvalSuite, Scenario, episode_seed, load_suite, make_plan
from dronez.eval.suite import run_baseline_suite
from dronez.eval.video import gl_backends, render_video
from dronez.sim import Sensed

pytestmark = pytest.mark.fast


@pytest.fixture
def env(synthetic_env: Callable[..., Any]) -> Any:
    """``DronezAttitude-v0`` on the synthetic model (the runner drives the env, §7)."""
    return synthetic_env()


class _Spy:
    action_mode = "motors"

    def __init__(self, u: np.ndarray) -> None:
        self.u = u.astype(np.float32)
        self.seen: list[tuple[object, object]] = []

    def reset(self, seed: int | None = None) -> None:
        self.seen.clear()

    def act(self, obs: np.ndarray, sensed: Sensed) -> np.ndarray:
        self.seen.append((obs, sensed))
        return self.u


def test_controller_receives_only_obs_and_sensed(env: Any) -> None:
    sc = Scenario(name="h", kind="hover_hold", n_episodes=1, duration_s=0.2)
    spy = _Spy(np.full(4, 0.3))
    traj = run_episode(env, spy, make_plan(sc, 1))
    assert len(spy.seen) == 20
    assert traj.u.shape == (20, 4)
    for obs, sensed in spy.seen:
        assert isinstance(obs, np.ndarray)
        assert isinstance(sensed, Sensed)


def test_policy_rate_and_seeding(env: Any) -> None:
    assert env.dt == pytest.approx(0.01)
    assert env.decimation == 10
    assert episode_seed(0, 1, 2) == episode_seed(0, 1, 2) != episode_seed(0, 1, 3)
    sc = load_suite("baseline").scenarios[1]
    p1, p2 = make_plan(sc, episode_seed(0, 1, 0)), make_plan(sc, episode_seed(0, 1, 0))
    np.testing.assert_array_equal(p1.qpos0, p2.qpos0)


def test_crash_is_detected(env: Any) -> None:
    sc = Scenario(name="h", kind="hover_hold", n_episodes=1, duration_s=2.0)
    traj = run_episode(env, _Spy(np.full(4, -1.0)), make_plan(sc, 1))  # a = -1: idle, falls
    assert traj.crashed
    assert len(traj.t) < 201


def test_suite_gate_and_report_files(env: Any, vehicle_art: NominalVehicle, tmp_path) -> None:
    hover = Scenario(name="hover_hold", kind="hover_hold", n_episodes=1, duration_s=2.0)
    recovery = Scenario(
        name="recovery_30deg", kind="recovery", n_episodes=2, duration_s=1.5,
        start_height_m=1.5, tilt_deg=30.0,
    )  # fmt: skip
    suite = EvalSuite(name="mini", scenarios=(hover, recovery))
    res = run_baseline_suite(CascadedPID(vehicle_art), env, suite)
    extra = {"noise": {}, "policy_rate_hz": 100.0, "physics_rate_hz": 1000.0}
    m = write_report(tmp_path, "pid", suite, "synthetic", res, extra)
    assert m["gates"]["G3"]["pass"]
    for name in (
        "metrics.json",
        "report.md",
        "suite.resolved.yaml",
        "plots/hover_hold.png",
        "trajectories/recovery_30deg_ep0.npz",
    ):
        assert (tmp_path / name).exists()
    loaded = json.loads((tmp_path / "metrics.json").read_text())
    assert loaded["scenarios"]["recovery_30deg"]["aggregate"]["success_rate"] == 1.0
    assert loaded["suite_hash"] == suite.hash()
    # degrees only in the human report
    assert "[deg]" in (tmp_path / "report.md").read_text()


@pytest.fixture
def vehicle_art(sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts) -> NominalVehicle:
    return NominalVehicle.from_drone(sim_cfg_drone, sim_art)


def test_render_never_sets_mujoco_gl_globally(monkeypatch, tmp_path) -> None:
    """Backend selection is per child process; the parent environment is untouched."""
    import os

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    assert gl_backends()  # non-empty list of backends to try
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        render_video(np.tile([0, 0, 1.0, 1, 0, 0, 0], (2, 1)), tmp_path / "x.mp4", timeout_s=1e-3)
    assert "MUJOCO_GL" not in os.environ


def test_render_degrades_to_none_with_warning(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("dronez.eval.video.gl_backends", lambda: ["definitely-not-a-gl-backend"])
    with pytest.warns(RuntimeWarning, match="no video"):
        out = render_video(np.tile([0, 0, 1.0, 1, 0, 0, 0], (2, 1)), tmp_path / "y.mp4")
    assert out is None
