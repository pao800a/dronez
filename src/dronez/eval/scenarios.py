"""Scenario / suite definitions for the baseline evaluation (``configs/eval/baseline.yaml``).

The suite is a pydantic model so its hash (``suite_hash``) identifies cached baseline rows
(docs/architecture.md §8). Initial conditions are drawn from an explicit ``np.random.Generator``
that comes from the per-episode ``SeedSequence`` (§6 seeds), so every controller sees the same
episodes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from dronez.control.base import Setpoint
from dronez.sim.disturbances import DisturbanceConfig, ImpulseSpec
from dronez.utils import rotations
from dronez.utils.config import load_model, stable_hash

F64 = NDArray[np.float64]


class Scenario(BaseModel):
    """One scenario. ``kind`` selects the initial state / setpoint / disturbance recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: Literal["hover_hold", "recovery", "step_attitude", "impulse"]
    n_episodes: int = Field(ge=1)
    duration_s: float = Field(gt=0.0)
    start_height_m: float = 1.0
    tilt_deg: float = 0.0  # initial tilt (recovery) or step magnitude (step_attitude)
    step_time_s: float = 0.5  # time of the attitude step
    impulse_time_s: float = 0.5
    impulse_duration_s: float = 0.02
    impulse_force_n: float = 0.0
    impulse_torque_nm: float = 0.0
    hold_s: float = 1.0  # final window over which success must hold


class EvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    seed: int = 0
    scenarios: tuple[Scenario, ...]

    def hash(self) -> str:
        return stable_hash(self)


def load_suite(name_or_path: str | Path = "baseline") -> EvalSuite:
    return load_model(EvalSuite, name_or_path, "eval")


@dataclass
class EpisodePlan:
    """Everything a runner needs for one episode (no controller or plant objects)."""

    scenario: Scenario
    seed: int
    qpos0: F64
    pos_target: F64
    quat_target_fn: Callable[[float], F64]
    setpoint_fn: Callable[[float], Setpoint]
    disturbance: DisturbanceConfig
    rngs: dict[str, np.random.Generator] = field(default_factory=dict)

    def quat_target(self, t: float) -> F64:
        return self.quat_target_fn(t)

    def setpoint(self, t: float) -> Setpoint:
        return self.setpoint_fn(t)


def episode_seed(suite_seed: int, scenario_idx: int, episode_idx: int) -> int:
    """``SeedSequence([suite_seed, scenario_idx, episode_idx]).generate_state(1)[0]`` (§6)."""
    return int(np.random.SeedSequence([suite_seed, scenario_idx, episode_idx]).generate_state(1)[0])


def _vec3(v: F64) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _random_horizontal_axis(rng: np.random.Generator) -> F64:
    phi = rng.uniform(-np.pi, np.pi)
    return np.array([np.cos(phi), np.sin(phi), 0.0])


def make_plan(scenario: Scenario, seed: int) -> EpisodePlan:
    """Draw the episode from ``seed``: ``spawn(4) -> rng_init, rng_dr, rng_noise, rng_dist``."""
    ss = np.random.SeedSequence(seed)
    rng_init, rng_dr, rng_noise, rng_dist = (np.random.default_rng(s) for s in ss.spawn(4))
    rngs = {"init": rng_init, "dr": rng_dr, "noise": rng_noise, "dist": rng_dist}
    z0 = scenario.start_height_m
    pos_target = np.array([0.0, 0.0, z0])
    identity = rotations.identity_quat()
    disturbance = DisturbanceConfig()
    q0 = identity

    if scenario.kind == "recovery":
        axis = _random_horizontal_axis(rng_init)
        yaw = float(rng_init.uniform(-np.pi, np.pi))
        q_yaw = rotations.quat_from_yaw(yaw)
        q0 = rotations.quat_mul(
            rotations.quat_from_axis_angle(axis, np.deg2rad(scenario.tilt_deg)), q_yaw
        )
        q_target_const = q_yaw  # heading hold: the target differs from q0 by exactly the tilt

        def quat_target(t: float) -> F64:
            return q_target_const

        def setpoint(t: float) -> Setpoint:
            return Setpoint(pos_w=pos_target, quat_wxyz=q_target_const, hold_xy=False)

    elif scenario.kind == "step_attitude":
        axis = _random_horizontal_axis(rng_init)
        q_step = rotations.quat_from_axis_angle(axis, np.deg2rad(scenario.tilt_deg))

        def quat_target(t: float) -> F64:
            return q_step if t >= scenario.step_time_s else identity

        def setpoint(t: float) -> Setpoint:
            return Setpoint(pos_w=pos_target, quat_wxyz=quat_target(t), hold_xy=False)

    else:  # hover_hold, impulse: level, position hold
        if scenario.kind == "impulse":
            f_dir = rng_dist.standard_normal(3)
            f_dir[2] = 0.0
            f_dir /= max(float(np.linalg.norm(f_dir)), 1e-12)
            tau_dir = rng_dist.standard_normal(3)
            tau_dir[2] *= 0.3
            tau_dir /= max(float(np.linalg.norm(tau_dir)), 1e-12)
            disturbance = DisturbanceConfig(
                impulses=(
                    ImpulseSpec(
                        t_s=scenario.impulse_time_s,
                        duration_s=scenario.impulse_duration_s,
                        force_w_n=_vec3(scenario.impulse_force_n * f_dir),
                        torque_w_nm=_vec3(scenario.impulse_torque_nm * tau_dir),
                    ),
                )
            )

        def quat_target(t: float) -> F64:
            return identity

        def setpoint(t: float) -> Setpoint:
            return Setpoint(pos_w=pos_target, yaw=0.0, hold_xy=True)

    qpos0 = np.concatenate([[0.0, 0.0, z0], q0])
    return EpisodePlan(
        scenario=scenario,
        seed=seed,
        qpos0=qpos0,
        pos_target=pos_target,
        quat_target_fn=quat_target,
        setpoint_fn=setpoint,
        disturbance=disturbance,
        rngs=rngs,
    )
