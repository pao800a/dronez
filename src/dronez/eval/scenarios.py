"""Scenario / suite definitions (``configs/eval/<suite>.yaml``).

The suite is a pydantic model so its hash (``suite_hash``) identifies cached baseline rows
(docs/architecture.md §8). Initial conditions are drawn from an explicit ``np.random.Generator``
that comes from the per-episode ``SeedSequence`` (§6 seeds), so every controller sees the same
episodes (paired comparison).

ADR-0006 §9 schema additions: ``tilt_min_deg`` / ``tilt_max_deg`` / ``tilt_dist``,
``rate_max_rad_s``, ``dr`` (``nominal | fixed | sample``, overrides, scale) and ``latency_s``
(fixed or a range). The env receives the resolved episode as plain data through
``reset(options=...)`` (see ``runner.plan_options``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from dronez.control.base import Setpoint
from dronez.sim.disturbances import DisturbanceConfig, ImpulseSpec
from dronez.utils import rotations
from dronez.utils.config import load_model, stable_hash

F64 = NDArray[np.float64]


class DrSpec(BaseModel):
    """Plant randomization of a scenario.

    * ``nominal``: nominal plant (d = 0).
    * ``fixed``: ``overrides`` are scale factors of the nominal: ``mass``, ``inertia``, ``k_f``,
      ``k_m``, ``battery``, ``noise`` (sensor-noise scale), ``tau_m_s`` (absolute [s]),
      ``com_mm`` (3), ``mismatch`` (``roll`` | ``yaw``: +-5 % per-motor thrust constant).
    * ``sample``: a fresh ``DomainRandomizer`` draw at width ``scale`` for each episode.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["nominal", "fixed", "sample"] = "nominal"
    overrides: dict[str, Any] = Field(default_factory=dict)
    scale: float = Field(default=1.0, ge=0.0, le=1.0)


class Scenario(BaseModel):
    """One scenario. ``kind`` selects the initial state / setpoint / disturbance recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: Literal["hover_hold", "recovery", "step_attitude", "impulse"]
    n_episodes: int = Field(ge=1)
    duration_s: float = Field(gt=0.0)
    start_height_m: float = 1.0
    tilt_deg: float = 0.0  # initial tilt (recovery, ``tilt_dist: fixed``) or step magnitude
    tilt_dist: Literal["fixed", "cap_uniform", "so3_uniform"] = "fixed"
    tilt_min_deg: float = 0.0  # cap_uniform: cos(delta) ~ U(cos max, cos min)
    tilt_max_deg: float = 0.0
    rate_max_rad_s: float = 0.0  # |omega0| ~ U(0, max), random direction
    random_yaw: bool = False  # hover_hold: heading target psi ~ U(-pi, pi) (recovery: always)
    step_time_s: float = 0.5  # time of the attitude step
    impulse_time_s: float = 0.5
    impulse_duration_s: float = 0.02
    impulse_force_n: float = 0.0
    impulse_torque_nm: float = 0.0
    hold_s: float = 1.0  # final window over which success must hold
    t_max_s: float | None = None  # success_recovery(t_max): recovered and stays within t_max
    dr: DrSpec = DrSpec()
    latency_s: float | tuple[float, float] = 0.0  # fixed [s] or a (lo, hi) range


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
    """Everything a runner needs for one episode (no controller, plant or env objects)."""

    scenario: Scenario
    seed: int
    qpos0: F64
    pos_target: F64
    quat_target_fn: Callable[[float], F64]
    setpoint_fn: Callable[[float], Setpoint]
    disturbance: DisturbanceConfig
    rngs: dict[str, np.random.Generator] = field(default_factory=dict)
    qvel0: F64 = field(default_factory=lambda: np.zeros(6))
    a0: F64 = field(default_factory=lambda: np.zeros(4))
    #: ``[(t_s, quat_wxyz), ...]`` target schedule (first entry at t = 0)
    target_schedule: list[tuple[float, F64]] = field(default_factory=list)
    #: resolved latency [s] for this episode
    latency_s: float = 0.0

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


def _tilt_angle(sc: Scenario, rng: np.random.Generator) -> float:
    """Initial geodesic error delta [rad]."""
    if sc.tilt_dist == "fixed":
        return float(np.deg2rad(sc.tilt_deg))
    if sc.tilt_dist == "so3_uniform":
        lo, hi = 0.0, math.pi
    else:
        lo, hi = math.radians(sc.tilt_min_deg), math.radians(sc.tilt_max_deg)
    cos_d = rng.uniform(math.cos(hi), math.cos(lo))
    return float(math.acos(min(max(cos_d, -1.0), 1.0)))


def _random_rate(sc: Scenario, rng: np.random.Generator) -> F64:
    if sc.rate_max_rad_s <= 0.0:
        return np.zeros(3)
    mag = rng.uniform(0.0, sc.rate_max_rad_s)
    d = rng.standard_normal(3)
    return np.asarray(mag * d / max(float(np.linalg.norm(d)), 1e-12))


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
    omega0 = np.zeros(3)
    schedule: list[tuple[float, F64]]

    if scenario.kind == "recovery":
        axis = _random_horizontal_axis(rng_init)
        yaw = float(rng_init.uniform(-np.pi, np.pi))
        q_yaw = rotations.quat_from_yaw(yaw)
        delta = _tilt_angle(scenario, rng_init)
        q0 = rotations.quat_mul(rotations.quat_from_axis_angle(axis, delta), q_yaw)
        omega0 = _random_rate(scenario, rng_init)
        q_target_const = q_yaw  # heading hold: the target differs from q0 by exactly the tilt
        schedule = [(0.0, q_target_const)]

        def quat_target(t: float) -> F64:
            return q_target_const

        def setpoint(t: float) -> Setpoint:
            return Setpoint(pos_w=pos_target, quat_wxyz=q_target_const, hold_xy=False)

    elif scenario.kind == "step_attitude":
        axis = _random_horizontal_axis(rng_init)
        q_step = rotations.quat_from_axis_angle(axis, np.deg2rad(scenario.tilt_deg))
        schedule = [(0.0, identity), (scenario.step_time_s, q_step)]

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
        if scenario.random_yaw:
            q_hold = rotations.quat_from_yaw(float(rng_init.uniform(-np.pi, np.pi)))
            q0 = q_hold
            schedule = [(0.0, q_hold)]

            def quat_target(t: float) -> F64:
                return q_hold

            def setpoint(t: float) -> Setpoint:
                return Setpoint(pos_w=pos_target, quat_wxyz=q_hold, hold_xy=False)

        else:
            schedule = [(0.0, identity)]

            def quat_target(t: float) -> F64:
                return identity

            def setpoint(t: float) -> Setpoint:
                return Setpoint(pos_w=pos_target, yaw=0.0, hold_xy=True)

    lat = scenario.latency_s
    latency = float(rng_dr.uniform(lat[0], lat[1])) if isinstance(lat, tuple) else float(lat)
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
        qvel0=np.concatenate([np.zeros(3), omega0]),
        a0=np.zeros(4),
        target_schedule=schedule,
        latency_s=latency,
    )
