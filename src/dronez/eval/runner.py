"""Minimal closed-loop runner over ``DroneSim`` at the policy rate (docs/architecture.md §6-§8).

The loop the eval harness will also use with the Gymnasium env in Phase 4:

    sensed = env.reset(plan); loop: a = controller.act(obs, sensed); sensed = env.step(a)

``ClosedLoopEnv`` is the tiny protocol between runner and plant; ``SimEnv`` implements it with
``DroneSim`` + ``ImuNoiseModel`` + ``DisturbanceModel`` (policy rate 100 Hz = 10 physics substeps
per action, zero-order hold). Phase 4 swaps in an adapter around ``DronezBaseEnv`` without touching
``run_episode``. Controllers never see ground truth: only ``Sensed`` reaches ``act``; the
``true`` record is kept for metrics.

Baseline controllers additionally accept a ``Setpoint`` (``set_setpoint``); policies ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from dronez.control.base import Controller
from dronez.eval.scenarios import EpisodePlan
from dronez.sim.disturbances import DisturbanceModel
from dronez.sim.plant import DroneSim
from dronez.sim.sensors import ImuNoiseConfig, ImuNoiseModel
from dronez.sim.state import Sensed, TrueState

F64 = NDArray[np.float64]

CRASH_HEIGHT_M = 0.05  # CoM below this = ground contact
CRASH_TILT_RAD = float(np.deg2rad(120.0))


class ClosedLoopEnv(Protocol):
    """What ``run_episode`` needs from a plant (``SimEnv`` now, the Gymnasium env in Phase 4)."""

    dt: float

    def reset(self, plan: EpisodePlan) -> Sensed: ...

    def step(self, action: NDArray[np.float32]) -> Sensed: ...

    def true_state(self) -> TrueState: ...

    def qpos(self) -> F64: ...


@dataclass
class Trajectory:
    """Recorded episode. State arrays have ``n + 1`` samples (index 0 = reset), ``u`` has ``n``."""

    scenario: str
    seed: int
    dt: float
    t: F64
    pos_w: F64
    quat_wxyz: F64
    linvel_w: F64
    angvel_b: F64
    qpos: F64  # (n+1, 7) for video replay
    quat_target: F64
    pos_target: F64
    u: F64  # (n, 4) applied action
    sensed_quat: F64
    crashed: bool

    def to_npz(self) -> dict[str, np.ndarray]:
        return {
            k: np.asarray(v)
            for k, v in self.__dict__.items()
            if k not in {"scenario", "crashed"} and not isinstance(v, str)
        } | {"crashed": np.asarray(self.crashed)}


class SimEnv:
    """``ClosedLoopEnv`` on a ``DroneSim`` with sensor noise and scripted disturbances."""

    def __init__(self, sim: DroneSim, noise: ImuNoiseConfig | None = None) -> None:
        self.sim = sim
        self.noise = ImuNoiseModel(noise)
        self.dist = DisturbanceModel()
        self.decimation = sim.sim.decimation
        self.dt = sim.sim.policy_dt_s
        self._rng_noise = np.random.default_rng(0)
        self._true = sim.true_state()

    def reset(self, plan: EpisodePlan) -> Sensed:
        sim = self.sim
        self._rng_noise = plan.rngs["noise"]
        self.noise.reset(self._rng_noise)
        self.dist.reset(plan.rngs["dist"], plan.disturbance)
        omega0 = sim.motor.command_to_omega(sim.trim_command())
        sim.reset(plan.qpos0, np.zeros(6), omega0)
        self._true = sim.true_state()
        return self.noise.apply(self._true, self.dt, self._rng_noise)

    def step(self, action: NDArray[np.float32]) -> Sensed:
        u = np.asarray(action, dtype=np.float64)
        sim = self.sim
        for _ in range(self.decimation):
            force, torque = self.dist.wrench(sim.time_s, self._true)
            sim.set_wrench(force, torque)
            sim.substep(u)
        self._true = sim.true_state()
        return self.noise.apply(self._true, self.dt, self._rng_noise)

    def true_state(self) -> TrueState:
        return self._true

    def qpos(self) -> F64:
        return np.asarray(self.sim.data.qpos, dtype=np.float64).copy()


def run_episode(env: ClosedLoopEnv, controller: Controller, plan: EpisodePlan) -> Trajectory:
    """Closed loop for ``plan.scenario.duration_s``; the controller sees ``Sensed`` only."""
    sc = plan.scenario
    n = round(sc.duration_s / env.dt)
    controller.reset(plan.seed)
    sensed = env.reset(plan)
    obs = np.zeros(0, dtype=np.float32)
    set_sp = getattr(controller, "set_setpoint", None)

    true0 = env.true_state()
    pos, quat, vel, ang, qpos, qtgt, sq = ([x] for x in _snap(env, true0, sensed))
    qtgt[0] = plan.quat_target(0.0)
    u_log: list[F64] = []
    t = 0.0
    crashed = False
    for k in range(n):
        t = k * env.dt
        if set_sp is not None:
            set_sp(plan.setpoint(t))
        action = controller.act(obs, sensed)
        u_log.append(np.asarray(action, dtype=np.float64))
        sensed = env.step(action)
        tr = env.true_state()
        s = _snap(env, tr, sensed)
        for lst, val in zip((pos, quat, vel, ang, qpos, qtgt, sq), s, strict=True):
            lst.append(val)
        qtgt[-1] = plan.quat_target((k + 1) * env.dt)
        if tr.pos_w[2] < CRASH_HEIGHT_M or _tilt(tr.quat_wxyz) > CRASH_TILT_RAD:
            crashed = True
            break
    nn = len(pos)
    return Trajectory(
        scenario=sc.name,
        seed=plan.seed,
        dt=env.dt,
        t=np.arange(nn, dtype=np.float64) * env.dt,
        pos_w=np.array(pos),
        quat_wxyz=np.array(quat),
        linvel_w=np.array(vel),
        angvel_b=np.array(ang),
        qpos=np.array(qpos),
        quat_target=np.array(qtgt),
        pos_target=np.tile(plan.pos_target, (nn, 1)),
        u=np.array(u_log),
        sensed_quat=np.array(sq),
        crashed=crashed,
    )


def _snap(env: ClosedLoopEnv, tr: TrueState, sensed: Sensed) -> tuple[F64, ...]:
    return (
        np.asarray(tr.pos_w, dtype=np.float64).copy(),
        np.asarray(tr.quat_wxyz, dtype=np.float64).copy(),
        np.asarray(tr.linvel_w, dtype=np.float64).copy(),
        np.asarray(tr.angvel_b, dtype=np.float64).copy(),
        env.qpos(),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.asarray(sensed.quat_wxyz, dtype=np.float64).copy(),
    )


def _tilt(q: F64) -> float:
    from dronez.utils import rotations  # noqa: PLC0415

    return float(rotations.tilt_angle(q))
