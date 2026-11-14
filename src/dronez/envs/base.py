"""``DronezBaseEnv``: the shared Gymnasium env (ADR-0006, docs/architecture.md §5-§6).

Step loop (motors mode, 100 Hz policy, 1 kHz physics, zero-order hold)::

    a = clip(action, -1, 1);  u = CommandMap(a)
    for k in range(decimation):  wrench -> substep(u) -> termination checks (every substep)
    true = sim.true_state();  sensed = ImuNoiseModel.apply(true);  obs from (sensed, target, hist)
    r, terms = reward(true)        # the reward uses the true state, the obs the sensed one

RNG: ``SeedSequence(episode_seed).spawn(4)`` -> ``rng_init, rng_dr, rng_noise, rng_dist``. Each
stream draws a fixed amount per reset, so changing DR width, noise or disturbances never changes
the ``rng_init`` draws (D2). ``make_env`` never reseeds; SB3 seeds ranks through
``env.reset(seed=...)``.

Subclasses supply: ``_sample_task`` (init + target), ``_extra_obs`` (task-specific obs block),
``_reward_extras``, ``_out_of_bounds`` and the success rule. ``reset(options=...)`` accepts plain
data overrides (ADR-0006 §9): ``qpos0, qvel0, a0, target_schedule, pos_target, phys_params,
disturbance, noise``; ``envs`` knows nothing about ``eval`` types.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from dronez.cad.artifacts import ModelArtifacts, load_artifacts
from dronez.cad.schema import DroneConfig, load_drone_config
from dronez.control.base import NominalVehicle
from dronez.control.command_map import CommandMap
from dronez.envs.config import EnvConfig
from dronez.envs.curriculum import LevelSpec
from dronez.envs.obs import HISTORY_K, N_MOTORS, ObsLayout
from dronez.envs.randomization import DomainRandomizer, draw_disturbance, scale_noise
from dronez.envs.rewards import Kind, RewardWeights, compute_reward, term_names
from dronez.sim import (
    DisturbanceConfig,
    DisturbanceModel,
    DroneSim,
    ImuNoiseConfig,
    ImuNoiseModel,
    PhysParams,
    Sensed,
    SimConfig,
    TrueState,
    build_model_from_artifacts,
    load_sim_config,
)
from dronez.utils import rotations

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]

TERMINATIONS = ("crash", "rate_limit", "ground_contact", "out_of_bounds", "time_limit")
_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass
class Task:
    """Initial state and target of one episode (all plain numpy)."""

    qpos0: F64
    qvel0: F64
    a0: F64
    pos_target: F64
    #: ``[(t_s, quat_wxyz), ...]`` sorted by time; the first entry is at ``t = 0``
    schedule: list[tuple[float, F64]] = field(default_factory=list)


# -- small sampling helpers (shared by the envs) ---------------------------------------------------
def unit_vector(cos_z: float, phi: float) -> F64:
    s = math.sqrt(max(1.0 - cos_z * cos_z, 0.0))
    return np.array([s * math.cos(phi), s * math.sin(phi), cos_z])


def horizontal_axis(phi: float) -> F64:
    return np.array([math.cos(phi), math.sin(phi), 0.0])


def cos_uniform_angle(u: float, max_angle: float) -> float:
    """Angle with ``cos(angle) ~ U(cos(max_angle), 1)`` (uniform on a spherical cap)."""
    c = math.cos(max_angle)
    return math.acos(min(max(c + u * (1.0 - c), -1.0), 1.0))


class DronezBaseEnv(gym.Env[F32, F32]):
    """Common machinery; see the module docstring."""

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012
    KIND: Kind = "attitude"
    DEFAULT_EPISODE_S: float = 5.0
    DEFAULT_LEVELS: tuple[LevelSpec, ...] = ()

    def __init__(
        self,
        env_cfg: EnvConfig | dict[str, Any] | None = None,
        sim_cfg: SimConfig | str | None = None,
        drone: DroneConfig | str = "default_5in",
        render_mode: str | None = None,
        artifacts: ModelArtifacts | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None:
            raise ValueError("DronezBaseEnv has no rendering; use dronez.eval.video")
        self.render_mode = None
        if env_cfg is None:
            env_cfg = EnvConfig()
        elif isinstance(env_cfg, dict):
            env_cfg = EnvConfig.model_validate(env_cfg)
        self.cfg: EnvConfig = env_cfg
        self.sim_cfg: SimConfig = (
            sim_cfg if isinstance(sim_cfg, SimConfig) else load_sim_config(sim_cfg or "default")
        )
        self.drone_cfg: DroneConfig = (
            drone if isinstance(drone, DroneConfig) else load_drone_config(drone)
        )
        art = artifacts if artifacts is not None else load_artifacts(self.drone_cfg)
        self.vehicle = NominalVehicle.from_drone(self.drone_cfg, art)
        self.cmap = CommandMap.from_vehicle(self.vehicle, self.cfg.authority)
        model, index = build_model_from_artifacts(self.drone_cfg, art, self.sim_cfg)
        self.sim = DroneSim(self.drone_cfg, self.sim_cfg, model=model, index=index)
        self.nominal: PhysParams = self.sim.nominal_params()
        self.randomizer = DomainRandomizer(self.nominal, self.cfg.dr, self.sim_cfg.physics_dt_s)
        self.model_hash: str = art.mass_props.model_hash
        self.dt = self.sim_cfg.policy_dt_s
        self.decimation = self.sim_cfg.decimation
        self._default_len_s = self.cfg.episode_len_s or self.DEFAULT_EPISODE_S
        self.episode_len_s = self._default_len_s
        self.max_steps = round(self.episode_len_s / self.dt)
        self.levels: tuple[LevelSpec, ...] = self.cfg.levels or self.DEFAULT_LEVELS
        if self.cfg.start_level >= len(self.levels):
            raise ValueError(f"start_level {self.cfg.start_level} >= {len(self.levels)} levels")
        self._level = self.cfg.start_level
        self._ep_level = self._level
        self.weights = RewardWeights.for_kind(self.KIND, self.cfg.reward)
        self.layout = self._make_layout()
        bound = self.layout.bound.astype(np.float32)
        self.observation_space = spaces.Box(-bound, bound, (self.layout.dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (N_MOTORS,), np.float32)
        self.reward_terms = term_names(self.KIND)

        self._win_steps = max(round(self.cfg.success_window_s / self.dt), 1)
        self._att_ok_rad = math.radians(self.cfg.success_att_deg)
        self._rate_limit_sq = self.cfg.rate_limit_rad_s**2
        self._noise = ImuNoiseModel(self.cfg.noise)
        self._dist = DisturbanceModel()
        self._rng_noise = np.random.default_rng(0)
        self._buf = np.zeros(self.layout.dim, dtype=np.float32)
        self._hist = np.zeros(HISTORY_K * N_MOTORS)
        self._a_prev = np.zeros(N_MOTORS)
        self._true: TrueState = self.sim.true_state()
        self._sensed: Sensed | None = None
        self._last_obs = np.zeros(self.layout.dim, dtype=np.float32)
        self._params: PhysParams = self.nominal
        self._task: Task | None = None
        self._schedule: list[tuple[float, F64]] = []
        self._sched_i = 0
        self._q_t: F64 = _IDENTITY
        self._r_t: F64 = np.eye(3)
        self._pos_t: F64 = np.zeros(3)
        self._p0: F64 = np.zeros(3)
        self._step_n = 0
        self._ok_steps = 0
        self._has_dist = False
        self._force = np.zeros(3)
        self._torque = np.zeros(3)
        self._episode_seed = 0
        self._done = True

    # -- hooks -------------------------------------------------------------------------------
    def _make_layout(self) -> ObsLayout:
        raise NotImplementedError

    def _sample_task(self, rng: np.random.Generator, level: int) -> Task:
        raise NotImplementedError

    def _extra_obs(self, out: F32, sensed: Sensed) -> None:
        """Fill the task-specific block (default: none)."""

    def _reward_extras(self, true: TrueState) -> tuple[float, float, float]:
        """``(vz, vel_norm, pos_err)`` of the true state for the reward."""
        raise NotImplementedError

    def _out_of_bounds(self, qpos: F64) -> bool:
        raise NotImplementedError

    def _step_ok(self, att_err: float, pos_err: float) -> bool:
        """Does this step count toward the final-window success rule?"""
        raise NotImplementedError

    def _pos_err_h(self, true: TrueState) -> F64:
        return np.zeros(3)

    # -- curriculum ----------------------------------------------------------------------------
    def set_curriculum_level(self, level: int | float) -> None:
        """Set the level used from the next reset on (called through ``VecEnv.env_method``)."""
        k = int(level)
        if not 0 <= k < len(self.levels):
            raise ValueError(f"level {k} outside [0, {len(self.levels) - 1}]")
        self._level = k

    def get_curriculum_level(self) -> int:
        return self._level

    # -- gym API -------------------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[F32, dict[str, Any]]:
        super().reset(seed=seed)
        opts = options or {}
        ep_seed = int(seed) if seed is not None else int(self.np_random.integers(2**63))
        self._episode_seed = ep_seed
        # eval may set the episode length per episode (plain data); default: the env's own
        self.episode_len_s = float(opts.get("episode_len_s") or self._default_len_s)
        self.max_steps = round(self.episode_len_s / self.dt)
        rng_init, rng_dr, rng_noise, rng_dist = (
            np.random.default_rng(s) for s in np.random.SeedSequence(ep_seed).spawn(4)
        )
        self._ep_level = self._level
        spec = self.levels[self._ep_level]

        task = self._sample_task(rng_init, self._ep_level)
        self._apply_task_options(task, opts)
        draw = self.randomizer.draw(rng_dr, spec.dr_scale)
        dist_draw = draw_disturbance(
            rng_dist, self.cfg.dr, spec.dr_scale, self.episode_len_s, spec.disturbances
        )
        params: PhysParams = (
            opts["phys_params"] if opts.get("phys_params") is not None else draw.phys
        )
        noise_cfg: ImuNoiseConfig = (
            opts["noise"]
            if opts.get("noise") is not None
            else scale_noise(self.cfg.noise, draw.noise_scale)
        )
        dist_cfg: DisturbanceConfig = (
            opts["disturbance"] if opts.get("disturbance") is not None else dist_draw
        )

        sim = self.sim
        sim.set_params(params)
        omega0 = sim.motor.command_to_omega(self.cmap.action_to_command(task.a0))
        sim.reset(task.qpos0, task.qvel0, omega0)
        self._params = params
        self._noise = ImuNoiseModel(noise_cfg)
        self._noise.reset(rng_noise)
        self._rng_noise = rng_noise
        self._dist.reset(rng_dist, dist_cfg)
        self._has_dist = bool(
            dist_cfg.impulses
            or dist_cfg.random_impulse_count
            or any(dist_cfg.wind_mean_w_m_s)
            or dist_cfg.wind_gust_std_m_s > 0.0
        )
        self._force = np.zeros(3)
        self._torque = np.zeros(3)

        self._task = task
        self._p0 = task.qpos0[:3].copy()
        self._pos_t = task.pos_target.copy()
        self._schedule = task.schedule
        self._sched_i = 0
        self._set_target(self._schedule[0][1])
        self._step_n = 0
        self._ok_steps = 0
        self._a_prev = task.a0.copy()
        self._hist = np.tile(task.a0, HISTORY_K)
        self._done = False

        self._true = sim.true_state()
        self._sensed = self._noise.apply(self._true, self.dt, self._rng_noise)
        obs = self._build_obs(self._sensed)
        self._last_obs = obs
        info = self._base_info(self._true, self._sensed)
        info["termination"] = None
        return obs.copy(), info

    def step(self, action: Any) -> tuple[F32, float, bool, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(N_MOTORS), -1.0, 1.0)
        a_prev = self._a_prev
        sim, d = self.sim, self.sim.data
        reason: str | None = None
        if not (np.isfinite(a).all() and np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            reason = "crash"
        else:
            u = self.cmap.action_to_command(a)
            ground_z, rate_sq = self.cfg.ground_z_m, self._rate_limit_sq
            has_dist, dist, true = self._has_dist, self._dist, self._true
            bid = sim.index.body_id
            for _ in range(self.decimation):
                if has_dist:
                    self._force, self._torque = dist.wrench(sim.time_s, true)
                    sim.set_wrench(self._force, self._torque)
                sim.substep(u)
                # CoM z (ADR-0006 §5); xipos is from the substep's forward pass (<= 1 ms old)
                if d.ncon > 0 or d.xipos[bid, 2] < ground_z:
                    reason = "ground_contact"
                    break
                w = d.qvel[3:6]
                if float(w @ w) > rate_sq:
                    reason = "rate_limit"
                    break
                if self._out_of_bounds(d.qpos):
                    reason = "out_of_bounds"
                    break
        true = sim.true_state()
        finite = bool(
            np.isfinite(true.pos_w).all()
            and np.isfinite(true.quat_wxyz).all()
            and np.isfinite(true.linvel_w).all()
            and np.isfinite(true.angvel_b).all()
        )
        if not finite:
            reason = "crash"
        self._true = true
        self._step_n += 1
        self._advance_target(self._step_n * self.dt)
        if finite:
            sensed = self._noise.apply(true, self.dt, self._rng_noise)
        else:  # keep the last measurement (copied: never share arrays between calls)
            assert self._sensed is not None
            sensed = copy.deepcopy(self._sensed)
        self._sensed = sensed
        if finite and np.isfinite(a).all():
            self._hist[N_MOTORS:] = self._hist[:-N_MOTORS].copy()
            self._hist[:N_MOTORS] = a
            obs = self._build_obs(sensed)
            if not np.isfinite(obs).all():
                reason, obs = "crash", self._last_obs
        else:
            reason, obs = "crash", self._last_obs
        self._last_obs = obs

        terminated = reason is not None
        truncated = (not terminated) and self._step_n >= self.max_steps
        if truncated:
            reason = "time_limit"

        # reward (true state); non-finite states score as the worst case
        if finite:
            att_err = float(rotations.geodesic_angle(true.quat_wxyz, self._q_t))
            vz, vel_norm, pos_err = self._reward_extras(true)
            ang = float(np.linalg.norm(true.angvel_b))
        else:
            att_err, vz, vel_norm, pos_err, ang = math.pi, 0.0, 0.0, 0.0, self.cfg.rate_limit_rad_s
        r, terms = compute_reward(
            self.KIND,
            self.weights,
            att_err=att_err,
            angvel_norm=ang,
            vz=vz,
            vel_norm=vel_norm,
            pos_err=pos_err,
            a=a if np.isfinite(a).all() else np.zeros(N_MOTORS),
            a_prev=a_prev,
            terminated=terminated,
        )
        self._ok_steps = self._ok_steps + 1 if self._step_ok(att_err, pos_err) else 0
        if np.isfinite(a).all():
            self._a_prev = a
        self._done = terminated or truncated

        info = self._base_info(true, sensed)
        info["reward_terms"] = terms
        info["metrics"] = {
            "att_err_rad": att_err,
            "tilt_rad": float(rotations.tilt_angle(true.quat_wxyz)) if finite else math.pi,
            "saturation_frac": float(np.mean(np.abs(a) >= self.weights.sat_threshold))
            if np.isfinite(a).all()
            else 1.0,
            **self._extra_metrics(pos_err),
        }
        info["termination"] = reason
        info["episode_seed"] = self._episode_seed
        info["curriculum_level"] = self._ep_level
        if self._done:
            info["episode_success"] = bool((not terminated) and self._ok_steps >= self._win_steps)
        return obs.copy(), r, terminated, truncated, info

    def close(self) -> None:
        return

    # -- internals -----------------------------------------------------------------------------
    def _extra_metrics(self, pos_err: float) -> dict[str, float]:
        return {}

    def _apply_task_options(self, task: Task, opts: dict[str, Any]) -> None:
        if "qpos0" in opts:
            task.qpos0 = np.asarray(opts["qpos0"], dtype=np.float64).copy()
        if "qvel0" in opts:
            task.qvel0 = np.asarray(opts["qvel0"], dtype=np.float64).copy()
        if "a0" in opts:
            task.a0 = np.clip(np.asarray(opts["a0"], dtype=np.float64), -1.0, 1.0)
        if "pos_target" in opts:
            task.pos_target = np.asarray(opts["pos_target"], dtype=np.float64).copy()
        if "target_schedule" in opts:
            sched = sorted(
                (
                    (float(t), np.asarray(q, dtype=np.float64).copy())
                    for t, q in opts["target_schedule"]
                ),
                key=lambda x: x[0],
            )
            if not sched or sched[0][0] > 0.0:
                sched.insert(0, (0.0, task.schedule[0][1]))
            task.schedule = sched

    def _set_target(self, q: F64) -> None:
        self._q_t = q
        self._r_t = rotations.quat_to_rotmat(q)

    def _advance_target(self, t: float) -> None:
        sched, i = self._schedule, self._sched_i
        while i + 1 < len(sched) and sched[i + 1][0] <= t + 1e-9:
            i += 1
        if i != self._sched_i:
            self._sched_i = i
            self._set_target(sched[i][1])

    def _attitude_features(self, q: F64, out: NDArray[Any]) -> F64:
        """Write ``r_err_6d`` and ``gravity_b`` for attitude ``q`` into ``out[0:9]``; return R."""
        r = rotations.quat_to_rotmat(q)
        r_err = self._r_t.T @ r
        out[0:3] = r_err[:, 0]
        out[3:6] = r_err[:, 1]
        out[6:9] = -r[2, :]
        return np.asarray(r)

    def _build_obs(self, s: Sensed) -> F32:
        buf = self._buf
        self._attitude_features(np.asarray(s.quat_wxyz, dtype=np.float64), buf)
        buf[9:12] = s.gyro_b
        assert s.vel_w is not None
        buf[12] = s.vel_w[2]
        buf[13:29] = self._hist
        self._extra_obs(buf, s)
        return buf.copy()

    def _privileged_obs(self, true: TrueState) -> F32:
        p, nm = self._params, self.nominal
        out = np.zeros(self.layout.privileged_dim, dtype=np.float32)
        sl = self.layout.privileged_slices
        tmp = np.zeros(9)
        self._attitude_features(true.quat_wxyz, tmp)
        out[sl["r_err_6d"]] = tmp[0:6]
        out[sl["gravity_b"]] = tmp[6:9]
        out[sl["angvel_b"]] = true.angvel_b
        out[sl["linvel_w"]] = true.linvel_w
        out[sl["pos_err_h"]] = self._pos_err_h(true)
        out[sl["omega_norm"]] = true.omega / nm.motor.omega_max
        out[sl["mass_scale"]] = p.mass_kg / nm.mass_kg
        out[sl["inertia_scale"]] = p.inertia_diag_kg_m2 / nm.inertia_diag_kg_m2
        out[sl["com_offset_b_m"]] = p.com_offset_b_m
        out[sl["k_f_scale"]] = p.motor.k_f / nm.motor.k_f
        out[sl["k_m_scale"]] = float(
            np.mean((p.motor.k_m / p.motor.k_f) / (nm.motor.k_m / nm.motor.k_f))
        )
        out[sl["tau_m_s"]] = float(np.mean(p.motor.tau_m_s))
        out[sl["latency_s"]] = p.latency_steps * self.sim_cfg.physics_dt_s
        out[sl["battery_scale"]] = p.battery_scale
        out[sl["force_w_n"]] = self._force
        out[sl["torque_w_nm"]] = self._torque
        out[sl["wind_w_m_s"]] = self._dist.wind_w(self.sim.time_s) if self._has_dist else 0.0
        return out

    def _base_info(self, true: TrueState, sensed: Sensed) -> dict[str, Any]:
        if self.cfg.info_level == "minimal":
            mini: dict[str, Any] = {
                "episode_seed": self._episode_seed,
                "curriculum_level": self._ep_level,
            }
            if self.cfg.privileged_obs:
                mini["privileged_obs"] = self._privileged_obs(true)
            return mini
        p = self._params
        info: dict[str, Any] = {
            "sensed": {
                "quat_wxyz": sensed.quat_wxyz,
                "gyro_b": sensed.gyro_b,
                "accel_b": sensed.accel_b,
                "pos_w": sensed.pos_w,
                "vel_w": sensed.vel_w,
            },
            "true": {
                "pos_w": true.pos_w,
                "quat_wxyz": true.quat_wxyz,
                "linvel_w": true.linvel_w,
                "angvel_b": true.angvel_b,
                "omega_rotor": true.omega,
                "thrust_n": true.thrust,
            },
            "params": {
                "mass_kg": p.mass_kg,
                "com_offset_b_m": p.com_offset_b_m.copy(),
                "inertia_diag_kg_m2": p.inertia_diag_kg_m2.copy(),
                "k_f": p.motor.k_f.copy(),
                "k_m": p.motor.k_m.copy(),
                "tau_m_s": p.motor.tau_m_s.copy(),
                "omega_max_rad_s": p.motor.omega_max * p.battery_scale,
                "latency_steps": p.latency_steps,
                "battery_scale": p.battery_scale,
            },
            "disturbance": {
                "force_w_n": self._force.copy(),
                "torque_w_nm": self._torque.copy(),
                "wind_w_m_s": self._dist.wind_w(self.sim.time_s) if self._has_dist else np.zeros(3),
            },
            "target": {"quat_wxyz": self._q_t.copy(), "pos_w": self._pos_t.copy()},
            "episode_seed": self._episode_seed,
            "curriculum_level": self._ep_level,
        }
        if self.cfg.privileged_obs:
            info["privileged_obs"] = self._privileged_obs(true)
        return info
