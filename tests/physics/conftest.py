"""Fixture layer for the physics sanity suite (``pytest -m physics``).

Every import of ``dronez.sim`` and every assumption about its constructor signatures lives in
THIS file, so a rename or a signature change in ``sim/`` is a one-line fix here. The test modules
only talk to :class:`Rig` and :class:`Plant`.

Assumptions (docs/architecture.md §2 "sim/", §5.2), all re-exported from ``dronez.sim``:
``SimConfig, build_model, DroneSim, PhysParams, MotorParams, MotorModel, hover_command``.
``SimConfig()`` has usable defaults; ``TrueState.accel_b`` is the ideal accelerometer;
``DroneSim(cfg, sim_cfg)`` builds its own model and exposes ``.model`` / ``.index``.

The built model comes from the CAD cache (``$DRONEZ_DATA_DIR/models``); the session fixture skips
with a clear message if it is missing (``uv run dronez-build-model --drone default_5in``).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from dronez.cad.artifacts import MissingModelError, ModelArtifacts, load_artifacts
from dronez.cad.schema import DroneConfig, load_drone_config

F64 = NDArray[np.float64]

DRONE = "default_5in"

# --- the only place that imports dronez.sim -----------------------------------------------------
try:
    from dronez.sim import (
        DroneSim,
        MotorModel,
        MotorParams,
        PhysParams,
        SimConfig,
        build_model,
        hover_command,
    )
except ImportError as exc:  # pragma: no cover - only while sim/ is being written
    _SIM_IMPORT_ERROR: ImportError | None = exc
else:
    _SIM_IMPORT_ERROR = None


def _nominal_params(cfg: DroneConfig, model: Any, index: Any, mass_kg: float) -> Any:
    """Nominal ``PhysParams`` from the drone YAML + the freshly built model."""
    p = cfg.propulsion
    n = len(cfg.motors)

    def full(x: float) -> F64:
        return np.full(n, x, dtype=np.float64)

    motor = MotorParams(
        k_f=full(p.k_f),
        k_m=full(p.k_m),
        tau_m_s=full(p.tau_m_s),
        omega_min=full(p.omega_min_rad_s),
        omega_max=full(p.omega_max_rad_s),
    )
    return PhysParams(
        mass_kg=mass_kg,
        com_offset_b_m=np.zeros(3),
        inertia_diag_kg_m2=np.array(model.body_inertia[index.body_id], dtype=np.float64),
        motor=motor,
        latency_steps=0,
        battery_scale=1.0,
    )


def _make_drone_sim(cfg: DroneConfig, sim_cfg: Any, mass_kg: float) -> tuple[Any, Any, Any, Any]:
    """Return ``(sim, model, index, nominal_params)``. Single point of constructor knowledge."""
    sim = DroneSim(cfg, sim_cfg)  # builds its own MjModel/MjData from the CAD cache
    return sim, sim.model, sim.index, _nominal_params(cfg, sim.model, sim.index, mass_kg)


class Plant:
    """One independent DroneSim (own MjModel/MjData) plus the helpers the tests need."""

    def __init__(self, rig: Rig) -> None:
        self.rig = rig
        self.sim, self.model, self.index, self.nominal = _make_drone_sim(
            rig.cfg, rig.sim_cfg, rig.mass_kg
        )
        self.params = self.nominal
        self.motor = MotorModel(self.nominal.motor)
        self.dt = float(self.model.opt.timestep)

    # --- parameters -----------------------------------------------------------------------
    def set_params(self, params: Any) -> None:
        self.sim.set_params(params)
        self.params = params
        self.motor = MotorModel(params.motor)

    def with_motor(self, **fields: Any) -> Any:
        """Nominal params with some ``MotorParams`` fields replaced (arrays or scalars)."""
        base = self.nominal.motor
        new = {k: np.broadcast_to(np.asarray(v, dtype=np.float64), base.k_f.shape).copy()
               for k, v in fields.items()}  # fmt: skip
        return dataclasses.replace(self.nominal, motor=dataclasses.replace(base, **new))

    def replace(self, **fields: Any) -> Any:
        """Nominal ``PhysParams`` with top-level fields replaced."""
        return dataclasses.replace(self.nominal, **fields)

    # --- command helpers ------------------------------------------------------------------
    def hover_u(self) -> F64:
        return np.asarray(hover_command(self.nominal.motor, self.rig.mass_kg), dtype=np.float64)

    def u_for_thrust(self, thrust_n: Any) -> F64:
        return np.asarray(
            MotorModel(self.nominal.motor).thrust_to_command(np.asarray(thrust_n, dtype=np.float64))
        )

    def omega_for_thrust(self, thrust_n: Any) -> F64:
        """Analytic rotor speed for a static thrust: sqrt(T / k_f) (independent of the sim)."""
        return np.sqrt(np.asarray(thrust_n, dtype=np.float64) / self.rig.k_f)

    # --- simulation -----------------------------------------------------------------------
    def reset_level(
        self, z: float = 1.0, omega0: Any = 0.0, yaw: float = 0.0, qvel: Any = None
    ) -> None:
        """Reset at (0, 0, z), level, at rest, rotors spinning at ``omega0`` (scalar or (4,))."""
        qpos = np.array([0.0, 0.0, z, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        qv = np.zeros(6) if qvel is None else np.asarray(qvel, dtype=np.float64)
        w0 = np.broadcast_to(np.asarray(omega0, dtype=np.float64), (4,)).copy()
        self.sim.reset(qpos, qv, w0)

    def reset_hover(self, z: float = 1.0, thrust: Any = None) -> None:
        """Level, at rest, rotors at the analytic speed for ``thrust`` (default: equal split)."""
        t = np.full(4, self.rig.mass_kg * self.rig.g / 4.0) if thrust is None else thrust
        self.reset_level(z=z, omega0=self.omega_for_thrust(t))

    def substep(self, u: Any) -> None:
        self.sim.substep(np.asarray(u, dtype=np.float64))

    def state(self) -> Any:
        """``TrueState``: pos_w, quat_wxyz, linvel_w, angvel_b, omega, thrust."""
        return self.sim.true_state()

    def accel_b(self) -> F64:
        """Ideal accelerometer (specific force, body frame) from ``TrueState.accel_b``."""
        return np.array(self.state().accel_b, dtype=np.float64)

    def snapshot(self) -> dict[str, F64]:
        """Copy of every model array that ``set_params`` is allowed to write."""
        m, b = self.model, self.index.body_id
        return {
            "body_mass": np.array(m.body_mass[b]),
            "body_inertia": np.array(m.body_inertia[b]),
            "body_ipos": np.array(m.body_ipos[b]),
            "body_iquat": np.array(m.body_iquat[b]),
            "actuator_gear": np.array(m.actuator_gear),
        }

    def rollout(self, u_seq: NDArray[np.float64], force_w: Any = None) -> F64:
        """Step through ``u_seq`` (n, 4); return per-step [pos, quat, linvel, angvel_b, omega]."""
        out = []
        if force_w is not None:
            self.sim.set_wrench(np.asarray(force_w, dtype=np.float64), np.zeros(3))
        for u in u_seq:
            self.substep(u)
            s = self.state()
            out.append(
                np.concatenate(
                    [s.pos_w, s.quat_wxyz, s.linvel_w, s.angvel_b, np.asarray(s.omega).ravel()]
                )
            )
        return np.asarray(out)


class Rig:
    """Session-wide constants (config, CAD mass properties, sim config) and a Plant factory."""

    def __init__(self, cfg: DroneConfig, art: ModelArtifacts, sim_cfg: Any) -> None:
        self.cfg = cfg
        self.art = art
        self.mp = art.mass_props
        self.sim_cfg = sim_cfg
        self.mass_kg = float(self.mp.mass_kg)
        self.g = 9.81
        pr = cfg.propulsion
        self.k_f, self.k_m, self.tau_m = pr.k_f, pr.k_m, pr.tau_m_s
        self.omega_min, self.omega_max = pr.omega_min_rad_s, pr.omega_max_rad_s
        self.spin = np.array(cfg.spin_signs, dtype=np.float64)  # +1 CCW, -1 CW (from the YAML)
        self.hubs = np.array(self.mp.motor_hubs_b_m, dtype=np.float64)  # (4, 3) body FLU, from CAD
        self.inertia = np.array(self.mp.inertia_com_kg_m2, dtype=np.float64)  # (3, 3) FLU axes
        self.dt = float(build_model(cfg, sim_cfg)[0].opt.timestep)

    def trim_thrust(self) -> F64:
        """Per-motor thrust for exact hover: solve ``A T = [m g, 0, 0, 0]`` with the CAD hubs.

        ``A`` columns are ``[1, y_i, -x_i, -s_i k_m/k_f]`` (conventions.md). The CAD centre of
        mass is not exactly at the hub centroid (~0.1 mm), so an equal split leaves a small pitch
        torque; the trim removes it. Built from the YAML/CAD constants only, not from the sim.
        """
        a = np.vstack(
            [
                np.ones(4),
                self.hubs[:, 1],
                -self.hubs[:, 0],
                -self.spin * self.k_m / self.k_f,
            ]
        )
        return np.linalg.solve(a, np.array([self.mass_kg * self.g, 0.0, 0.0, 0.0]))

    def new_plant(self) -> Plant:
        return Plant(self)

    def steps(self, seconds: float) -> int:
        return round(seconds / self.dt)


def _build_rig(dt: float | None) -> Rig:
    if _SIM_IMPORT_ERROR is not None:
        pytest.fail(f"dronez.sim is not importable yet: {_SIM_IMPORT_ERROR}", pytrace=False)
    cfg = load_drone_config(DRONE)
    try:
        art = load_artifacts(cfg)
    except MissingModelError as exc:
        pytest.skip(f"CAD cache missing ({exc}); run `uv run dronez-build-model --drone {DRONE}`")
    sim_cfg = SimConfig()
    if dt is not None:
        sim_cfg = sim_cfg.model_copy(update={"physics_dt_s": dt})
    return Rig(cfg, art, sim_cfg)


@pytest.fixture(scope="session")
def rig() -> Rig:
    """Default ``SimConfig`` (the rate the envs use)."""
    return _build_rig(None)


@pytest.fixture(scope="session", params=[None, 0.002], ids=["default-rate", "500hz"])
def rig_rate(request: pytest.FixtureRequest) -> Rig:
    """The default rate and an explicit 500 Hz (``physics_dt_s = 0.002``) configuration."""
    return _build_rig(request.param)
