"""``DroneSim``: the deterministic, RNG-free plant (one MjModel + one MjData + motors + delay).

Per physics step (``substep``): ``ActionDelay.push`` -> ``command_to_omega`` -> ``MotorModel.step``
-> ``data.ctrl = T`` (N; the actuator gear adds the yaw reaction ``-s_i k_m/k_f T``) ->
``xfrc_applied`` (external wrench at the CoM, plus optional rotor drag) -> ``mj_step``.

Randomized parameters are written by ``set_params`` **from the nominal snapshot**
(``ModelIndex.nominal``), then ``mj_setConst`` (required for mass/inertia/ipos edits, verified)
is run on a scratch ``MjData`` so a mid-episode call cannot corrupt the live state; call ``reset``
afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from dronez.cad.schema import DroneConfig
from dronez.sim.builder import ModelIndex, build_model
from dronez.sim.config import GRAVITY_M_S2, SimConfig
from dronez.sim.disturbances import ActionDelay
from dronez.sim.motor_model import (
    MotorModel,
    MotorParams,
    ground_effect_gain,
    hover_command,
    rotor_drag_force_b,
)
from dronez.sim.state import TrueState
from dronez.utils import rotations

F64 = NDArray[np.float64]


@dataclass
class PhysParams:
    """Everything domain randomization may change. Plain dataclass (also goes into ``info``).

    ``com_offset_b_m`` is added to the nominal CoM (``ipos``) in the body frame. ``inertia_diag``
    are the principal moments [kg m^2] about the CoM, in the nominal inertial frame.
    ``battery_scale`` multiplies ``omega_max`` of every motor.
    """

    mass_kg: float
    com_offset_b_m: F64
    inertia_diag_kg_m2: F64
    motor: MotorParams
    latency_steps: int = 0
    battery_scale: float = 1.0


def inertia_triangle_ok(inertia_diag: ArrayLike, tol: float = 0.0) -> bool:
    """True if the principal moments are positive and satisfy ``I_a + I_b >= I_c`` for all perms."""
    i = np.asarray(inertia_diag, dtype=np.float64)
    if i.shape != (3,) or np.any(i <= 0.0):
        return False
    total = float(np.sum(i))
    return bool(np.all(total - 2.0 * i >= -tol))  # I_a + I_b - I_c = total - 2 I_c


def project_inertia_triangle(inertia_diag: ArrayLike, margin: float = 1e-3) -> F64:
    """Minimal fix of invalid principal moments: raise the two smaller ones until valid.

    Used by randomizers when a draw violates the triangle inequality (flat quads sit near
    ``I_zz = I_xx + I_yy``).
    """
    i = np.asarray(inertia_diag, dtype=np.float64).copy()
    k = int(np.argmax(i))
    others = [j for j in range(3) if j != k]
    deficit = i[k] * (1.0 + margin) - (i[others[0]] + i[others[1]])
    if deficit > 0.0:
        i[others] += 0.5 * deficit
    return i


class DroneSim:
    """Single-env plant. Build with ``DroneSim(cfg, sim)`` (needs the cached CAD artifacts)."""

    def __init__(
        self,
        cfg: DroneConfig,
        sim: SimConfig | None = None,
        *,
        model: mujoco.MjModel | None = None,
        index: ModelIndex | None = None,
    ) -> None:
        self.cfg = cfg
        self.sim = sim if sim is not None else SimConfig()
        if model is None or index is None:
            model, index = build_model(cfg, self.sim)
        self.model = model
        self.index = index
        self.data = mujoco.MjData(model)
        self._scratch = mujoco.MjData(model)
        self.dt = float(model.opt.timestep)
        self.nominal_motor = MotorParams.from_drone(cfg)
        self.motor = MotorModel(self.nominal_motor.copy())
        self.delay = ActionDelay(0)
        self.params = self.nominal_params()
        self._force_w = np.zeros(3)
        self._torque_w = np.zeros(3)
        self._acc = index.sensor_slices["accelerometer"]
        self._vel = index.sensor_slices["framelinvel"]
        self._body = index.body_id
        self._sites = index.motor_site_ids
        self._rotor_r = index.rotor_radius_m
        self.reset(self.default_qpos(), np.zeros(6), self.hover_omega())

    # -- parameters ---------------------------------------------------------------------------
    def nominal_params(self) -> PhysParams:
        nom = self.index.nominal
        return PhysParams(
            mass_kg=nom.mass,
            com_offset_b_m=np.zeros(3),
            inertia_diag_kg_m2=nom.inertia.copy(),
            motor=self.nominal_motor.copy(),
            latency_steps=0,
            battery_scale=1.0,
        )

    def set_params(self, p: PhysParams) -> None:
        """Write ``p`` into the model (from the nominal snapshot) and the motor model.

        Raises ``ValueError`` if the inertia violates the triangle inequality or ``mass_kg <= 0``.
        Call ``reset`` afterwards (this does not touch the live ``MjData`` state).
        """
        if p.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        inertia = np.asarray(p.inertia_diag_kg_m2, dtype=np.float64)
        if not inertia_triangle_ok(inertia):
            raise ValueError(
                f"inertia {inertia} violates positivity / the triangle inequality; "
                "use project_inertia_triangle"
            )
        m, b, nom = self.model, self.index.body_id, self.index.nominal
        m.body_mass[b] = p.mass_kg
        m.body_ipos[b] = nom.ipos + np.asarray(p.com_offset_b_m, dtype=np.float64)
        m.body_inertia[b] = inertia
        m.body_iquat[b] = nom.iquat
        motor = p.motor.with_battery_scale(p.battery_scale)
        m.actuator_gear[self.index.actuator_ids, 5] = -self.index.spin * motor.k_m_over_k_f
        mujoco.mj_setConst(m, self._scratch)
        self.motor.set_params(motor)
        self.delay.set_steps(p.latency_steps)
        self.params = p

    def hover_command(self) -> F64:
        """Normalized command that balances the current mass with the current motor parameters."""
        return hover_command(self.motor.params, float(self.model.body_mass[self._body]))

    def trim_thrusts(self) -> F64:
        """Per-motor thrusts [N] with total = weight and zero net torque about the *current* CoM.

        Solves the Quad-X mixer ``A T = [m g, 0, 0, 0]`` with ``A[:, i] = [1, y_i, -x_i,
        -s_i k_m/k_f]`` and the hubs measured from the current CoM (nominal CoM + DR offset). The
        symmetric ``hover_command`` differs from this by the CoM offset of the real airframe.
        """
        m, idx = self.model, self.index
        rel = m.site_pos[self._sites] - m.body_ipos[self._body]
        ratio = self.motor.params.k_m_over_k_f
        a = np.vstack([np.ones(len(rel)), rel[:, 1], -rel[:, 0], -idx.spin * ratio])
        return np.asarray(np.linalg.solve(a, np.array([self.weight_n(), 0.0, 0.0, 0.0])))

    def trim_command(self) -> F64:
        """Normalized command realizing ``trim_thrusts`` (clipped to [0, 1])."""
        return self.motor.thrust_to_command(self.trim_thrusts())

    def hover_omega(self) -> F64:
        return self.motor.command_to_omega(self.hover_command())

    def default_qpos(self, height_m: float = 1.0) -> F64:
        """Level attitude at ``height_m`` above the ground."""
        return np.array([0.0, 0.0, height_m, 1.0, 0.0, 0.0, 0.0])

    # -- stepping -----------------------------------------------------------------------------
    def reset(self, qpos: ArrayLike, qvel: ArrayLike, omega0: ArrayLike) -> None:
        """Reset to ``qpos`` (7: body-origin position, quaternion wxyz), ``qvel`` (6: world linear
        velocity of the body origin, body-frame angular velocity) and rotor speeds ``omega0``."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
        self.motor.reset(omega0)
        self.delay.reset()
        self._force_w[:] = 0.0
        self._torque_w[:] = 0.0
        self.data.ctrl[:] = self.motor.thrust()
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_wrench(self, force_w: ArrayLike, torque_w: ArrayLike) -> None:
        """External world-frame force [N] (at the CoM) and torque [N m] for the next step(s)."""
        self._force_w[:] = np.asarray(force_w, dtype=np.float64)
        self._torque_w[:] = np.asarray(torque_w, dtype=np.float64)

    def substep(self, u: ArrayLike) -> None:
        """One physics step with normalized motor commands ``u`` (4,), in [0, 1]."""
        d = self.data
        u_delayed = self.delay.push(u)
        thrust = self.motor.step(self.motor.command_to_omega(u_delayed), self.dt)
        if self.sim.ground_effect:
            thrust = thrust * ground_effect_gain(
                d.site_xpos[self._sites, 2], self._rotor_r, self.sim.ground_effect_max_gain
            )
        d.ctrl[:] = thrust
        force = self._force_w
        if self.sim.rotor_drag:
            quat = d.xquat[self._body]
            v_b = rotations.quat_rotate_inverse(quat, d.cvel[self._body, 3:])
            f_b = rotor_drag_force_b(self.motor.omega, v_b, self.sim.rotor_drag_coeff)
            force = force + rotations.quat_rotate(quat, f_b)
        d.xfrc_applied[self._body, :3] = force
        d.xfrc_applied[self._body, 3:] = self._torque_w
        mujoco.mj_step(self.model, d)

    # -- readout ------------------------------------------------------------------------------
    @property
    def time_s(self) -> float:
        return float(self.data.time)

    def true_state(self) -> TrueState:
        """Ground truth after the last step.

        ``mj_step`` evaluates sensors on the state *before* integration, so a ``mj_forward`` is run
        first: the accelerometer and CoM position/velocity then belong to the current state.
        """
        d = self.data
        mujoco.mj_forward(self.model, d)
        s = self.index.sensor_slices
        return TrueState(
            pos_w=d.xipos[self._body].copy(),
            quat_wxyz=d.xquat[self._body].copy(),
            linvel_w=d.sensordata[self._vel].copy(),
            angvel_b=d.qvel[3:6].copy(),
            omega=self.motor.omega.copy(),
            thrust=d.ctrl.copy(),
            accel_b=d.sensordata[s["accelerometer"]].copy(),
        )

    def weight_n(self) -> float:
        return float(self.model.body_mass[self._body]) * GRAVITY_M_S2
