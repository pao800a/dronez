"""Template shared by ``CascadedPID`` and ``GeometricSO3``: outer loop -> inner loop -> mixer.

``act`` pipeline (policy rate, zero-order hold in the plant):

    Sensed -> position loop (optional) -> (R_d, T) -> subclass torque law -> Mixer.allocate
           -> ``thrusts_to_action`` -> action in [-1, 1]^4 (``CommandMap``, ADR-0006 §3)

The observation vector is ignored: baselines read ``Sensed`` and their own state only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dronez.control.base import NominalVehicle, Setpoint, thrusts_to_action
from dronez.control.command_map import CommandMap
from dronez.control.mixer import Mixer, MixInfo
from dronez.control.position import PositionGains, PositionLoop, desired_rotation
from dronez.sim.motor_model import MotorModel
from dronez.sim.state import Sensed
from dronez.utils import rotations

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]

_COS_MIN = 0.5  # never ask for more than 2x thrust to hold altitude on a tilted target


class BaselineController:
    """Common machinery; subclasses implement ``_torque`` and ``_reset_inner``."""

    action_mode: Literal["motors", "ctbr"] = "motors"

    def __init__(
        self,
        vehicle: NominalVehicle,
        position: PositionGains,
        dt: float,
        cmap: CommandMap | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.dt = dt
        self.cmap = cmap if cmap is not None else CommandMap.from_vehicle(vehicle)
        self.mixer = Mixer.from_vehicle(vehicle, self.cmap)
        self.motor = MotorModel(vehicle.motor.copy())
        self.position = PositionLoop(position, dt)
        self.setpoint = Setpoint()
        self.last_mix = MixInfo(False, 1.0, 1.0, 0.0)
        self.last_q_d: F64 = rotations.identity_quat()
        self._inertia = vehicle.inertia_kg_m2

    # -- Controller protocol ---------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Clear integrators and filters. ``seed`` is accepted for protocol parity (no RNG)."""
        del seed
        self.position.reset()
        self.last_mix = MixInfo(False, 1.0, 1.0, 0.0)
        self._reset_inner()

    def set_setpoint(self, setpoint: Setpoint) -> None:
        self.setpoint = setpoint

    def act(self, obs: NDArray[np.float32], sensed: Sensed) -> F32:
        del obs  # baselines use Sensed only
        q = np.asarray(sensed.quat_wxyz, dtype=np.float64)
        r_d, t_total = self._desired(q, sensed)
        self.last_q_d = rotations.rotmat_to_quat(r_d)
        tau = self._torque(q, np.asarray(sensed.gyro_b, dtype=np.float64), r_d)
        thrusts, self.last_mix = self.mixer.allocate_info(t_total, tau)
        return thrusts_to_action(thrusts, self.motor, self.cmap)

    # -- outer loop ------------------------------------------------------------------------------
    def _desired(self, q: F64, sensed: Sensed) -> tuple[F64, float]:
        """Desired attitude matrix and collective thrust [N]."""
        v, sp = self.vehicle, self.setpoint
        r = rotations.quat_to_rotmat(q)
        e3 = np.array([0.0, 0.0, 1.0])
        a_cmd = np.zeros(3)
        if sp.pos_w is not None and sensed.pos_w is not None and sensed.vel_w is not None:
            a_cmd = self.position.update(
                np.asarray(sensed.pos_w, dtype=np.float64),
                np.asarray(sensed.vel_w, dtype=np.float64),
                np.asarray(sp.pos_w, dtype=np.float64),
                np.asarray(sp.vel_w, dtype=np.float64),
                hold_xy=sp.hold_xy and sp.quat_wxyz is None,
                gravity=v.gravity,
            )
        if sp.quat_wxyz is not None:
            r_d = rotations.quat_to_rotmat(sp.quat_wxyz)
            b3d = r_d[:, 2]
            fz = max(v.mass_kg * (v.gravity + a_cmd[2]), 0.0)
            force = fz / max(float(b3d[2]), _COS_MIN) * b3d
        else:
            force = v.mass_kg * (a_cmd + v.gravity * e3)
            if float(np.linalg.norm(force)) < 1e-6:
                force = e3 * 1e-6
            r_d = desired_rotation(force, sp.yaw)
        t_total = max(float(force @ r[:, 2]), 0.0)
        return r_d, t_total

    # -- subclass hooks --------------------------------------------------------------------------
    def _torque(self, q: F64, gyro: F64, r_d: F64) -> F64:
        raise NotImplementedError

    def _reset_inner(self) -> None:
        raise NotImplementedError
