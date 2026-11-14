"""Plain data records exchanged between the plant, the sensor model and the env layer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def _zeros3() -> F64:
    return np.zeros(3)


@dataclass
class TrueState:
    """Ground truth. World frame z-up; body FLU; quaternion (w, x, y, z), body -> world.

    ``pos_w`` and ``linvel_w`` refer to the **centre of mass**. ``angvel_b`` is the body-frame
    angular velocity. ``omega`` is the rotor speed [rad/s] and ``thrust`` the motor thrust [N],
    both ``(n_motors,)``. ``accel_b`` is the ideal accelerometer (specific force at the IMU site,
    body frame), needed by the IMU noise model.
    """

    pos_w: F64
    quat_wxyz: F64
    linvel_w: F64
    angvel_b: F64
    omega: F64
    thrust: F64
    accel_b: F64 = field(default_factory=_zeros3)


@dataclass
class Sensed:
    """What a flight controller would have: attitude estimate, gyro and accelerometer (body FLU)."""

    quat_wxyz: F64
    gyro_b: F64
    accel_b: F64
