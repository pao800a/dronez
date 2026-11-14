"""IMU / attitude measurement model on top of the ground-truth sensors.

MuJoCo's ``sensor_noise`` is not used. ``ImuNoiseModel`` adds, per measurement channel,

* white noise with standard deviation ``density / sqrt(dt)`` (``dt`` = sampling period),
* a constant bias drawn at reset (``bias_init_std``) plus a random walk ``rw * sqrt(dt)`` per step,

and for the attitude a small random rotation vector (``quat_noise_rad``) applied on the true
quaternion. All randomness comes from the ``np.random.Generator`` passed in; the model holds no RNG.
Defaults are order-of-magnitude MEMS values (placeholders, not measured), tunable per config.

**Position / velocity is an estimator abstraction, not a raw sensor.** ``Sensed.pos_w`` and
``Sensed.vel_w`` model the output of a mocap system or a state estimator (EKF with GPS / optical
flow / VIO): the CoM position and velocity in the world frame, corrupted by white noise
(``pos_noise_std``, ``vel_noise_std``). No bias, dropout or latency is modelled yet (placeholders,
not measured). Hover controllers consume them; attitude-only policies ignore them. Their noise is
drawn *after* the IMU channels so the IMU samples of a given seed are unchanged.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from dronez.sim.state import Sensed, TrueState
from dronez.utils import rotations

F64 = NDArray[np.float64]


class ImuNoiseConfig(BaseModel):
    """Noise parameters (SI). ``ImuNoiseConfig.ideal()`` gives a noise-free sensor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gyro_noise_density: float = Field(default=1.5e-3, ge=0.0)  # rad/s/sqrt(Hz)
    gyro_bias_init_std: float = Field(default=5e-3, ge=0.0)  # rad/s
    gyro_bias_rw: float = Field(default=1e-4, ge=0.0)  # rad/s/sqrt(s)
    accel_noise_density: float = Field(default=2e-3, ge=0.0)  # m/s^2/sqrt(Hz)
    accel_bias_init_std: float = Field(default=5e-2, ge=0.0)  # m/s^2
    accel_bias_rw: float = Field(default=1e-3, ge=0.0)  # m/s^2/sqrt(s)
    quat_noise_rad: float = Field(default=2e-3, ge=0.0)  # rad, per-sample rotation-vector std
    pos_noise_std: float = Field(default=2e-3, ge=0.0)  # m, estimator position (mocap-class)
    vel_noise_std: float = Field(default=1e-2, ge=0.0)  # m/s, estimator velocity

    @classmethod
    def ideal(cls) -> ImuNoiseConfig:
        return cls(
            gyro_noise_density=0.0,
            gyro_bias_init_std=0.0,
            gyro_bias_rw=0.0,
            accel_noise_density=0.0,
            accel_bias_init_std=0.0,
            accel_bias_rw=0.0,
            quat_noise_rad=0.0,
            pos_noise_std=0.0,
            vel_noise_std=0.0,
        )


class ImuNoiseModel:
    """Noise model, stateful in the biases only: ``reset(rng)``, then ``apply(true, dt, rng)``."""

    def __init__(self, cfg: ImuNoiseConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else ImuNoiseConfig()
        self.gyro_bias: F64 = np.zeros(3)
        self.accel_bias: F64 = np.zeros(3)

    def reset(self, rng: np.random.Generator, batch_shape: tuple[int, ...] = ()) -> None:
        """Draw fresh constant biases (shape ``batch_shape + (3,)``)."""
        shape = (*batch_shape, 3)
        self.gyro_bias = self.cfg.gyro_bias_init_std * rng.standard_normal(shape)
        self.accel_bias = self.cfg.accel_bias_init_std * rng.standard_normal(shape)

    def apply(self, true: TrueState, dt: float, rng: np.random.Generator) -> Sensed:
        """One measurement taken after ``dt`` seconds since the previous one."""
        c = self.cfg
        gyro = np.asarray(true.angvel_b, dtype=np.float64)
        acc = np.asarray(true.accel_b, dtype=np.float64)
        quat = np.asarray(true.quat_wxyz, dtype=np.float64)
        if self.gyro_bias.shape != gyro.shape:  # first use without reset, or batch change
            self.gyro_bias = np.zeros_like(gyro)
            self.accel_bias = np.zeros_like(acc)
        sqrt_dt = np.sqrt(dt)
        self.gyro_bias = self.gyro_bias + c.gyro_bias_rw * sqrt_dt * rng.standard_normal(gyro.shape)
        self.accel_bias = self.accel_bias + c.accel_bias_rw * sqrt_dt * rng.standard_normal(
            acc.shape
        )
        gyro_m = (
            gyro + self.gyro_bias + c.gyro_noise_density / sqrt_dt * rng.standard_normal(gyro.shape)
        )
        acc_m = (
            acc + self.accel_bias + c.accel_noise_density / sqrt_dt * rng.standard_normal(acc.shape)
        )
        rotvec = c.quat_noise_rad * rng.standard_normal((*quat.shape[:-1], 3))
        angle = np.linalg.norm(rotvec, axis=-1)
        axis = rotvec / np.maximum(angle, 1e-12)[..., None]
        dq = rotations.quat_from_axis_angle(axis, angle)
        quat_m = rotations.quat_normalize(rotations.quat_mul(quat, dq))
        pos = np.asarray(true.pos_w, dtype=np.float64)
        vel = np.asarray(true.linvel_w, dtype=np.float64)
        pos_m = pos + c.pos_noise_std * rng.standard_normal(pos.shape)
        vel_m = vel + c.vel_noise_std * rng.standard_normal(vel.shape)
        return Sensed(quat_wxyz=quat_m, gyro_b=gyro_m, accel_b=acc_m, pos_w=pos_m, vel_w=vel_m)
