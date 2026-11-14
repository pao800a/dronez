"""Motor / propeller model: command -> target speed -> first-order lag -> thrust.

Pure numpy, vectorized over leading batch dimensions (``(n_envs, n_motors)`` or ``(n_motors,)``),
no RNG, no MuJoCo. Stepped at the *physics* rate inside the decimation loop.

* ``command_to_omega``: normalized command ``u in [0, 1]`` -> target speed, **linear in omega**:
  ``omega* = omega_min + u (omega_max - omega_min)`` (open decision 4 of docs/architecture.md §13).
* ``step``: exact discretization of ``d omega/dt = (omega* - omega)/tau``:
  ``omega += (1 - exp(-dt/tau)) (omega* - omega)`` (stable for any dt/tau), then saturation to
  ``[omega_min, omega_max]``; thrust ``T = k_f omega^2``.
* Optional state-dependent effects (``rotor_drag_force_b``, ``ground_effect_gain``) are plain
  functions; the plant applies them behind ``SimConfig`` flags (both off by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dronez.sim.config import GRAVITY_M_S2

if TYPE_CHECKING:
    from dronez.cad.schema import DroneConfig

F64 = NDArray[np.float64]


def _arr(x: ArrayLike) -> F64:
    return np.asarray(x, dtype=np.float64)


@dataclass
class MotorParams:
    """Per-motor constants. Each field is an array broadcastable to ``(..., n_motors)``.

    ``k_f`` [N/(rad/s)^2], ``k_m`` [N m/(rad/s)^2], ``tau_m_s`` [s], ``omega_min/max`` [rad/s].
    Scalars are accepted and stored as float64 arrays.
    """

    k_f: F64
    k_m: F64
    tau_m_s: F64
    omega_min: F64
    omega_max: F64

    def __post_init__(self) -> None:
        self.k_f = _arr(self.k_f)
        self.k_m = _arr(self.k_m)
        self.tau_m_s = _arr(self.tau_m_s)
        self.omega_min = _arr(self.omega_min)
        self.omega_max = _arr(self.omega_max)
        if np.any(self.k_f <= 0.0) or np.any(self.tau_m_s <= 0.0):
            raise ValueError("k_f and tau_m_s must be positive")
        if np.any(self.omega_min < 0.0) or np.any(self.omega_min >= self.omega_max):
            raise ValueError("need 0 <= omega_min < omega_max")

    @classmethod
    def from_drone(cls, cfg: DroneConfig, n_motors: int | None = None) -> MotorParams:
        """Nominal parameters from ``DroneConfig.propulsion`` as ``(n_motors,)`` arrays."""
        n = len(cfg.motors) if n_motors is None else n_motors
        p = cfg.propulsion
        ones = np.ones(n)
        return cls(
            k_f=p.k_f * ones,
            k_m=p.k_m * ones,
            tau_m_s=p.tau_m_s * ones,
            omega_min=p.omega_min_rad_s * ones,
            omega_max=p.omega_max_rad_s * ones,
        )

    def copy(self) -> MotorParams:
        return MotorParams(
            self.k_f.copy(),
            self.k_m.copy(),
            self.tau_m_s.copy(),
            self.omega_min.copy(),
            self.omega_max.copy(),
        )

    def with_battery_scale(self, scale: float | ArrayLike) -> MotorParams:
        """Copy with ``omega_max`` scaled (sagging battery lowers the top speed)."""
        out = self.copy()
        out.omega_max = out.omega_max * _arr(scale)
        return out

    @property
    def k_m_over_k_f(self) -> F64:
        return self.k_m / self.k_f


@dataclass
class MotorModel:
    """First-order motor lag with saturation. State: rotor speed ``omega`` [rad/s]."""

    params: MotorParams
    omega: F64 = field(init=False)

    def __post_init__(self) -> None:
        self.omega = np.broadcast_to(self.params.omega_min, self._shape()).copy()

    def _shape(self) -> tuple[int, ...]:
        p = self.params
        return np.broadcast_shapes(
            p.k_f.shape, p.tau_m_s.shape, p.omega_min.shape, p.omega_max.shape
        )

    def set_params(self, params: MotorParams) -> None:
        """Swap the parameters, keeping the state (clipped to the new limits)."""
        self.params = params
        self.omega = np.clip(
            np.broadcast_to(self.omega, np.broadcast_shapes(self.omega.shape, self._shape())),
            params.omega_min,
            params.omega_max,
        ).copy()

    def reset(self, omega0: ArrayLike) -> None:
        """Set the rotor speeds (clipped to ``[omega_min, omega_max]``), broadcast to the batch."""
        w = _arr(omega0)
        shape = np.broadcast_shapes(w.shape, self._shape())
        self.omega = np.clip(
            np.broadcast_to(w, shape), self.params.omega_min, self.params.omega_max
        ).copy()

    def command_to_omega(self, u: ArrayLike) -> F64:
        """Normalized command ``[0, 1]`` (clipped) -> target rotor speed [rad/s]."""
        p = self.params
        return np.asarray(p.omega_min + np.clip(_arr(u), 0.0, 1.0) * (p.omega_max - p.omega_min))

    def step(self, omega_cmd: ArrayLike, dt: float) -> F64:
        """Advance the lag by ``dt`` toward ``omega_cmd``; return thrust ``T = k_f omega^2`` [N]."""
        alpha = 1.0 - np.exp(-dt / self.params.tau_m_s)
        w = self.omega + alpha * (_arr(omega_cmd) - self.omega)
        self.omega = np.clip(w, self.params.omega_min, self.params.omega_max)
        return self.thrust()

    def thrust(self) -> F64:
        """Thrust of the current state [N]."""
        return np.asarray(self.params.k_f * self.omega**2)

    def thrust_to_command(self, thrust: ArrayLike) -> F64:
        """Inverse of ``command_to_omega`` o ``T = k_f omega^2``; result clipped to ``[0, 1]``."""
        p = self.params
        omega = np.sqrt(np.maximum(_arr(thrust), 0.0) / p.k_f)
        return np.asarray(np.clip((omega - p.omega_min) / (p.omega_max - p.omega_min), 0.0, 1.0))


def hover_command(params: MotorParams, mass_kg: float) -> F64:
    """Normalized command at which the total thrust equals the weight (equal split over motors).

    Returns an array broadcast like ``params`` (one entry per motor).
    """
    n = params.k_f.shape[-1] if params.k_f.ndim else 4
    thrust = mass_kg * GRAVITY_M_S2 / n
    return MotorModel(params).thrust_to_command(thrust)


def hover_thrust_per_motor(mass_kg: float, n_motors: int = 4) -> float:
    """Thrust each motor must produce for hover [N]."""
    return mass_kg * GRAVITY_M_S2 / n_motors


# -- optional state-dependent effects -------------------------------------------------------------
def rotor_drag_force_b(omega: ArrayLike, v_b: ArrayLike, coeff: float) -> F64:
    """Rotor drag force on the body [N], body frame: ``-k_d * sum(omega) * v_xy`` (no z part).

    ``omega`` ``(..., n)``, ``v_b`` ``(..., 3)`` body-frame air-relative velocity [m/s].
    """
    w_sum = np.sum(_arr(omega), axis=-1, keepdims=True)
    v = _arr(v_b)
    out = np.zeros_like(v)
    out[..., :2] = -coeff * w_sum * v[..., :2]
    return out


def ground_effect_gain(height_m: ArrayLike, rotor_radius_m: float, max_gain: float = 1.5) -> F64:
    """Cheeseman-Bennett thrust gain ``1 / (1 - (R / (4 z))^2)`` in ``[1, max_gain]``.

    ``height_m`` is the rotor height above ground (any shape); gain is 1 far from the ground.
    """
    z = np.maximum(_arr(height_m), 1e-6)
    ratio2 = (rotor_radius_m / (4.0 * z)) ** 2
    gain = 1.0 / np.maximum(1.0 - ratio2, 1.0 / max_gain)
    return np.minimum(gain, max_gain)
