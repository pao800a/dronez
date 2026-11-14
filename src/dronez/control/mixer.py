"""Quad-X motor mixer derived from the drone config (never typed by hand).

``[T, tau_x, tau_y, tau_z]^T = A @ thrusts`` with column ``i`` of ``A`` equal to
``[1, y_i, -x_i, -s_i k_m/k_f]`` (docs/conventions.md). ``(x_i, y_i)`` are the hub positions
relative to the **centre of mass** (CAD mass properties), ``s_i`` the spin sign from the YAML.

Saturation handling (``allocate``), in priority order roll/pitch > collective thrust > yaw:

1. Solve ``T = A^-1 w``. If every thrust is inside ``[T_min_i, T_max_i]``, done.
2. Otherwise drop yaw first: ``T = A^-1 [T, tau_x, tau_y, s*tau_z]`` with the largest
   ``s in [0, 1]`` that fits (a 1-D interval problem per motor).
3. If even ``s = 0`` does not fit, shift the collective: ``T + delta * A^-1 e_T`` with the
   ``delta`` of smallest magnitude that fits (thrust is sacrificed, roll/pitch moments are
   exactly preserved because ``A^-1 e_T`` produces no moment).
4. If no shift fits, the roll/pitch demand itself is too large: scale ``(tau_x, tau_y)`` by the
   largest ``gamma in [0, 1]`` (bisection) for which a shift exists, keeping their ratio, then
   apply 3 and 2 with the remaining margin.
5. A final clip to the limits guards against round-off.

Every step is a pure function of ``(T, tau)`` (no state), so the mapping is deterministic and
batch-free. ``allocate_info`` also reports which step engaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dronez.control.base import NominalVehicle

F64 = NDArray[np.float64]

_EPS = 1e-12


@dataclass(frozen=True)
class MixInfo:
    """Which saturation step engaged (all ``False`` / 1.0 when the request fit)."""

    saturated: bool
    yaw_scale: float  # fraction of the requested yaw torque kept
    rp_scale: float  # fraction of the requested roll/pitch torque kept
    collective_shift_n: float  # total-thrust change [N] traded for attitude


def _interval(base: F64, direction: F64, lo: F64, hi: F64) -> tuple[float, float]:
    """Range of ``s`` with ``lo <= base + s * direction <= hi`` (empty -> ``(inf, -inf)``)."""
    s_lo, s_hi = -np.inf, np.inf
    for b, d, a, c in zip(base, direction, lo, hi, strict=True):
        if abs(d) < _EPS:
            if b < a - 1e-12 or b > c + 1e-12:
                return np.inf, -np.inf
            continue
        s1, s2 = (a - b) / d, (c - b) / d
        s_lo = max(s_lo, min(s1, s2))
        s_hi = min(s_hi, max(s1, s2))
    return float(s_lo), float(s_hi)


class Mixer:
    """Allocation matrix, its inverse and saturation-aware allocation."""

    def __init__(
        self,
        hubs_b: ArrayLike,
        spin: ArrayLike,
        k_m_over_k_f: float | ArrayLike,
        t_min: ArrayLike,
        t_max: ArrayLike,
    ) -> None:
        hubs = np.asarray(hubs_b, dtype=np.float64)
        s = np.asarray(spin, dtype=np.float64)
        ratio = np.broadcast_to(np.asarray(k_m_over_k_f, dtype=np.float64), s.shape)
        self.A: F64 = np.vstack([np.ones(len(s)), hubs[:, 1], -hubs[:, 0], -s * ratio])
        self.A_inv: F64 = np.linalg.inv(self.A)
        self.t_min = np.broadcast_to(np.asarray(t_min, dtype=np.float64), s.shape).copy()
        self.t_max = np.broadcast_to(np.asarray(t_max, dtype=np.float64), s.shape).copy()
        if np.any(self.t_min >= self.t_max):
            raise ValueError("need t_min < t_max for every motor")

    @classmethod
    def from_vehicle(cls, v: NominalVehicle) -> Mixer:
        lo, hi = v.thrust_limits_n
        return cls(v.hubs_b, v.spin, v.motor.k_m_over_k_f, lo, hi)

    # -- linear maps -----------------------------------------------------------------------------
    def wrench(self, thrusts: ArrayLike) -> F64:
        """``[T, tau_x, tau_y, tau_z] = A @ thrusts``."""
        return np.asarray(self.A @ np.asarray(thrusts, dtype=np.float64))

    def allocate_unsaturated(self, t_total: float, tau_b: ArrayLike) -> F64:
        """``A^-1 [T, tau]`` without any limit handling."""
        w = np.concatenate([[t_total], np.asarray(tau_b, dtype=np.float64)])
        return np.asarray(self.A_inv @ w)

    # -- saturation-aware allocation -------------------------------------------------------------
    def allocate(self, t_total: float, tau_b: ArrayLike) -> F64:
        """Per-motor thrusts [N] realizing ``(T_total, tau_b)`` as well as the limits permit."""
        return self.allocate_info(t_total, tau_b)[0]

    def allocate_info(self, t_total: float, tau_b: ArrayLike) -> tuple[F64, MixInfo]:
        tau = np.asarray(tau_b, dtype=np.float64)
        lo, hi = self.t_min, self.t_max
        full = self.allocate_unsaturated(t_total, tau)
        if np.all(full >= lo) and np.all(full <= hi):
            return full, MixInfo(False, 1.0, 1.0, 0.0)

        c_t, c_x, c_y, c_z = (self.A_inv[:, k] for k in range(4))
        rp = tau[0] * c_x + tau[1] * c_y
        yaw = tau[2] * c_z
        thrust_part = t_total * c_t

        def shift_for(gamma: float) -> tuple[float, F64] | None:
            base = thrust_part + gamma * rp
            d_lo, d_hi = _interval(base, c_t, lo, hi)
            if d_lo > d_hi:
                return None
            delta = float(np.clip(0.0, d_lo, d_hi))
            return delta, base + delta * c_t

        sol = shift_for(1.0)
        gamma = 1.0
        if sol is None:  # roll/pitch demand alone does not fit: bisect the largest feasible scale
            g_lo, g_hi = 0.0, 1.0  # gamma = 0 is always feasible (a uniform shift fits one motor)
            for _ in range(40):
                mid = 0.5 * (g_lo + g_hi)
                if shift_for(mid) is None:
                    g_hi = mid
                else:
                    g_lo = mid
            gamma = g_lo
            sol = shift_for(gamma)
        assert sol is not None
        delta, base = sol

        s_lo, s_hi = _interval(base, yaw, lo, hi)
        s_yaw = float(np.clip(1.0, max(s_lo, 0.0), min(s_hi, 1.0))) if s_lo <= s_hi else 0.0
        thrusts = np.clip(base + s_yaw * yaw, lo, hi)
        # c_t = A^-1 e_T realizes a pure total-thrust change, so delta is that change in newtons
        return thrusts, MixInfo(True, s_yaw, gamma, delta)
