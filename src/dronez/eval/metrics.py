"""Pure episode metrics over trajectories (numpy only, SI units: rad, s, N m).

Degrees appear only in ``report``. Conventions:

* ``att_err`` is the geodesic angle [rad] between the attitude and its target at every sample.
* ``u`` is the action actually applied, ``(n_steps, n_motors)``; for baselines this is the
  normalized motor command in ``[0, 1]`` (Phase 4 may change the scale; the metrics are
  scale-aware only through ``sat_lo`` / ``sat_hi``).
* ``dt`` is the policy period [s]; samples are uniformly spaced.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dronez.utils import rotations

F64 = NDArray[np.float64]

RECOVERY_THRESHOLD_RAD = float(np.deg2rad(5.0))
HF_CUTOFF_HZ = 20.0


def attitude_error(quat: ArrayLike, quat_target: ArrayLike) -> F64:
    """Geodesic angle [rad] per sample (``quat`` and ``quat_target`` broadcast, ``(..., 4)``)."""
    return np.asarray(rotations.geodesic_angle(quat, quat_target), dtype=np.float64)


def time_to_recover(
    att_err: ArrayLike,
    dt: float,
    threshold: float = RECOVERY_THRESHOLD_RAD,
    t_start: float = 0.0,
) -> float:
    """Time [s] after ``t_start`` from which the error stays below ``threshold`` to the end.

    ``nan`` if the error is above the threshold at the last sample (never settled). Sample ``k`` is
    at ``t = k * dt``.
    """
    e = np.asarray(att_err, dtype=np.float64)
    k0 = round(t_start / dt)
    tail = e[k0:]
    if tail.size == 0 or tail[-1] >= threshold:
        return float("nan")
    above = np.nonzero(tail >= threshold)[0]
    first_ok = 0 if above.size == 0 else int(above[-1]) + 1
    return float(first_ok * dt)


def rms_attitude_error(att_err: ArrayLike, dt: float, t_start: float = 0.0) -> float:
    """Root-mean-square of the geodesic error [rad] from ``t_start`` on."""
    e = np.asarray(att_err, dtype=np.float64)[round(t_start / dt) :]
    return float(np.sqrt(np.mean(e**2))) if e.size else float("nan")


def overshoot(quat: ArrayLike, quat_start: ArrayLike, quat_target: ArrayLike) -> float:
    """Overshoot [rad]: how far the attitude travels past the target along its way from the start.

    ``max_t angle(q_t, q_start) - angle(q_target, q_start)``, clipped at 0. Exact for a rotation
    about a fixed axis (the step and recovery scenarios); a pure-distance measure otherwise.
    """
    q = np.asarray(quat, dtype=np.float64)
    travelled = np.asarray(rotations.geodesic_angle(q, np.asarray(quat_start)))
    full = float(rotations.geodesic_angle(quat_target, quat_start))
    return float(max(np.max(travelled) - full, 0.0))


def saturation_pct(
    u: ArrayLike, sat_lo: float = 0.0, sat_hi: float = 1.0, tol: float = 1e-3
) -> float:
    """Percent of (step, motor) command samples at either limit (within ``tol``)."""
    a = np.asarray(u, dtype=np.float64)
    if a.size == 0:
        return 0.0
    hit = (a <= sat_lo + tol) | (a >= sat_hi - tol)
    return float(100.0 * np.mean(hit))


def mean_abs_delta(u: ArrayLike) -> float:
    """Mean ``|u_k - u_{k-1}|`` over steps and motors (action smoothness)."""
    a = np.asarray(u, dtype=np.float64)
    if a.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(a, axis=0))))


def hf_energy_fraction(u: ArrayLike, rate_hz: float, f_cut_hz: float = HF_CUTOFF_HZ) -> float:
    """Fraction of the action spectrum energy above ``f_cut_hz`` (DC removed, summed over motors).

    FFT at the policy rate, so only frequencies up to ``rate_hz / 2`` exist. 0 for a constant
    action.
    """
    a = np.asarray(u, dtype=np.float64)
    if a.shape[0] < 4:
        return 0.0
    a = a - np.mean(a, axis=0, keepdims=True)
    power = np.sum(np.abs(np.fft.rfft(a, axis=0)) ** 2, axis=1)
    freqs = np.fft.rfftfreq(a.shape[0], d=1.0 / rate_hz)
    total = float(np.sum(power))
    if total <= 1e-18:  # constant action (round-off only)
        return 0.0
    return float(np.sum(power[freqs > f_cut_hz]) / total)


def success(
    att_err: ArrayLike,
    dt: float,
    hold_s: float = 1.0,
    threshold: float = RECOVERY_THRESHOLD_RAD,
    crashed: bool = False,
) -> bool:
    """True if the error stays below ``threshold`` for the last ``hold_s`` seconds, no crash."""
    if crashed:
        return False
    e = np.asarray(att_err, dtype=np.float64)
    n = max(round(hold_s / dt), 1)
    return bool(e.size >= n and np.all(e[-n:] < threshold))


def position_error(pos: ArrayLike, pos_target: ArrayLike) -> F64:
    """Euclidean position error [m] per sample."""
    return np.asarray(
        np.linalg.norm(np.asarray(pos) - np.asarray(pos_target), axis=-1), dtype=np.float64
    )
