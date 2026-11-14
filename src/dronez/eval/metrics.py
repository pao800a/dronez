"""Pure episode metrics over trajectories (numpy only, SI units: rad, s, N m).

Degrees appear only in ``report``. Conventions:

* ``att_err`` is the geodesic angle [rad] between the attitude and its target at every sample.
* ``u`` is the normalized motor command ``cmap(a)`` of the env, ``(n_steps, n_motors)`` (ADR-0006
  §3); the policy action ``a`` lives in ``[-1, 1]``. The smoothness metrics (``hf_rms``,
  ``mean_abs_delta``, ``hf_energy_fraction``) are computed on ``u``; saturation on ``a``.
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
    """Percent of (step, motor) samples at either limit (within ``tol``).

    For the env action ``a`` use ``sat_lo=-1, sat_hi=1`` (the envelope edges).
    """
    a = np.asarray(u, dtype=np.float64)
    if a.size == 0:
        return 0.0
    hit = (a <= sat_lo + tol) | (a >= sat_hi - tol)
    return float(100.0 * np.mean(hit))


def mean_abs_delta(u: ArrayLike, rate_hz: float | None = None, t_start: float = 0.0) -> float:
    """Mean ``|u_k - u_{k-1}|`` over steps and motors (action smoothness), from ``t_start`` on.

    ``rate_hz`` is needed only to convert ``t_start`` into samples (ADR-0006 §9 uses 1.0 s).
    """
    a = np.asarray(u, dtype=np.float64)
    if rate_hz is not None:
        a = a[round(t_start * rate_hz) :]
    if a.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(a, axis=0))))


def _detrend(a: F64) -> F64:
    """Remove the least-squares line over time from every column."""
    n = a.shape[0]
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    slope = (t @ (a - a.mean(axis=0))) / max(float(t @ t), 1e-30)
    return np.asarray(a - a.mean(axis=0) - np.outer(t, slope))


def hf_rms(
    u: ArrayLike,
    rate_hz: float,
    f_cut: float = HF_CUTOFF_HZ,
    t_start: float = 1.0,
) -> float:
    """RMS of the command content above ``f_cut`` Hz (ADR-0006 §9), in units of ``u``.

    Per motor: take ``u`` from ``t_start`` on, remove a linear detrend, ``rfft``, zero the bins
    with ``f <= f_cut``, ``irfft``; the RMS is taken over time and motors. 0 for a constant or a
    ramp; ``A / sqrt(2)`` for a sinusoid of amplitude ``A`` above the cutoff.
    """
    a = np.asarray(u, dtype=np.float64)[round(t_start * rate_hz) :]
    if a.shape[0] < 4:
        return 0.0
    spec = np.fft.rfft(_detrend(a), axis=0)
    freqs = np.fft.rfftfreq(a.shape[0], d=1.0 / rate_hz)
    spec[freqs <= f_cut, :] = 0.0
    hf = np.fft.irfft(spec, n=a.shape[0], axis=0)
    return float(np.sqrt(np.mean(hf**2)))


#: below this detrended AC RMS the spectral energy *fraction* is meaningless (noise-driven)
HF_FRACTION_AC_FLOOR = 1e-3


def hf_energy_fraction(u: ArrayLike, rate_hz: float, f_cut_hz: float = HF_CUTOFF_HZ) -> float:
    """Fraction of the action spectrum energy above ``f_cut_hz`` (DC removed, summed over motors).

    FFT at the policy rate, so only frequencies up to ``rate_hz / 2`` exist. **Report-only**
    (ADR-0006 §9): ``nan`` when the detrended AC RMS is below ``HF_FRACTION_AC_FLOOR`` (a constant
    or noise-driven tiny action), where a scale-free fraction is meaningless.
    """
    a = np.asarray(u, dtype=np.float64)
    if a.shape[0] < 4:
        return float("nan")
    if float(np.sqrt(np.mean(_detrend(a) ** 2))) < HF_FRACTION_AC_FLOOR:
        return float("nan")
    a = a - np.mean(a, axis=0, keepdims=True)
    power = np.sum(np.abs(np.fft.rfft(a, axis=0)) ** 2, axis=1)
    freqs = np.fft.rfftfreq(a.shape[0], d=1.0 / rate_hz)
    total = float(np.sum(power))
    if total <= 1e-18:
        return float("nan")
    return float(np.sum(power[freqs > f_cut_hz]) / total)


def success_recovery(att_err: ArrayLike, dt: float, t_max: float, crashed: bool) -> bool:
    """``not crashed and time_to_recover(att_err, dt) <= t_max`` (ADR-0006 §9).

    ``time_to_recover`` already requires the error to stay below 5 deg until the end, so this is
    "recovered within ``t_max`` and stayed recovered".
    """
    if crashed:
        return False
    t = time_to_recover(att_err, dt)
    return bool(np.isfinite(t) and t <= t_max + 1e-9)


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
