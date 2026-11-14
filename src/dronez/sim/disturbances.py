"""External disturbances and the action-delay line.

``DisturbanceModel`` produces a world-frame ``(force, torque)`` pair applied at the centre of mass
(MuJoCo ``xfrc_applied``). It is a *pure function of time* after ``reset`` (all randomness is drawn
at reset from the ``np.random.Generator`` the env passes in), so the same schedule replays exactly.

* impulses: rectangular force/torque pulses (random schedule and/or an explicit list);
* wind: constant + gust field. Gusts are a sum of random-phase sinusoids (deterministic in ``t``).
  The wind force is the quadratic drag on the air-relative velocity ``0.5 rho CdA |u| u``,
  ``u = v_wind - v_body``.

``ActionDelay`` is a ring buffer measured in physics steps.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

from dronez.sim.state import TrueState

F64 = NDArray[np.float64]

AIR_DENSITY_KG_M3 = 1.225


class ImpulseSpec(BaseModel):
    """A scripted pulse: constant world-frame force/torque during ``[t_s, t_s + duration_s)``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    t_s: float = Field(ge=0.0)
    duration_s: float = Field(gt=0.0)
    force_w_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque_w_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)


class DisturbanceConfig(BaseModel):
    """Disturbance parameters. All zero (the default) means no disturbance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # scripted impulses (eval scenarios)
    impulses: tuple[ImpulseSpec, ...] = ()
    # random impulses: Poisson-like count over ``horizon_s``, uniform start times
    random_impulse_count: int = Field(default=0, ge=0)
    random_impulse_horizon_s: float = Field(default=10.0, gt=0.0)
    random_impulse_duration_s: float = Field(default=0.02, gt=0.0)
    random_impulse_force_n: float = Field(default=0.0, ge=0.0)  # magnitude, random direction
    random_impulse_torque_nm: float = Field(default=0.0, ge=0.0)  # magnitude, random direction
    # wind
    wind_mean_w_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wind_gust_std_m_s: float = Field(default=0.0, ge=0.0)  # per-axis RMS of the gust field
    wind_gust_freq_hz: tuple[float, float] = (0.2, 2.0)  # gust frequency band
    wind_n_components: int = Field(default=4, ge=1)
    wind_cda_m2: float = Field(
        default=0.01, ge=0.0
    )  # effective Cd * A of the airframe (placeholder)


class DisturbanceModel:
    """World-frame CoM force/torque as a function of time. Call ``reset`` before ``wrench``."""

    def __init__(self) -> None:
        self.cfg = DisturbanceConfig()
        self._pulses: list[tuple[float, float, F64, F64]] = []  # (t0, t1, force, torque)
        self._freq = np.zeros((3, 1))
        self._phase = np.zeros((3, 1))
        self._amp = np.zeros((3, 1))

    def reset(self, rng: np.random.Generator, cfg: DisturbanceConfig | None = None) -> None:
        """Draw the schedule for a new episode (all randomness happens here)."""
        self.cfg = cfg if cfg is not None else DisturbanceConfig()
        c = self.cfg
        pulses: list[tuple[float, float, F64, F64]] = []
        for p in c.impulses:
            pulses.append(
                (p.t_s, p.t_s + p.duration_s, np.array(p.force_w_n), np.array(p.torque_w_nm))
            )
        for _ in range(c.random_impulse_count):
            t0 = float(rng.uniform(0.0, c.random_impulse_horizon_s))
            f = c.random_impulse_force_n * _random_unit(rng)
            tau = c.random_impulse_torque_nm * _random_unit(rng)
            pulses.append((t0, t0 + c.random_impulse_duration_s, f, tau))
        self._pulses = pulses
        n = c.wind_n_components
        lo, hi = c.wind_gust_freq_hz
        self._freq = rng.uniform(lo, hi, size=(3, n))
        self._phase = rng.uniform(0.0, 2.0 * np.pi, size=(3, n))
        # sum of n sinusoids with amplitude a has RMS a*sqrt(n/2); pick a so the RMS = gust std
        self._amp = np.full((3, n), c.wind_gust_std_m_s * np.sqrt(2.0 / n))

    def wind_w(self, t: float) -> F64:
        """Wind velocity [m/s] in the world frame at time ``t``."""
        gust = np.sum(self._amp * np.sin(2.0 * np.pi * self._freq * t + self._phase), axis=1)
        return np.asarray(np.array(self.cfg.wind_mean_w_m_s) + gust)

    def wrench(self, t: float, true: TrueState) -> tuple[F64, F64]:
        """``(force_w [N], torque_w [N m])`` at time ``t`` for the given true state."""
        force = np.zeros(3)
        torque = np.zeros(3)
        for t0, t1, f, tau in self._pulses:
            if t0 <= t < t1:
                force += f
                torque += tau
        c = self.cfg
        if c.wind_cda_m2 > 0.0 and (any(c.wind_mean_w_m_s) or c.wind_gust_std_m_s > 0.0):
            u = self.wind_w(t) - np.asarray(true.linvel_w, dtype=np.float64)
            force += 0.5 * AIR_DENSITY_KG_M3 * c.wind_cda_m2 * np.linalg.norm(u) * u
        return force, torque


def _random_unit(rng: np.random.Generator) -> F64:
    v = rng.standard_normal(3)
    return np.asarray(v / max(float(np.linalg.norm(v)), 1e-12))


class ActionDelay:
    """Ring buffer delaying commands by ``n_steps`` physics steps.

    ``push(u)`` stores ``u`` and returns the command pushed ``n_steps`` calls ago. Until the buffer
    has filled, the *first* pushed command is returned (hold-first, so a reset episode does not
    start with a spurious zero command); call ``reset(fill)`` to prefill with an explicit value.
    ``n_steps = 0`` is a pass-through.
    """

    def __init__(self, n_steps: int) -> None:
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        self.n_steps = int(n_steps)
        self._buf: deque[F64] = deque()

    def reset(self, fill: ArrayLike | None = None) -> None:
        """Empty the line; with ``fill`` set, prefill it so the first outputs equal ``fill``."""
        self._buf.clear()
        if fill is not None:
            f = np.array(fill, dtype=np.float64)
            for _ in range(self.n_steps):
                self._buf.append(f.copy())

    def set_steps(self, n_steps: int) -> None:
        """Change the delay (used by domain randomization); empties the line."""
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        self.n_steps = int(n_steps)
        self._buf.clear()

    def push(self, u: ArrayLike) -> F64:
        x = np.array(u, dtype=np.float64)  # copy
        if self.n_steps == 0:
            return x
        if not self._buf:
            for _ in range(self.n_steps):
                self._buf.append(x.copy())
        self._buf.append(x)
        return self._buf.popleft()
