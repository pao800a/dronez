"""Observation layout (ADR-0006 §2): the single source for ``policy_meta.json`` and the frozen
normalization (``dronez.rl.normalization``).

Observations are raw SI float32, built **only from ``Sensed``**, the target and the action history,
never from the true state. ``offset`` / ``scale`` per block define the frozen normalization
``(x - offset) / scale`` applied by VecNormalize with fixed statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

HISTORY_K = 4
N_MOTORS = 4


@dataclass(frozen=True)
class ObsBlock:
    """A named, contiguous slice of the observation."""

    name: str
    size: int
    units: str
    frame: str
    offset: float = 0.0
    scale: float = 1.0
    #: symmetric bound of the raw observation space (finite, so ``check_env`` is quiet)
    bound: float = 1000.0

    def to_meta(self, start: int) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": start,
            "end": start + self.size,
            "units": self.units,
            "frame": self.frame,
            "offset": self.offset,
            "scale": self.scale,
            "bound": self.bound,
        }


_ATTITUDE_BLOCKS: tuple[ObsBlock, ...] = (
    ObsBlock("r_err_6d", 6, "1", "target<-body", 0.0, 1.0, 1.001),
    ObsBlock("gravity_b", 3, "1", "body FLU", 0.0, 1.0, 1.001),
    ObsBlock("gyro_b", 3, "rad/s", "body FLU", 0.0, 5.0),
    ObsBlock("vel_z_w", 1, "m/s", "world z (up)", 0.0, 2.0),
    ObsBlock("action_hist", HISTORY_K * N_MOTORS, "1", "motor index", 0.0, 1.0, 1.001),
)
_HOVER_BLOCKS: tuple[ObsBlock, ...] = (
    ObsBlock("pos_err_h", 3, "m", "target heading frame", 0.0, 1.0),
    ObsBlock("vel_xy_h", 2, "m/s", "target heading frame", 0.0, 2.0),
)
_PRIVILEGED_BLOCKS: tuple[ObsBlock, ...] = (
    ObsBlock("r_err_6d", 6, "1", "target<-body"),
    ObsBlock("gravity_b", 3, "1", "body FLU"),
    ObsBlock("angvel_b", 3, "rad/s", "body FLU"),
    ObsBlock("linvel_w", 3, "m/s", "world"),
    ObsBlock("pos_err_h", 3, "m", "target heading frame"),
    ObsBlock("omega_norm", 4, "1", "motor index"),
    ObsBlock("mass_scale", 1, "1", "-"),
    ObsBlock("inertia_scale", 3, "1", "body principal"),
    ObsBlock("com_offset_b_m", 3, "m", "body FLU"),
    ObsBlock("k_f_scale", 4, "1", "motor index"),
    ObsBlock("k_m_scale", 1, "1", "-"),
    ObsBlock("tau_m_s", 1, "s", "-"),
    ObsBlock("latency_s", 1, "s", "-"),
    ObsBlock("battery_scale", 1, "1", "-"),
    ObsBlock("force_w_n", 3, "N", "world"),
    ObsBlock("torque_w_nm", 3, "N m", "world"),
    ObsBlock("wind_w_m_s", 3, "m/s", "world"),
)


def _starts(blocks: tuple[ObsBlock, ...]) -> list[int]:
    out, k = [], 0
    for b in blocks:
        out.append(k)
        k += b.size
    return out


@dataclass(frozen=True)
class ObsLayout:
    """Ordered blocks. ``dim``, ``slices``, ``offset`` and ``scale`` are derived."""

    blocks: tuple[ObsBlock, ...]
    privileged: tuple[ObsBlock, ...] = _PRIVILEGED_BLOCKS
    history_k: int = HISTORY_K

    # -- constructors -----------------------------------------------------------------------
    @classmethod
    def attitude(cls) -> ObsLayout:
        return cls(_ATTITUDE_BLOCKS)

    @classmethod
    def hover(cls) -> ObsLayout:
        return cls(_ATTITUDE_BLOCKS + _HOVER_BLOCKS)

    # -- derived ----------------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return sum(b.size for b in self.blocks)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.blocks)

    @property
    def units(self) -> dict[str, str]:
        return {b.name: b.units for b in self.blocks}

    @property
    def frames(self) -> dict[str, str]:
        return {b.name: b.frame for b in self.blocks}

    @property
    def slices(self) -> dict[str, slice]:
        out = {}
        for b, s in zip(self.blocks, _starts(self.blocks), strict=True):
            out[b.name] = slice(s, s + b.size)
        return out

    @property
    def offset(self) -> F64:
        return np.concatenate([np.full(b.size, b.offset) for b in self.blocks])

    @property
    def bound(self) -> F64:
        return np.concatenate([np.full(b.size, b.bound) for b in self.blocks])

    @property
    def scale(self) -> F64:
        return np.concatenate([np.full(b.size, b.scale) for b in self.blocks])

    @property
    def privileged_dim(self) -> int:
        return sum(b.size for b in self.privileged)

    @property
    def privileged_slices(self) -> dict[str, slice]:
        out = {}
        for b, s in zip(self.privileged, _starts(self.privileged), strict=True):
            out[b.name] = slice(s, s + b.size)
        return out

    # -- (de)serialization ------------------------------------------------------------------
    def to_meta(self) -> dict[str, Any]:
        """``policy_meta.obs`` content (``normalization`` mean/var is added by the exporter)."""
        return {
            "dim": self.dim,
            "history_k": self.history_k,
            "layout": [
                b.to_meta(s) for b, s in zip(self.blocks, _starts(self.blocks), strict=True)
            ],
            "privileged": {
                "dim": self.privileged_dim,
                "layout": [
                    b.to_meta(s)
                    for b, s in zip(self.privileged, _starts(self.privileged), strict=True)
                ],
            },
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> ObsLayout:
        def blocks(rows: list[dict[str, Any]]) -> tuple[ObsBlock, ...]:
            return tuple(
                ObsBlock(
                    r["name"],
                    int(r["end"]) - int(r["start"]),
                    r["units"],
                    r["frame"],
                    float(r.get("offset", 0.0)),
                    float(r.get("scale", 1.0)),
                    float(r.get("bound", 1000.0)),
                )
                for r in rows
            )

        priv = meta.get("privileged", {}).get("layout")
        return cls(
            blocks(meta["layout"]),
            blocks(priv) if priv is not None else _PRIVILEGED_BLOCKS,
            int(meta.get("history_k", HISTORY_K)),
        )

    def normalize(self, obs: F64 | F32, clip: float = 10.0) -> F32:
        """Frozen normalization ``clip((x - offset) / scale, +-clip)`` (float32)."""
        x = (np.asarray(obs, dtype=np.float64) - self.offset) / self.scale
        return np.asarray(np.clip(x, -clip, clip), dtype=np.float32)
