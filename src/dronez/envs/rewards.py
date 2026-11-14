"""Reward (ADR-0006 §4), computed from the **true** state, logged per term.

``reward_terms`` holds the *weighted* contributions and sums to ``r`` exactly. Penalty terms are
summed and clipped at ``-1`` (``penalty_clip`` adds the difference back), so every non-terminal
step has ``r >= 0`` and the ``crash`` term makes termination strictly worse than surviving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
Kind = Literal["attitude", "hover"]

ATTITUDE_TERMS: tuple[str, ...] = (
    "alive",
    "att",
    "att_fine",
    "rate",
    "vz",
    "act_mag",
    "act_smooth",
    "act_sat",
    "penalty_clip",
    "crash",
)
HOVER_TERMS: tuple[str, ...] = (
    "alive",
    "att",
    "att_fine",
    "rate",
    "pos",
    "pos_fine",
    "vel",
    "act_mag",
    "act_smooth",
    "act_sat",
    "penalty_clip",
    "crash",
)
#: Terms that are penalties (<= 0) and enter the clipped sum ``P``.
PENALTY_TERMS_ATT = ("att", "rate", "vz", "act_mag", "act_smooth", "act_sat")
PENALTY_TERMS_HOVER = ("att", "rate", "pos", "vel", "act_mag", "act_smooth", "act_sat")


def term_names(kind: Kind) -> tuple[str, ...]:
    return ATTITUDE_TERMS if kind == "attitude" else HOVER_TERMS


@dataclass(frozen=True)
class RewardWeights:
    """Weights and scales. Defaults are for the Attitude env; ``for_kind`` gives the Hover set."""

    alive: float = 1.0
    att: float = 0.5
    att_fine: float = 0.5
    sigma_e_rad: float = 0.10
    rate: float = 0.1
    rate_scale_rad_s: float = 5.0
    vz: float = 0.1
    vz_scale_m_s: float = 1.0
    pos: float = 0.0
    pos_scale_m: float = 2.0
    pos_fine: float = 0.0
    pos_sigma_m: float = 0.10
    vel: float = 0.0
    vel_scale_m_s: float = 1.0
    act_mag: float = 0.05
    act_smooth: float = 0.25
    act_sat: float = 0.1
    sat_threshold: float = 0.99
    crash: float = 10.0
    penalty_floor: float = -1.0

    @classmethod
    def for_kind(cls, kind: Kind, overrides: dict[str, float] | None = None) -> RewardWeights:
        base: dict[str, float] = {}
        if kind == "hover":
            base = {
                "att": 0.2,
                "att_fine": 0.2,
                "sigma_e_rad": 0.2,
                "vz": 0.0,
                "pos": 0.5,
                "pos_fine": 0.5,
                "vel": 0.1,
            }
        valid = {f.name for f in fields(cls)}
        for k in overrides or {}:
            if k not in valid:
                raise ValueError(f"unknown reward weight {k!r}; valid: {sorted(valid)}")
        return cls(**{**base, **(overrides or {})})

    def max_step_reward(self, kind: Kind) -> float:
        if kind == "attitude":
            return self.alive + self.att_fine
        return self.alive + self.att_fine + self.pos_fine


def compute_reward(
    kind: Kind,
    w: RewardWeights,
    *,
    att_err: float,
    angvel_norm: float,
    vz: float,
    vel_norm: float,
    pos_err: float,
    a: F64,
    a_prev: F64,
    terminated: bool,
) -> tuple[float, dict[str, float]]:
    """``(r, terms)``; every term is a weighted contribution and ``sum(terms) == r``."""
    terms: dict[str, float] = {"alive": w.alive}
    terms["att"] = -w.att * att_err / math.pi
    terms["att_fine"] = w.att_fine * math.exp(-((att_err / w.sigma_e_rad) ** 2))
    terms["rate"] = -w.rate * min((angvel_norm / w.rate_scale_rad_s) ** 2, 1.0)
    if kind == "attitude":
        terms["vz"] = -w.vz * min((vz / w.vz_scale_m_s) ** 2, 1.0)
    else:
        terms["pos"] = -w.pos * min(pos_err / w.pos_scale_m, 1.0)
        terms["pos_fine"] = w.pos_fine * math.exp(-((pos_err / w.pos_sigma_m) ** 2))
        terms["vel"] = -w.vel * min((vel_norm / w.vel_scale_m_s) ** 2, 1.0)
    terms["act_mag"] = -w.act_mag * float(np.mean(a * a))
    d = a - a_prev
    terms["act_smooth"] = -w.act_smooth * float(np.mean(d * d))
    terms["act_sat"] = -w.act_sat * float(np.mean(np.abs(a) >= w.sat_threshold))
    penalties = PENALTY_TERMS_ATT if kind == "attitude" else PENALTY_TERMS_HOVER
    p_sum = sum(terms[k] for k in penalties)
    terms["penalty_clip"] = max(p_sum, w.penalty_floor) - p_sum
    terms["crash"] = -w.crash if terminated else 0.0
    # fixed key order (and a fixed summation order, so r is reproducible bit for bit)
    ordered = {k: terms[k] for k in term_names(kind)}
    return float(sum(ordered.values())), ordered
