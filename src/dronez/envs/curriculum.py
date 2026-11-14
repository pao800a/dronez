"""Curriculum levels (ADR-0006 §6) and the promotion rule.

One integer level ``k`` per env, set with ``VecEnv.env_method("set_curriculum_level", k)``; it takes
effect at each env's next reset. ``LevelSpec`` describes the init distribution and the DR width of a
level; ``Curriculum`` is the pure-Python promotion logic driven by deterministic mini-evals
(promote on ``consecutive`` successive evals at or above the threshold and >= ``min_steps`` spent
at the level; never demote).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class LevelSpec(BaseModel):
    """Init distribution, target mix, DR width and promotion rule of one level."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    #: initial attitude error cap (cos-uniform), omega0 / v0 / a0 magnitudes
    delta_max_deg: float = 15.0
    omega_max_rad_s: float = 0.5
    v0_max_m_s: float = 0.0
    a0_max: float = 0.0
    #: Attitude target mix: probability of a tilted target / of a mid-episode target step
    p_tilted: float = 0.0
    p_step: float = 0.0
    #: Hover: initial offset ball radius [m] and heading deviation from the target heading [deg]
    offset_m: float = 0.0
    heading_dev_deg: float = 0.0
    #: DR width d in [0, 1]; wind and impulses are active only when ``disturbances`` is set
    dr_scale: float = Field(default=0.0, ge=0.0, le=1.0)
    disturbances: bool = False
    #: promotion: eval success rate threshold (None = last level of a stage), step budget
    promote_sr: float | None = None
    min_steps: int = 0
    max_steps: int | None = None


def _att(  # noqa: PLR0917 - one positional row per level keeps the table readable
    name: str,
    delta: float,
    omega: float,
    v0: float,
    a0: float,
    mix: float,
    d: float,
    dist: bool,
    sr: float | None,
    lo: int,
    hi: int | None,
) -> LevelSpec:
    return LevelSpec(
        name=name,
        delta_max_deg=delta,
        omega_max_rad_s=omega,
        v0_max_m_s=v0,
        a0_max=a0,
        p_tilted=mix,
        p_step=mix,
        dr_scale=d,
        disturbances=dist,
        promote_sr=sr,
        min_steps=lo,
        max_steps=hi,
    )


M = 1_000_000
#: ADR-0006 §6 attitude levels. Level 3 ends ``att_p5``; levels 4-6 belong to ``att_p6_dr``.
ATTITUDE_LEVELS: tuple[LevelSpec, ...] = (
    _att("att_p5_0", 15.0, 0.5, 0.0, 0.0, 0.0, 0.00, False, 0.90, 1 * M, 6 * M),
    _att("att_p5_1", 45.0, 1.5, 0.0, 0.2, 0.2, 0.10, False, 0.90, 2 * M, 8 * M),
    _att("att_p5_2", 90.0, 3.0, 1.0, 0.2, 0.2, 0.25, False, 0.90, 2 * M, 12 * M),
    _att("att_p5_3", 180.0, 5.0, 1.0, 0.2, 0.2, 0.25, False, None, 0, None),
    _att("att_p6_4", 180.0, 5.0, 1.0, 0.2, 0.2, 0.50, True, 0.85, 3 * M, 12 * M),
    _att("att_p6_5", 180.0, 5.0, 1.0, 0.2, 0.2, 0.75, True, 0.85, 3 * M, 12 * M),
    _att("att_p6_6", 180.0, 5.0, 1.0, 0.2, 0.2, 1.00, True, None, 0, None),
)


def _hov(  # noqa: PLR0917 - one positional row per level keeps the table readable
    name: str,
    offset: float,
    v0: float,
    delta: float,
    omega: float,
    head: float,
    d: float,
    sr: float | None,
) -> LevelSpec:
    return LevelSpec(
        name=name,
        delta_max_deg=delta,
        omega_max_rad_s=omega,
        v0_max_m_s=v0,
        a0_max=0.0 if d == 0.0 else 0.2,
        offset_m=offset,
        heading_dev_deg=head,
        dr_scale=d,
        disturbances=d >= 0.5,
        promote_sr=sr,
        min_steps=2 * M if sr is not None else 0,
        max_steps=10 * M if sr is not None else None,
    )


#: ADR-0006 §6 hover levels (Phase 7).
HOVER_LEVELS: tuple[LevelSpec, ...] = (
    _hov("hover_h0", 0.2, 0.2, 10.0, 0.5, 15.0, 0.00, 0.9),
    _hov("hover_h1", 0.5, 0.5, 30.0, 1.5, 45.0, 0.25, 0.9),
    _hov("hover_h2", 1.0, 1.0, 60.0, 3.0, 180.0, 0.50, 0.9),
    _hov("hover_h3", 1.0, 1.0, 60.0, 3.0, 180.0, 1.00, None),
)


class Curriculum:
    """Promotion logic: ``update`` after every deterministic mini-eval, never demotes.

    ``update(success_rate, timesteps)`` returns ``True`` when the level was promoted. ``timesteps``
    is the total number of training steps so far; the steps spent at the level are counted from the
    step at which it was entered.
    """

    def __init__(
        self, levels: Sequence[LevelSpec], start_level: int = 0, consecutive: int = 2
    ) -> None:
        if not 0 <= start_level < len(levels):
            raise ValueError(f"start_level {start_level} outside [0, {len(levels) - 1}]")
        self.levels = tuple(levels)
        self.level = int(start_level)
        self.consecutive = consecutive
        self.n_ok = 0
        self.level_start_steps = 0
        self.stalled = False

    @property
    def spec(self) -> LevelSpec:
        return self.levels[self.level]

    def steps_at_level(self, timesteps: int) -> int:
        return max(timesteps - self.level_start_steps, 0)

    def update(self, success_rate: float, timesteps: int) -> bool:
        spec = self.spec
        if spec.promote_sr is None or self.level + 1 >= len(self.levels):
            return False
        self.n_ok = self.n_ok + 1 if success_rate >= spec.promote_sr else 0
        spent = self.steps_at_level(timesteps)
        if self.n_ok >= self.consecutive and spent >= spec.min_steps:
            self.level += 1
            self.n_ok = 0
            self.level_start_steps = int(timesteps)
            self.stalled = False
            return True
        self.stalled = spec.max_steps is not None and spent > spec.max_steps
        return False

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "level": self.level,
            "n_ok": self.n_ok,
            "level_start_steps": self.level_start_steps,
            "stalled": self.stalled,
        }

    def load_state_dict(self, state: dict[str, int | bool]) -> None:
        self.level = int(state["level"])
        self.n_ok = int(state.get("n_ok", 0))
        self.level_start_steps = int(state.get("level_start_steps", 0))
        self.stalled = bool(state.get("stalled", False))
