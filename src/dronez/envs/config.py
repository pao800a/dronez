"""``EnvConfig``: everything an env needs beyond the drone and sim configs (ADR-0006).

Frozen pydantic model, YAML-friendly; ``hash()`` identifies it in the eval baseline cache key.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dronez.envs.curriculum import LevelSpec
from dronez.envs.randomization import DRConfig
from dronez.sim.sensors import ImuNoiseConfig
from dronez.utils.config import stable_hash


class EnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    #: ``motors`` is gated; CTBR is specified in ADR-0006 §3 but not implemented in Phase 4
    action_mode: Literal["motors"] = "motors"
    #: command-map authority (ADR-0006 §3); 1.0 = the full symmetric envelope
    authority: float = Field(default=1.0, gt=0.0, le=1.0)
    #: episode length [s]; ``None`` = the env default (Attitude 5 s, Hover 8 s)
    episode_len_s: float | None = Field(default=None, gt=0.0)
    #: ``minimal`` keeps only the keys training needs (SPS); ``full`` adds ``sensed``/``true``/...
    info_level: Literal["full", "minimal"] = "full"
    #: emit ``info["privileged_obs"]`` (D = 46) for an optional asymmetric critic
    privileged_obs: bool = False
    #: overrides of ``RewardWeights`` fields (e.g. ``{act_smooth: 0.5}``)
    reward: dict[str, float] = Field(default_factory=dict)
    #: initial curriculum level (``set_curriculum_level`` changes it at runtime)
    start_level: int = Field(default=0, ge=0)
    #: replace the default level table of the env (ADR-0006 §6)
    levels: tuple[LevelSpec, ...] | None = None
    #: probability of drawing the task from a uniformly chosen lower level (k >= 1)
    p_lower_level: float = Field(default=0.2, ge=0.0, le=1.0)
    #: nominal sensor noise (scaled per episode by DR; ``ImuNoiseConfig.ideal()`` for none)
    noise: ImuNoiseConfig = Field(default_factory=ImuNoiseConfig)
    dr: DRConfig = Field(default_factory=DRConfig)

    # -- termination (ADR-0006 §5) ------------------------------------------------------------
    rate_limit_rad_s: float = 25.0
    ground_z_m: float = 0.05
    oob_xy_m: float = 20.0  # Attitude: horizontal distance from the start
    oob_z_m: float = 10.0  # Attitude: height above the start
    oob_hover_m: float = 3.0  # Hover: distance from the target

    # -- start state / success ---------------------------------------------------------------
    z0_range_m: tuple[float, float] = (3.0, 5.0)  # Attitude start height
    hover_z_range_m: tuple[float, float] = (1.5, 3.0)  # Hover target height
    success_window_s: float = 1.0
    success_att_deg: float = 5.0
    success_pos_m: float = 0.10

    def hash(self) -> str:
        return stable_hash(self)
