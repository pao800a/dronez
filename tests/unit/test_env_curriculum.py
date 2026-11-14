"""T10 (ADR-0006 §6): level bounds through ``VecEnv.env_method`` and the promotion logic."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest

from dronez.utils import rotations as rot

pytestmark = pytest.mark.fast

M = 1_000_000

# ADR-0006 §6 attitude table: level -> (delta_max deg, |omega0| max rad/s, |v0| max m/s, |a0| max)
ATT_LEVELS = {
    0: (15.0, 0.5, 0.0, 0.0),
    1: (45.0, 1.5, 0.0, 0.2),
    2: (90.0, 3.0, 1.0, 0.2),
    3: (180.0, 5.0, 1.0, 0.2),
    4: (180.0, 5.0, 1.0, 0.2),
    5: (180.0, 5.0, 1.0, 0.2),
    6: (180.0, 5.0, 1.0, 0.2),
}
# promotion rule: level -> (threshold, min_steps, max_steps); None = no promotion
ATT_PROMOTION = {0: (0.90, 1 * M, 6 * M), 1: (0.90, 2 * M, 8 * M), 2: (0.90, 2 * M, 12 * M)}
N_RESETS = 60


def _reset_all(venv: Any, seed0: int) -> list[tuple[np.ndarray, dict[str, Any]]]:
    return venv.env_method("reset", seed=seed0)  # [(obs, info)] per sub-env


@pytest.mark.parametrize("level", sorted(ATT_LEVELS))
def test_level_bounds_the_initial_error_rates_velocity_and_history(
    api: ModuleType, model_cache: None, level: int
) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([api.make_fn("DronezAttitude-v0") for _ in range(2)])
    try:
        api.set_level(venv, level)
        d_max, w_max, v_max, a_max = ATT_LEVELS[level]
        deltas, ws, vs, a0s = [], [], [], []
        for i in range(N_RESETS // 2):
            for obs, info in _reset_all(venv, 1000 * level + i):
                tq = np.asarray(info["true"]["quat_wxyz"])
                deltas.append(float(rot.geodesic_angle(tq, info["target"]["quat_wxyz"])))
                w = float(np.linalg.norm(info["true"]["angvel_b"]))
                ws.append(w)
                # CoM velocity = origin velocity + w x r_com (|r_com| ~ 25 mm, plus DR offsets)
                vs.append((float(np.linalg.norm(info["true"]["linvel_w"])), w))
                a0s.append(float(np.max(np.abs(obs[13:17]))))
                np.testing.assert_allclose(obs[13:17], obs[17:21], atol=1e-7)  # history prefilled
        tol = 1e-6
        assert max(deltas) <= np.deg2rad(d_max) + tol
        assert max(ws) <= w_max + tol
        assert all(v <= v_max + 0.05 * w + 0.01 + tol for v, w in vs)
        assert max(a0s) <= a_max + tol
        # the range is actually used (the p = 0.2 lower-level mix cannot hide it in 60 draws)
        assert max(deltas) >= 0.6 * np.deg2rad(d_max)
        assert max(ws) >= 0.6 * w_max
        if v_max > 0:
            assert max(v for v, _ in vs) >= 0.5 * v_max
        if a_max > 0:
            assert max(a0s) >= 0.5 * a_max
    finally:
        venv.close()


def test_level_applies_at_the_next_reset_and_never_mid_episode(api: ModuleType, mk: Any) -> None:
    env = mk("DronezAttitude-v0")
    nominal_mass = env.unwrapped.nominal.mass_kg
    _, info = env.reset(seed=3)
    assert info["curriculum_level"] == 0
    env.unwrapped.set_curriculum_level(4)  # DR width 0.5
    for _ in range(3):
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert info["params"]["mass_kg"] == nominal_mass, "the plant changed mid-episode"
    masses = []
    for seed in range(10):
        _, info = env.reset(seed=seed)
        assert info["curriculum_level"] == 4
        masses.append(info["params"]["mass_kg"])
    assert any(m != nominal_mass for m in masses)


def test_invalid_levels_are_rejected(mk: Any) -> None:
    env = mk("DronezAttitude-v0")
    with pytest.raises(ValueError, match="level"):
        env.unwrapped.set_curriculum_level(99)
    with pytest.raises(ValueError, match="level"):
        env.unwrapped.set_curriculum_level(-1)


def test_dr_width_follows_the_level(api: ModuleType, mk: Any) -> None:
    """d = 0 at level 0, so the plant is the nominal one; at level 4 (d = 0.5) it is perturbed."""
    env = mk("DronezAttitude-v0")
    nominal_mass = env.unwrapped.nominal.mass_kg
    for seed in range(5):
        _, info = env.reset(seed=seed)
        assert info["params"]["mass_kg"] == nominal_mass
    env.unwrapped.set_curriculum_level(4)
    masses = [env.reset(seed=s)[1]["params"]["mass_kg"] for s in range(20)]
    assert max(abs(m / nominal_mass - 1.0) for m in masses) > 0.02
    assert max(abs(m / nominal_mass - 1.0) for m in masses) <= 0.5 * 0.2 + 1e-9  # d * h = 10 %


# --- promotion logic on synthetic eval streams -------------------------------------------------
@pytest.fixture
def cur(api: ModuleType) -> Any:
    return api.Curriculum(api.ATTITUDE_LEVELS)


def test_promotes_on_two_consecutive_successes_after_min_steps(cur: Any) -> None:
    assert cur.update(0.95, 1 * M) is False  # first success: need two in a row
    assert cur.level == 0
    assert cur.update(0.95, 2 * M) is True
    assert cur.level == 1


def test_threshold_is_inclusive_and_a_miss_resets_the_streak(cur: Any) -> None:
    assert cur.update(0.90, 1 * M) is False
    assert cur.update(0.89, 1.1 * M) is False  # a miss: the streak restarts
    assert cur.update(0.95, 1.2 * M) is False
    assert cur.level == 0
    assert cur.update(0.90, 1.3 * M) is True  # exactly at the threshold counts
    assert cur.level == 1


def test_no_promotion_before_min_steps_but_promotes_as_soon_as_they_are_spent(cur: Any) -> None:
    for steps in (0.2 * M, 0.5 * M, 0.9 * M):
        assert cur.update(1.0, steps) is False  # perfect eval, but < 1M steps at the level
    assert cur.level == 0
    assert cur.update(1.0, 1.0 * M) is True  # 1M spent and the last evals were successes


def test_steps_at_level_count_from_the_promotion(cur: Any) -> None:
    cur.update(1.0, 1 * M)
    assert cur.update(1.0, 1 * M + 1) is True  # level 1 starts at 1M + 1
    assert cur.level == 1
    # level 1 needs 2M steps spent *at that level*
    assert cur.update(1.0, 2 * M) is False
    assert cur.update(1.0, 2.5 * M) is False
    assert cur.update(1.0, 3 * M) is False  # 1.99M spent
    assert cur.update(1.0, 3 * M + 2) is True
    assert cur.level == 2


def test_never_demotes(cur: Any) -> None:
    cur.update(1.0, 1 * M)
    cur.update(1.0, 2 * M)
    assert cur.level == 1
    for k in range(10):
        assert cur.update(0.0, (3 + k) * M) is False
    assert cur.level == 1


def test_stalls_are_flagged_and_do_not_demote_or_promote(cur: Any) -> None:
    for k in range(1, 6):
        cur.update(0.5, k * M)
    assert not cur.stalled  # 5M < the 6M max of level 0
    cur.update(0.5, 7 * M)
    assert cur.stalled
    assert cur.level == 0


def test_level_without_a_threshold_is_the_end_of_the_stage(api: ModuleType) -> None:
    cur = api.Curriculum(api.ATTITUDE_LEVELS, start_level=3)  # end of att_p5
    for k in range(5):
        assert cur.update(1.0, (10 + k) * M) is False
    assert cur.level == 3
    cur = api.Curriculum(api.ATTITUDE_LEVELS, start_level=6)
    assert cur.update(1.0, 100 * M) is False


def test_table_thresholds_match_the_adr(api: ModuleType) -> None:
    for k, (sr, lo, hi) in ATT_PROMOTION.items():
        spec = api.ATTITUDE_LEVELS[k]
        assert (spec.promote_sr, spec.min_steps, spec.max_steps) == (sr, lo, hi)
    assert [api.ATTITUDE_LEVELS[k].promote_sr for k in (4, 5)] == [0.85, 0.85]
    assert [api.ATTITUDE_LEVELS[k].dr_scale for k in range(7)] == [
        0,
        0.1,
        0.25,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    for k, (d, w, v, a) in ATT_LEVELS.items():
        spec = api.ATTITUDE_LEVELS[k]
        assert (spec.delta_max_deg, spec.omega_max_rad_s, spec.v0_max_m_s, spec.a0_max) == (
            d,
            w,
            v,
            a,
        )
    assert [lv.disturbances for lv in api.ATTITUDE_LEVELS] == [False] * 4 + [True] * 3
