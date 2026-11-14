"""T1 + T2 (ADR-0006 §2, §10): env API compliance and the observation layout."""

from __future__ import annotations

import json
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

pytestmark = pytest.mark.fast

ENV_IDS = ["DronezAttitude-v0", "DronezHover-v0"]
DIMS = {"DronezAttitude-v0": 29, "DronezHover-v0": 34}

# ADR-0006 §2 tables: (name, size, units, frame, offset, scale)
ATT_BLOCKS = [
    ("r_err_6d", 6, "1", 0.0, 1.0),
    ("gravity_b", 3, "1", 0.0, 1.0),
    ("gyro_b", 3, "rad/s", 0.0, 5.0),
    ("vel_z_w", 1, "m/s", 0.0, 2.0),
    ("action_hist", 16, "1", 0.0, 1.0),
]
HOVER_EXTRA = [("pos_err_h", 3, "m", 0.0, 1.0), ("vel_xy_h", 2, "m/s", 0.0, 2.0)]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_gymnasium_check_env(mk: Any, env_id: str) -> None:
    gym_check_env(mk(env_id).unwrapped, skip_render_check=True)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_sb3_check_env(mk: Any, env_id: str) -> None:
    sb3_check_env(mk(env_id), warn=True, skip_render_check=True)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_spaces_and_rates(mk: Any, env_id: str) -> None:
    env = mk(env_id)
    assert env.observation_space.shape == (DIMS[env_id],)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (4,)
    assert env.action_space.dtype == np.float32
    np.testing.assert_array_equal(env.action_space.low, -1.0)
    np.testing.assert_array_equal(env.action_space.high, 1.0)
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_layout_is_contiguous_unique_and_matches_the_space(
    api: ModuleType, mk: Any, env_id: str
) -> None:
    env = mk(env_id)
    lay = api.layout_of(env)
    dim = DIMS[env_id]
    assert lay.dim == dim == env.observation_space.shape[0]
    slices = lay.slices  # name -> slice
    assert list(slices) == list(lay.names)
    assert len(set(lay.names)) == len(lay.names), "duplicate block names"
    cursor = 0
    for name in lay.names:
        sl = slices[name]
        assert sl.start == cursor, f"{name} is not contiguous"
        assert sl.stop > sl.start
        cursor = sl.stop
    assert cursor == dim, "the slices must cover the whole observation"
    for name in lay.names:
        assert lay.units[name], f"{name}: units not set"
        assert lay.frames[name], f"{name}: frame not set"
    assert lay.offset.shape == lay.scale.shape == (dim,)
    assert np.all(lay.scale > 0.0)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_layout_matches_the_adr_tables(api: ModuleType, mk: Any, env_id: str) -> None:
    lay = api.layout_of(mk(env_id))
    expected = ATT_BLOCKS + (HOVER_EXTRA if env_id == "DronezHover-v0" else [])
    assert list(lay.names) == [b[0] for b in expected]
    for name, size, units, offset, scale in expected:
        sl = lay.slices[name]
        assert sl.stop - sl.start == size, name
        assert lay.units[name] == units, name
        np.testing.assert_array_equal(lay.offset[sl], offset)
        np.testing.assert_array_equal(lay.scale[sl], scale)
    # the first block of the Hover layout is the whole Attitude layout (prefix compatible)
    assert lay.slices["action_hist"] == slice(13, 29)
    if env_id == "DronezHover-v0":
        assert lay.slices["pos_err_h"] == slice(29, 32)
        assert lay.slices["vel_xy_h"] == slice(32, 34)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_layout_to_meta_round_trip(api: ModuleType, mk: Any, env_id: str) -> None:
    lay = api.layout_of(mk(env_id))
    meta = lay.to_meta()
    assert json.loads(json.dumps(meta)) == meta, "to_meta() must be plain JSON data"
    assert meta["dim"] == DIMS[env_id]
    rebuilt = api.ObsLayout.from_meta(json.loads(json.dumps(meta)))
    assert rebuilt.names == lay.names
    assert rebuilt.dim == lay.dim
    assert rebuilt.slices == lay.slices
    assert rebuilt.units == lay.units
    assert rebuilt.frames == lay.frames
    np.testing.assert_array_equal(rebuilt.offset, lay.offset)
    np.testing.assert_array_equal(rebuilt.scale, lay.scale)


def test_privileged_layout_is_46_wide(api: ModuleType) -> None:
    for lay in (api.ObsLayout.attitude(), api.ObsLayout.hover()):
        assert lay.privileged_dim == 46
        last = max(sl.stop for sl in lay.privileged_slices.values())
        assert last == 46
