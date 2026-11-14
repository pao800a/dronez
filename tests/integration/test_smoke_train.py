"""T15 (ADR-0006 §8, §10): the smoke training run, then the eval harness on its checkpoint + PID.

CPU, ``DummyVecEnv``, 2 envs, 8192 steps (``configs/train/smoke.yaml``). No learning-curve asserts:
the point is that every stage of the pipeline runs and leaves consistent artifacts.
The run writes into a private ``DRONEZ_DATA_DIR`` (the cached model is copied in first).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from dronez.cad.artifacts import load_artifacts, model_dir
from dronez.cad.schema import load_drone_config
from dronez.utils.paths import ENV_VAR

pytestmark = pytest.mark.integration

RUN_ID = "smoke-test"
TERMS = (
    "alive", "att", "att_fine", "rate", "vz", "act_mag", "act_smooth", "act_sat",
    "penalty_clip", "crash",
)  # fmt: skip


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A private data dir holding a copy of the cached model; ``DRONEZ_DATA_DIR`` points to it."""
    cfg = load_drone_config("default_5in")
    try:
        load_artifacts(cfg)
    except Exception as exc:
        pytest.skip(
            f"CAD cache missing ({exc}); run `uv run dronez-build-model --drone default_5in`"
        )
    src = model_dir(cfg)
    root = tmp_path_factory.mktemp("dronez-data")
    shutil.copytree(src, root / "models" / src.name)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(ENV_VAR, str(root))
        yield root


@pytest.fixture(scope="module")
def run_dir(api: ModuleType, data_dir: Path) -> Path:
    cfg = api.load_train_config("smoke")
    assert cfg.run.total_timesteps == 8192
    assert cfg.run.n_envs == 2
    assert cfg.run.vec_env == "dummy"
    out = Path(api.train(cfg, run_id=RUN_ID, config_stem="smoke"))
    assert out.parent == data_dir / "runs"
    return out


def test_run_reaches_finished(run_dir: Path) -> None:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "finished", status.get("error")
    assert status["timesteps_done"] >= 8192
    assert status["error"] is None
    assert status["run_id"] == RUN_ID
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "train.log").is_file()


def test_checkpoint_directories_are_complete(run_dir: Path) -> None:
    steps = sorted((run_dir / "checkpoints").glob("step_*"))
    assert steps, "no step_* checkpoint"
    for d in steps:
        for name in ("model.zip", "vecnormalize.pkl", "ckpt.json"):
            assert (d / name).is_file(), f"{d.name}/{name} missing"
        meta = json.loads((d / "ckpt.json").read_text(encoding="utf-8"))
        assert {"timesteps", "created_at"} <= set(meta)
    assert not list((run_dir / "checkpoints").glob(".tmp-*")), "half-written checkpoint left over"
    assert max(json.loads((d / "ckpt.json").read_text())["timesteps"] for d in steps) >= 8192


def test_tensorboard_has_every_reward_term(run_dir: Path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    tags: set[str] = set()
    event_dirs = {p.parent for p in (run_dir / "tb").rglob("events.out.tfevents.*")}
    assert event_dirs, "no TensorBoard event files under tb/"
    for d in event_dirs:
        acc = EventAccumulator(str(d), size_guidance={"scalars": 0})
        acc.Reload()
        tags |= set(acc.Tags()["scalars"])
    missing = [t for t in TERMS if f"reward/{t}" not in tags]
    assert not missing, f"missing reward tags {missing}; have {sorted(tags)}"
    for tag in (
        "curriculum/level",
        "action/saturation_frac",
        "obs/clip_frac",
        "metrics/att_err_rad",
    ):
        assert tag in tags, f"{tag} missing"


def test_reloaded_checkpoint_gives_a_finite_deterministic_action(
    api: ModuleType, run_dir: Path
) -> None:
    ckpt = sorted((run_dir / "checkpoints").glob("step_*"))[-1]
    model, vn = api.load_checkpoint(ckpt)
    layout = api.ObsLayout.attitude()
    obs = np.zeros((1, layout.dim), dtype=np.float32)
    obs[0, 0], obs[0, 4], obs[0, 8] = 1.0, 1.0, -1.0  # level, at the target: gravity (0, 0, -1)
    normalized = vn.normalize_obs(obs)
    a1, _ = model.predict(normalized, deterministic=True)
    a2, _ = model.predict(normalized, deterministic=True)
    assert a1.shape == (1, 4)
    assert np.all(np.isfinite(a1))
    assert np.all(np.abs(a1) <= 1.0)
    np.testing.assert_array_equal(a1, a2)


def test_frozen_observation_statistics_survive_training(api: ModuleType, run_dir: Path) -> None:
    layout = api.ObsLayout.attitude()
    for ckpt in sorted((run_dir / "checkpoints").glob("step_*")):
        _, vn = api.load_checkpoint(ckpt)
        assert isinstance(vn.obs_rms, api.FixedRunningMeanStd)
        np.testing.assert_array_equal(vn.obs_rms.mean, layout.offset)
        np.testing.assert_array_equal(vn.obs_rms.var, layout.scale**2)
        assert vn.clip_obs == 10.0
        assert vn.norm_reward  # the reward normalization stays on (and adaptive)
