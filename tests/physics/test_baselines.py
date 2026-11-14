"""Gate G3: both baseline controllers stabilize from 30 deg and hold hover within 5 cm.

Closed loop on the real CAD-built model at the policy rate (100 Hz, decimation 10), nominal plant,
sensor noise on, episodes seeded as in the eval suite (``configs/eval/baseline.yaml``).
"""

from __future__ import annotations

import numpy as np
import pytest

from dronez.control import CascadedPID, GeometricSO3, NominalVehicle
from dronez.eval import metrics as M
from dronez.eval.runner import SimEnv, run_episode
from dronez.eval.scenarios import episode_seed, load_suite, make_plan
from dronez.sim import DroneSim, ImuNoiseConfig

pytestmark = pytest.mark.physics

SUITE = load_suite("baseline")
SCENARIOS = {sc.name: (i, sc) for i, sc in enumerate(SUITE.scenarios)}


@pytest.fixture(scope="module", params=["pid", "geometric"])
def harness(request: pytest.FixtureRequest, rig):  # type: ignore[no-untyped-def]
    sim = DroneSim(rig.cfg, rig.sim_cfg)
    veh = NominalVehicle.from_drone(rig.cfg, rig.art)
    ctrl = CascadedPID(veh) if request.param == "pid" else GeometricSO3(veh)
    return ctrl, SimEnv(sim, ImuNoiseConfig())


def _episodes(name: str, harness):  # type: ignore[no-untyped-def]
    ctrl, env = harness
    idx, sc = SCENARIOS[name]
    for ei in range(sc.n_episodes):
        yield sc, run_episode(env, ctrl, make_plan(sc, episode_seed(SUITE.seed, idx, ei)))


def test_noise_is_on_for_the_gate() -> None:
    n = ImuNoiseConfig()
    assert n.gyro_noise_density > 0
    assert n.quat_noise_rad > 0
    assert n.pos_noise_std > 0


def test_g3_recovery_from_30deg(harness) -> None:  # type: ignore[no-untyped-def]
    """Attitude < 5 deg within 1.0 s (and staying there), every seed (>= 20)."""
    n = 0
    for sc, tr in _episodes("recovery_30deg", harness):
        err = M.attitude_error(tr.quat_wxyz, tr.quat_target)
        assert np.rad2deg(err[0]) == pytest.approx(30.0, abs=1e-6)
        ttr = M.time_to_recover(err, tr.dt)
        assert not tr.crashed, f"seed {tr.seed} crashed"
        assert ttr <= 1.0, f"seed {tr.seed}: recovered in {ttr:.2f} s"
        assert M.success(err, tr.dt, hold_s=sc.hold_s)
        n += 1
    assert n >= 20


def test_g3_hover_within_5cm_from_rest_for_10s(harness) -> None:  # type: ignore[no-untyped-def]
    for sc, tr in _episodes("hover_hold", harness):
        assert sc.duration_s == 10.0
        assert tr.t[-1] == pytest.approx(10.0)
        perr = M.position_error(tr.pos_w, tr.pos_target)
        assert perr.max() <= 0.05, f"seed {tr.seed}: max error {100 * perr.max():.2f} cm"


def test_step_attitude_20deg_settles_without_large_overshoot(harness) -> None:  # type: ignore[no-untyped-def]
    for sc, tr in _episodes("step_attitude_20deg", harness):
        err = M.attitude_error(tr.quat_wxyz, tr.quat_target)
        assert M.time_to_recover(err, tr.dt, t_start=sc.step_time_s) <= 1.0
        os_ = M.overshoot(tr.quat_wxyz[50:], tr.quat_target[0], tr.quat_target[-1])
        assert np.rad2deg(os_) < 5.0


def test_small_impulse_is_rejected(harness) -> None:  # type: ignore[no-untyped-def]
    for sc, tr in _episodes("impulse_small", harness):
        assert not tr.crashed
        err = M.attitude_error(tr.quat_wxyz, tr.quat_target)
        assert M.success(err, tr.dt, hold_s=sc.hold_s)
        assert M.position_error(tr.pos_w, tr.pos_target)[-100:].max() < 0.05
