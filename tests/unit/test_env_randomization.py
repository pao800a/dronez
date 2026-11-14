"""T9 (ADR-0006 §7): domain randomization at d = 0 and d = 1, the extremes, and no model drift."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from dronez.cad.artifacts import ModelArtifacts
from dronez.cad.schema import DroneConfig
from dronez.sim import (
    DroneSim,
    PhysParams,
    SimConfig,
    build_model_from_artifacts,
    inertia_triangle_ok,
)

pytestmark = pytest.mark.fast

N_DRAWS = 10_000
P_EXTREME = 0.15


@pytest.fixture(scope="module")
def sim(sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts) -> DroneSim:
    cfg = SimConfig()
    model, index = build_model_from_artifacts(sim_cfg_drone, sim_art, cfg)
    return DroneSim(cfg=sim_cfg_drone, sim=cfg, model=model, index=index)


@pytest.fixture(scope="module")
def rd(api: ModuleType, sim: DroneSim):
    return api.DomainRandomizer(sim.nominal_params(), physics_dt_s=sim.dt)


def _same(a: PhysParams, b: PhysParams) -> None:
    assert a.mass_kg == b.mass_kg
    np.testing.assert_array_equal(a.com_offset_b_m, b.com_offset_b_m)
    np.testing.assert_array_equal(a.inertia_diag_kg_m2, b.inertia_diag_kg_m2)
    for f in ("k_f", "k_m", "tau_m_s", "omega_min", "omega_max"):
        np.testing.assert_array_equal(getattr(a.motor, f), getattr(b.motor, f), err_msg=f)
    assert a.latency_steps == b.latency_steps
    assert a.battery_scale == b.battery_scale


def test_width_zero_is_exactly_nominal(rd, sim: DroneSim) -> None:
    nominal = sim.nominal_params()
    for seed in range(50):
        draw = rd.draw(np.random.default_rng(seed), 0.0)
        _same(draw.phys, nominal)
        assert draw.noise_scale == 1.0  # nominal sensor noise stays on (the G4 condition)


@pytest.fixture(scope="module")
def draws(rd):
    rng = np.random.default_rng(2024)
    return [rd.draw(rng, 1.0) for _ in range(N_DRAWS)]


def test_full_width_draws_stay_inside_the_adr_ranges(draws, sim: DroneSim) -> None:
    nom = sim.nominal_params()
    m_ratio = np.array([d.phys.mass_kg for d in draws]) / nom.mass_kg
    assert m_ratio.min() >= 0.8 - 1e-12
    assert m_ratio.max() <= 1.2 + 1e-12

    inertia = np.array([d.phys.inertia_diag_kg_m2 for d in draws]) / nom.inertia_diag_kg_m2
    assert inertia.min() >= 0.8 * 0.95 - 1e-12  # common +-20 %, anisotropy +-5 % per axis
    assert inertia.max() <= 1.2 * 1.05 + 1e-12
    assert all(inertia_triangle_ok(d.phys.inertia_diag_kg_m2) for d in draws)

    com = np.array([d.phys.com_offset_b_m for d in draws])
    assert np.abs(com[:, :2]).max() <= 3e-3 + 1e-12
    assert np.abs(com[:, 2]).max() <= 5e-3 + 1e-12

    kf = np.array([d.phys.motor.k_f for d in draws]) / nom.motor.k_f
    assert kf.min() >= 0.85 * 0.95 - 1e-12  # common +-15 %, per-motor mismatch +-5 %
    assert kf.max() <= 1.15 * 1.05 + 1e-12
    km_ratio = np.array([d.phys.motor.k_m / d.phys.motor.k_f for d in draws])
    km_ratio = km_ratio / (nom.motor.k_m / nom.motor.k_f)
    assert km_ratio.min() >= 0.8 - 1e-9  # k_m +-20 % (the yaw gear k_m / k_f)
    assert km_ratio.max() <= 1.2 + 1e-9

    tau = np.array([d.phys.motor.tau_m_s for d in draws]) / nom.motor.tau_m_s
    assert tau.min() >= 0.5 * 0.9 - 1e-9  # 15..60 ms common, +-10 % per motor
    assert tau.max() <= 2.0 * 1.1 + 1e-9

    steps = np.array([d.phys.latency_steps for d in draws])
    assert steps.min() >= 0
    assert steps.max() <= round(0.020 / sim.dt)

    battery = np.array([d.phys.battery_scale for d in draws])
    assert battery.min() >= 0.85 - 1e-12
    assert battery.max() <= 1.05 + 1e-12

    noise = np.array([d.noise_scale for d in draws])
    assert noise.min() >= 0.5 - 1e-12
    assert noise.max() <= 2.0 + 1e-12


def test_edges_are_visited_at_about_p_extreme(draws, sim: DroneSim) -> None:
    nom = sim.nominal_params()
    m_ratio = np.array([d.phys.mass_kg for d in draws]) / nom.mass_kg
    at_edge = np.isclose(m_ratio, 0.8, atol=1e-12) | np.isclose(m_ratio, 1.2, atol=1e-12)
    assert at_edge.mean() == pytest.approx(P_EXTREME, abs=0.02)
    # both corners, roughly equally
    assert 0.05 < np.isclose(m_ratio, 1.2, atol=1e-12).mean() < 0.10

    com_z = np.array([d.phys.com_offset_b_m[2] for d in draws])
    edge_z = np.isclose(np.abs(com_z), 5e-3, atol=1e-12)
    assert edge_z.mean() == pytest.approx(P_EXTREME, abs=0.02)

    # the rest is spread over the interior, not piled on the edges
    interior = m_ratio[~at_edge]
    assert interior.min() < 0.85
    assert interior.max() > 1.15
    assert abs(np.mean(m_ratio) - 1.0) < 0.01


def test_apply_a_b_a_leaves_an_identical_model(rd, sim: DroneSim) -> None:
    m, b = sim.model, sim.index.body_id

    def snap() -> dict[str, np.ndarray]:
        return {
            "body_mass": np.array(m.body_mass),
            "body_inertia": np.array(m.body_inertia),
            "body_ipos": np.array(m.body_ipos),
            "body_iquat": np.array(m.body_iquat),
            "actuator_gear": np.array(m.actuator_gear),
            "body_subtreemass": np.array(m.body_subtreemass),
            "body_invweight0": np.array(m.body_invweight0),
            "dof_invweight0": np.array(m.dof_invweight0),
            "dof_M0": np.array(m.dof_M0),
            "motor_omega_max": np.array(sim.motor.params.omega_max),
            "motor_k_f": np.array(sim.motor.params.k_f),
            "motor_tau": np.array(sim.motor.params.tau_m_s),
            "delay": np.array([sim.delay.n_steps]),
        }

    pristine = snap()
    d_a = rd.draw(np.random.default_rng(1), 1.0).phys
    d_b = rd.draw(np.random.default_rng(2), 1.0).phys
    rd.apply(sim, d_a)
    after_a = snap()
    assert not np.array_equal(after_a["body_mass"], pristine["body_mass"])  # it really changed
    rd.apply(sim, d_b)
    assert not np.array_equal(snap()["body_ipos"], after_a["body_ipos"])
    rd.apply(sim, d_a)
    again = snap()
    for k, v in after_a.items():
        np.testing.assert_array_equal(again[k], v, err_msg=f"drift in {k} after A -> B -> A")
    # and a long random walk through parameter space still returns exactly to the nominal
    rng = np.random.default_rng(5)
    for _ in range(100):
        rd.apply(sim, rd.draw(rng, 1.0).phys)
    rd.apply(sim, sim.nominal_params())
    final = snap()
    for k, v in pristine.items():
        np.testing.assert_array_equal(final[k], v, err_msg=f"nominal not restored: {k}")
    assert b == sim.index.body_id


def test_same_seed_perturbs_in_the_same_direction_at_every_width(rd, sim: DroneSim) -> None:
    """D2: parameter k uses the same uniform at every d (linear scaling of the deviation)."""
    nom = sim.nominal_params()
    for seed in range(40):
        full = rd.draw(np.random.default_rng(seed), 1.0)
        half = rd.draw(np.random.default_rng(seed), 0.5)
        np.testing.assert_array_equal(full.z, half.z)  # the same unit variates and extreme flags
        for get, ref in (
            (lambda p: p.mass_kg, nom.mass_kg),
            (lambda p: p.battery_scale, 1.0),
            (lambda p: p.com_offset_b_m[2], 0.0),
        ):
            dev_full = get(full.phys) - ref
            dev_half = get(half.phys) - ref
            assert dev_half == pytest.approx(0.5 * dev_full, abs=1e-12, rel=1e-9)


def test_the_draw_consumes_a_width_independent_amount_of_randomness(rd) -> None:
    after = []
    for d in (0.0, 0.1, 1.0):
        rng = np.random.default_rng(8)
        rd.draw(rng, d)
        after.append(rng.random())
    assert after[0] == after[1] == after[2]
