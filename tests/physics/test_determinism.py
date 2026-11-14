"""Determinism, idempotent parameter writes, and model/mass-property consistency."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.physics


def _u_seq(rig, n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t_h = rig.mass_kg * rig.g / 4.0
    return rng.uniform(0.7, 1.3, size=(n, 4)) * t_h


def _run(plant, rig, n: int = 300) -> np.ndarray:
    plant.reset_hover(z=1.0)
    u = np.array([plant.u_for_thrust(t) for t in _u_seq(rig, n)])
    return plant.rollout(u, force_w=[0.05, -0.02, 0.01])


def test_two_rollouts_bit_identical(rig):
    """Same model, same commands, same wrench: trajectories identical bit for bit, both for a
    second reset of the same sim and for a freshly built sim (no hidden state, no RNG)."""
    a = rig.new_plant()
    first = _run(a, rig)
    second = _run(a, rig)
    other = _run(rig.new_plant(), rig)
    assert np.array_equal(first, second), "re-reset of the same sim is not reproducible"
    assert np.array_equal(first, other), "a fresh sim gives a different trajectory"
    assert np.all(np.isfinite(first))


def test_set_params_nominal_is_idempotent(rig):
    """``set_params(nominal)`` must leave mass, inertia, CoM offset and actuator gear at their
    built values (rtol 1e-12) and must not drift: 20 perturbed/nominal cycles reproduce the
    first nominal write bit for bit, and so does the rollout."""
    plant = rig.new_plant()
    built = plant.snapshot()
    plant.set_params(plant.nominal)
    first = plant.snapshot()
    for key, ref in built.items():
        np.testing.assert_allclose(first[key], ref, rtol=1e-12, atol=1e-15, err_msg=key)
    traj_first = _run(plant, rig, 100)

    perturbed = plant.replace(
        mass_kg=1.1 * plant.nominal.mass_kg,
        inertia_diag_kg_m2=1.1 * plant.nominal.inertia_diag_kg_m2,
        com_offset_b_m=np.array([0.002, -0.001, 0.003]),
        latency_steps=3,
        battery_scale=0.9,
        motor=plant.with_motor(k_f=0.9 * rig.k_f, tau_m_s=1.2 * rig.tau_m).motor,
    )
    for _ in range(20):
        plant.set_params(perturbed)
        mid = plant.snapshot()
        assert not np.array_equal(mid["body_mass"], first["body_mass"]), "perturbation not applied"
        plant.set_params(plant.nominal)
    last = plant.snapshot()
    for key, ref in first.items():
        assert np.array_equal(last[key], ref), f"{key} drifted after perturb/nominal cycles"
    assert np.array_equal(_run(plant, rig, 100), traj_first), "rollout drifted after cycles"


def test_model_matches_mass_properties(rig):
    """Compiled model vs ``mass_props.json``: mass rel 1e-9, sorted principal moments rel 1e-6,
    body-frame motor positions 1e-9 m, spin signs equal the YAML's, gravity = (0, 0, -9.81)."""
    plant = rig.new_plant()
    b = plant.index.body_id
    np.testing.assert_allclose(plant.model.body_mass[b], rig.mass_kg, rtol=1e-9)
    np.testing.assert_allclose(
        np.sort(plant.model.body_inertia[b]),
        np.sort(np.array(rig.mp.principal_moments_kg_m2)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(plant.index.motor_pos_b, rig.hubs, atol=1e-9)
    np.testing.assert_array_equal(np.asarray(plant.index.spin), rig.spin)
    np.testing.assert_allclose(plant.model.opt.gravity, [0.0, 0.0, -rig.g])
    np.testing.assert_allclose(plant.model.body_ipos[b], 0.0, atol=1e-12)


def _triangle_ok(i: np.ndarray, tol: float = 1e-12) -> bool:
    a, b, c = i
    return bool(a + b >= c - tol and b + c >= a - tol and a + c >= b - tol)


def test_inertia_triangle_inequality_under_scaling(rig):
    """Nominal principal moments satisfy I_a + I_b >= I_c (flat quad: I_zz ~ I_xx + I_yy).

    Uniform scaling (mass-scale DR) keeps it. A triangle-violating anisotropic request (I_zz
    x1.2, I_xx and I_yy x0.8) must either be rejected with ValueError or leave a model that
    still satisfies it; a silently invalid inertia is a failure.
    """
    plant = rig.new_plant()
    b = plant.index.body_id
    nominal = np.array(plant.nominal.inertia_diag_kg_m2)
    assert _triangle_ok(nominal), f"nominal inertia {nominal} violates the triangle inequality"

    for scale in (0.8, 1.2):
        plant.set_params(plant.replace(inertia_diag_kg_m2=scale * nominal))
        np.testing.assert_allclose(plant.model.body_inertia[b], scale * nominal, rtol=1e-12)
        assert _triangle_ok(np.array(plant.model.body_inertia[b]))

    order = np.argsort(nominal)  # smallest two are the in-plane moments, largest is I_zz
    bad = nominal.copy()
    bad[order[:2]] *= 0.8
    bad[order[2]] *= 1.2
    assert not _triangle_ok(bad), "test premise: the anisotropic request must be invalid"
    try:
        plant.set_params(plant.replace(inertia_diag_kg_m2=bad))
    except ValueError:
        pass
    else:
        assert _triangle_ok(np.array(plant.model.body_inertia[b])), "invalid inertia was accepted"
    plant.set_params(plant.nominal)
    np.testing.assert_allclose(plant.model.body_inertia[b], nominal, rtol=1e-12)
