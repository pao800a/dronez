"""Tests for dronez.utils.rotations, cross-checked against MuJoCo's own math."""

from __future__ import annotations

import numpy as np
import pytest

from dronez.utils import rotations as rot

pytestmark = pytest.mark.fast

RNG = np.random.default_rng(1234)


def _rand_quats(n: int = 64) -> np.ndarray:
    return rot.random_quat(RNG, n)


def test_identity_and_normalize() -> None:
    np.testing.assert_allclose(rot.identity_quat(), [1, 0, 0, 0])
    q = rot.quat_normalize([2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q, [1, 0, 0, 0])
    np.testing.assert_allclose(np.linalg.norm(_rand_quats(), axis=-1), 1.0, atol=1e-12)


def test_mul_matches_mujoco() -> None:
    mujoco = pytest.importorskip("mujoco")
    a, b = _rand_quats(32), _rand_quats(32)
    ours = rot.quat_mul(a, b)
    for i in range(len(a)):
        ref = np.zeros(4)
        mujoco.mju_mulQuat(ref, a[i], b[i])
        np.testing.assert_allclose(ours[i], ref, atol=1e-12)


def test_rotmat_matches_mujoco_and_roundtrips() -> None:
    mujoco = pytest.importorskip("mujoco")
    qs = _rand_quats(32)
    mats = rot.quat_to_rotmat(qs)
    for q, m in zip(qs, mats, strict=True):
        ref = np.zeros(9)
        mujoco.mju_quat2Mat(ref, q)
        np.testing.assert_allclose(m, ref.reshape(3, 3), atol=1e-12)
    back = rot.rotmat_to_quat(mats)
    np.testing.assert_allclose(back, rot.quat_canonical(qs), atol=1e-10)


def test_rotmat_to_quat_near_pi() -> None:
    # 180° rotations exercise the non-trace pivots of Shepperd's method.
    for axis in np.eye(3):
        q = rot.quat_from_axis_angle(axis, np.pi)
        r = rot.quat_to_rotmat(q)
        np.testing.assert_allclose(rot.quat_to_rotmat(rot.rotmat_to_quat(r)), r, atol=1e-12)


def test_rotate_matches_matrix_and_mujoco() -> None:
    mujoco = pytest.importorskip("mujoco")
    qs = _rand_quats(16)
    v = RNG.normal(size=(16, 3))
    ours = rot.quat_rotate(qs, v)
    np.testing.assert_allclose(ours, np.einsum("nij,nj->ni", rot.quat_to_rotmat(qs), v), atol=1e-12)
    for i in range(16):
        ref = np.zeros(3)
        mujoco.mju_rotVecQuat(ref, v[i], qs[i])
        np.testing.assert_allclose(ours[i], ref, atol=1e-12)
    np.testing.assert_allclose(rot.quat_rotate_inverse(qs, ours), v, atol=1e-12)


def test_body_to_world_convention() -> None:
    # Pitch nose-down by +90° about body/world y: body +x points to world -z.
    q = rot.quat_from_axis_angle([0, 1, 0], np.pi / 2)
    np.testing.assert_allclose(rot.quat_rotate(q, [1, 0, 0]), [0, 0, -1], atol=1e-12)
    # Roll +90° about x (right-hand rule): body +y (left) points to world +z.
    q = rot.quat_from_axis_angle([1, 0, 0], np.pi / 2)
    np.testing.assert_allclose(rot.quat_rotate(q, [0, 1, 0]), [0, 0, 1], atol=1e-12)


def test_geodesic_angle() -> None:
    q = _rand_quats(32)
    np.testing.assert_allclose(rot.geodesic_angle(q, q), 0.0, atol=1e-6)
    np.testing.assert_allclose(rot.geodesic_angle(q, -q), 0.0, atol=1e-6)  # double cover
    angles = RNG.uniform(0, np.pi, 32)
    axes = RNG.normal(size=(32, 3))
    dq = rot.quat_from_axis_angle(axes, angles)
    np.testing.assert_allclose(rot.geodesic_angle(rot.quat_mul(q, dq), q), angles, atol=1e-9)
    # Symmetric.
    p = _rand_quats(32)
    np.testing.assert_allclose(rot.geodesic_angle(p, q), rot.geodesic_angle(q, p), atol=1e-12)


def test_error_matrix_is_rt_r() -> None:
    q, t = _rand_quats(8), _rand_quats(8)
    expected = np.swapaxes(rot.quat_to_rotmat(t), -1, -2) @ rot.quat_to_rotmat(q)
    np.testing.assert_allclose(rot.rotation_error_matrix(q, t), expected, atol=1e-12)


def test_tilt_gravity_yaw() -> None:
    assert rot.tilt_angle(rot.identity_quat()) == pytest.approx(0.0)
    inverted = rot.quat_from_axis_angle([1, 0, 0], np.pi)
    assert rot.tilt_angle(inverted) == pytest.approx(np.pi)
    np.testing.assert_allclose(rot.gravity_in_body(rot.identity_quat()), [0, 0, -1])
    np.testing.assert_allclose(rot.gravity_in_body(inverted), [0, 0, 1], atol=1e-12)
    # Yaw is unaffected by pure tilt at small angles and recovers a pure yaw.
    for yaw in (-2.0, 0.3, 3.0):
        assert rot.yaw_angle(rot.quat_from_yaw(yaw)) == pytest.approx(yaw)


def test_axis_angle_roundtrip() -> None:
    axes = RNG.normal(size=(16, 3))
    angles = RNG.uniform(0, np.pi, 16)
    axis, angle = rot.quat_to_axis_angle(rot.quat_from_axis_angle(axes, angles))
    np.testing.assert_allclose(angle, angles, atol=1e-9)
    np.testing.assert_allclose(axis, axes / np.linalg.norm(axes, axis=-1, keepdims=True), atol=1e-9)


def test_random_tilted_respects_bounds() -> None:
    qs = rot.random_tilted_quat(RNG, max_tilt=np.deg2rad(30), size=2000, min_tilt=np.deg2rad(10))
    tilt = rot.tilt_angle(qs)
    assert tilt.min() >= np.deg2rad(10) - 1e-9
    assert tilt.max() <= np.deg2rad(30) + 1e-9


def test_random_quat_uniform_tilt_distribution() -> None:
    # Uniform on SO(3) => cos(tilt) is uniform on [-1, 1].
    qs = rot.random_quat(np.random.default_rng(0), 20000)
    cos_t = np.cos(rot.tilt_angle(qs))
    assert abs(cos_t.mean()) < 0.03
    assert np.var(cos_t) == pytest.approx(1.0 / 3.0, abs=0.02)


def test_frame_vectors() -> None:
    np.testing.assert_allclose(rot.vec_enu_to_ned([1.0, 2.0, 3.0]), [2.0, 1.0, -3.0])
    np.testing.assert_allclose(rot.vec_flu_to_frd([1.0, 2.0, 3.0]), [1.0, -2.0, -3.0])


def test_quat_frame_conversion() -> None:
    # Level, nose east in ENU/FLU -> level, nose east in NED/FRD = yaw +90° in NED.
    q_ned = rot.quat_enu_flu_to_ned_frd(rot.identity_quat())
    np.testing.assert_allclose(
        rot.quat_rotate(q_ned, [1, 0, 0]), [0, 1, 0], atol=1e-12
    )  # FRD forward = NED east
    np.testing.assert_allclose(
        rot.quat_rotate(q_ned, [0, 0, 1]), [0, 0, 1], atol=1e-12
    )  # FRD down = NED down
    # Consistency with vector conversions for random attitudes.
    qs = _rand_quats(16)
    v_flu = RNG.normal(size=(16, 3))
    v_enu = rot.quat_rotate(qs, v_flu)
    lhs = rot.quat_rotate(rot.quat_enu_flu_to_ned_frd(qs), rot.vec_flu_to_frd(v_flu))
    np.testing.assert_allclose(lhs, rot.vec_enu_to_ned(v_enu), atol=1e-12)
    back = rot.quat_ned_frd_to_enu_flu(rot.quat_enu_flu_to_ned_frd(qs))
    np.testing.assert_allclose(rot.geodesic_angle(back, qs), 0.0, atol=1e-6)
