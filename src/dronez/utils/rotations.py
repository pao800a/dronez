"""The single home of all rotation math in Dronez.

Conventions (see docs/conventions.md):

* Quaternions are scalar-first ``(w, x, y, z)`` — MuJoCo order — and Hamilton
  product. A quaternion ``q`` maps body-frame vectors to world-frame vectors:
  ``v_world = q ⊗ v_body ⊗ q*``, equivalently ``v_world = R(q) @ v_body``.
* World frame is z-up (ENU-like); body frame is FLU (x fwd, y left, z up).
* Attitude error is the geodesic angle on SO(3), never Euler angles.

Every function accepts a single quaternion of shape ``(4,)`` or a batch of
shape ``(..., 4)`` (rotation matrices ``(..., 3, 3)``) and broadcasts.
Nothing else in the code base may implement quaternion math ad hoc.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

F64 = NDArray[np.float64]

_EPS = 1e-12

#: Rotation that maps ENU (x east, y north, z up) coordinates to NED
#: (x north, y east, z down) coordinates: ``v_ned = R_NED_ENU @ v_enu``.
R_NED_ENU: F64 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

#: Rotation that maps FLU body coordinates to FRD body coordinates:
#: ``v_frd = R_FRD_FLU @ v_flu`` (a 180° turn about body x).
R_FRD_FLU: F64 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _as_f64(a: ArrayLike) -> F64:
    return np.asarray(a, dtype=np.float64)


def identity_quat() -> F64:
    """Return the identity quaternion ``(1, 0, 0, 0)``."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: ArrayLike) -> F64:
    """Normalize to unit length; the sign is left unchanged."""
    q = _as_f64(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, _EPS)


def quat_canonical(q: ArrayLike) -> F64:
    """Return the unit quaternion representing the same rotation with ``w >= 0``."""
    q = quat_normalize(q)
    return np.where(q[..., :1] < 0.0, -q, q)


def quat_conj(q: ArrayLike) -> F64:
    """Conjugate (the inverse, for unit quaternions)."""
    out: F64 = _as_f64(q) * np.array([1.0, -1.0, -1.0, -1.0])
    return out


def quat_mul(q1: ArrayLike, q2: ArrayLike) -> F64:
    """Hamilton product ``q1 ⊗ q2`` (apply ``q2`` first, then ``q1``)."""
    a = _as_f64(q1)
    b = _as_f64(q2)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _cross3(a: F64, b: F64) -> F64:
    """Cross product over the last axis (3 components), broadcasting.

    Same result as ``np.cross`` but ~5x faster for small arrays (``np.cross`` spends most of its
    time in ``moveaxis``), which matters: ``quat_rotate`` runs once per physics substep.
    """
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def quat_rotate(q: ArrayLike, v: ArrayLike) -> F64:
    """Rotate vector(s) ``v`` by ``q``: body → world for an attitude quaternion."""
    q = _as_f64(q)
    v = _as_f64(v)
    if q.shape == (4,) and v.shape == (3,):  # scalar fast path (hot: once per physics substep)
        w, x, y, z = q.tolist()
        vx, vy, vz = v.tolist()
        tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
        return np.array(
            [
                vx + w * tx + (y * tz - z * ty),
                vy + w * ty + (z * tx - x * tz),
                vz + w * tz + (x * ty - y * tx),
            ]
        )
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * _cross3(u, v)
    return v + w * t + _cross3(u, t)


def quat_rotate_inverse(q: ArrayLike, v: ArrayLike) -> F64:
    """Rotate vector(s) ``v`` by ``q*``: world → body for an attitude quaternion."""
    return quat_rotate(quat_conj(q), v)


def quat_from_axis_angle(axis: ArrayLike, angle: ArrayLike) -> F64:
    """Quaternion for a rotation of ``angle`` rad about ``axis`` (normalized here)."""
    axis = _as_f64(axis)
    angle = _as_f64(angle)
    n = np.linalg.norm(axis, axis=-1, keepdims=True)
    unit = axis / np.maximum(n, _EPS)
    half = 0.5 * angle[..., None]
    return np.concatenate([np.cos(half), np.sin(half) * unit], axis=-1)


def quat_to_axis_angle(q: ArrayLike) -> tuple[F64, F64]:
    """Return ``(axis, angle)`` with ``angle`` in ``[0, π]``; axis is x̂ for identity."""
    q = quat_canonical(q)
    vec = q[..., 1:]
    s = np.linalg.norm(vec, axis=-1)
    angle = 2.0 * np.arctan2(s, q[..., 0])
    safe = s > _EPS
    axis = np.where(
        safe[..., None], vec / np.maximum(s, _EPS)[..., None], np.array([1.0, 0.0, 0.0])
    )
    return axis, angle


def quat_to_rotmat(q: ArrayLike) -> F64:
    """Rotation matrix ``R`` with ``v_world = R @ v_body``."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty((*q.shape[:-1], 3, 3))
    r[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    r[..., 0, 1] = 2.0 * (x * y - w * z)
    r[..., 0, 2] = 2.0 * (x * z + w * y)
    r[..., 1, 0] = 2.0 * (x * y + w * z)
    r[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    r[..., 1, 2] = 2.0 * (y * z - w * x)
    r[..., 2, 0] = 2.0 * (x * z - w * y)
    r[..., 2, 1] = 2.0 * (y * z + w * x)
    r[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return r


def rotmat_to_quat(r: ArrayLike) -> F64:
    """Convert rotation matrices to canonical (``w >= 0``) unit quaternions.

    Uses Shepperd's method, choosing the numerically largest pivot per element.
    """
    r = _as_f64(r)
    m00, m11, m22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    trace = m00 + m11 + m22
    cands = np.stack([trace, m00, m11, m22], axis=-1)
    pivot = np.argmax(cands, axis=-1)

    # Candidate 0: w is largest.
    s0 = np.sqrt(np.maximum(1.0 + trace, _EPS)) * 2.0
    q0 = np.stack(
        [
            0.25 * s0,
            (r[..., 2, 1] - r[..., 1, 2]) / s0,
            (r[..., 0, 2] - r[..., 2, 0]) / s0,
            (r[..., 1, 0] - r[..., 0, 1]) / s0,
        ],
        axis=-1,
    )
    # Candidate 1: x is largest.
    s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, _EPS)) * 2.0
    q1 = np.stack(
        [
            (r[..., 2, 1] - r[..., 1, 2]) / s1,
            0.25 * s1,
            (r[..., 0, 1] + r[..., 1, 0]) / s1,
            (r[..., 0, 2] + r[..., 2, 0]) / s1,
        ],
        axis=-1,
    )
    # Candidate 2: y is largest.
    s2 = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, _EPS)) * 2.0
    q2 = np.stack(
        [
            (r[..., 0, 2] - r[..., 2, 0]) / s2,
            (r[..., 0, 1] + r[..., 1, 0]) / s2,
            0.25 * s2,
            (r[..., 1, 2] + r[..., 2, 1]) / s2,
        ],
        axis=-1,
    )
    # Candidate 3: z is largest.
    s3 = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, _EPS)) * 2.0
    q3 = np.stack(
        [
            (r[..., 1, 0] - r[..., 0, 1]) / s3,
            (r[..., 0, 2] + r[..., 2, 0]) / s3,
            (r[..., 1, 2] + r[..., 2, 1]) / s3,
            0.25 * s3,
        ],
        axis=-1,
    )
    all_q = np.stack([q0, q1, q2, q3], axis=-2)
    q = np.take_along_axis(all_q, pivot[..., None, None], axis=-2)[..., 0, :]
    return quat_canonical(q)


def quat_error(q: ArrayLike, q_target: ArrayLike) -> F64:
    """Body-frame error rotation ``q_e = q_target* ⊗ q`` (identity when aligned).

    Its rotation matrix equals ``R_target.T @ R``.
    """
    return quat_mul(quat_conj(q_target), q)


def rotation_error_matrix(q: ArrayLike, q_target: ArrayLike) -> F64:
    """``R_err = R_target.T @ R``: the attitude of the body expressed in the target frame."""
    return quat_to_rotmat(quat_error(q, q_target))


def geodesic_angle(q1: ArrayLike, q2: ArrayLike) -> F64:
    """Geodesic distance on SO(3) in rad, in ``[0, π]``; sign-invariant (q ≡ -q)."""
    e = quat_error(q1, q2)
    vec_norm = np.linalg.norm(e[..., 1:], axis=-1)
    return 2.0 * np.arctan2(vec_norm, np.abs(e[..., 0]))


def tilt_angle(q: ArrayLike) -> F64:
    """Angle in rad between body +z and world +z (0 = level, π = inverted)."""
    z_body_in_world = quat_rotate(q, np.array([0.0, 0.0, 1.0]))
    return np.arccos(np.clip(z_body_in_world[..., 2], -1.0, 1.0))


def gravity_in_body(q: ArrayLike) -> F64:
    """Unit gravity direction (world ``-z``) expressed in the body frame."""
    return quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0]))


def yaw_angle(q: ArrayLike) -> F64:
    """Heading of body +x projected on the world xy-plane, in rad (``atan2``).

    Only for human-facing reports and yaw *setpoints*; ill-defined near
    vertical body-x. Never use it inside reward or observation terms.
    """
    x_world = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    return np.arctan2(x_world[..., 1], x_world[..., 0])


def quat_from_yaw(yaw: ArrayLike) -> F64:
    """Pure rotation about world z."""
    yaw = _as_f64(yaw)
    return quat_from_axis_angle(np.broadcast_to([0.0, 0.0, 1.0], (*yaw.shape, 3)), yaw)


def random_quat(rng: np.random.Generator, size: int | None = None) -> F64:
    """Uniformly distributed random rotations (Shoemake's method), canonical sign."""
    shape = () if size is None else (size,)
    u1, u2, u3 = rng.random((3, *shape))
    a = np.sqrt(1.0 - u1)
    b = np.sqrt(u1)
    q = np.stack(
        [
            b * np.cos(2.0 * np.pi * u3),
            a * np.sin(2.0 * np.pi * u2),
            a * np.cos(2.0 * np.pi * u2),
            b * np.sin(2.0 * np.pi * u3),
        ],
        axis=-1,
    )
    return quat_canonical(q)


def random_tilted_quat(
    rng: np.random.Generator,
    max_tilt: float,
    size: int | None = None,
    min_tilt: float = 0.0,
    random_yaw: bool = True,
) -> F64:
    """Random attitude with tilt uniformly distributed over the spherical cap.

    The tilt (angle of body z from world z) is sampled so that the body z-axis
    is uniform on the cap ``min_tilt <= tilt <= max_tilt``. With
    ``max_tilt = π`` and ``random_yaw`` this is uniform on SO(3).
    """
    shape = () if size is None else (size,)
    cos_lo, cos_hi = np.cos(max_tilt), np.cos(min_tilt)
    cos_t = rng.uniform(cos_lo, cos_hi, shape)
    tilt = np.arccos(np.clip(cos_t, -1.0, 1.0))
    phi = rng.uniform(-np.pi, np.pi, shape)
    axis = np.stack([np.cos(phi), np.sin(phi), np.zeros(shape)], axis=-1)
    q_tilt = quat_from_axis_angle(axis, tilt)
    if random_yaw:
        q_yaw = quat_from_yaw(rng.uniform(-np.pi, np.pi, shape))
        return quat_canonical(quat_mul(q_tilt, q_yaw))
    return quat_canonical(q_tilt)


# --- Frame conversions for deployment (FLU/ENU <-> FRD/NED) -------------------

_Q_NED_ENU: F64 = rotmat_to_quat(R_NED_ENU)  # (0, √½, √½, 0)
_Q_FLU_FRD: F64 = rotmat_to_quat(R_FRD_FLU.T)  # (0, 1, 0, 0)


def vec_enu_to_ned(v: ArrayLike) -> F64:
    """World vector ENU → NED: ``(e, n, u) → (n, e, -u)``."""
    return _as_f64(v) @ R_NED_ENU.T


def vec_flu_to_frd(v: ArrayLike) -> F64:
    """Body vector (incl. angular rate) FLU → FRD: ``(x, y, z) → (x, -y, -z)``."""
    return _as_f64(v) @ R_FRD_FLU.T


def quat_enu_flu_to_ned_frd(q_enu_flu: ArrayLike) -> F64:
    """Convert attitude ``q`` (FLU body → ENU world) to (FRD body → NED world).

    ``R_ned_frd = R_NED_ENU @ R_enu_flu @ R_FLU_FRD``; result is canonical.
    """
    return quat_canonical(quat_mul(quat_mul(_Q_NED_ENU, q_enu_flu), _Q_FLU_FRD))


def quat_ned_frd_to_enu_flu(q_ned_frd: ArrayLike) -> F64:
    """Inverse of :func:`quat_enu_flu_to_ned_frd`."""
    return quat_canonical(
        quat_mul(quat_mul(quat_conj(_Q_NED_ENU), q_ned_frd), quat_conj(_Q_FLU_FRD))
    )
