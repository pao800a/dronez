"""``quat_rotate`` scalar fast path and ``_cross3`` agree with the batched / ``np.cross`` forms."""

from __future__ import annotations

import numpy as np
import pytest

from dronez.utils import rotations

pytestmark = pytest.mark.fast


def test_cross3_matches_numpy_cross() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 5, 3))
    np.testing.assert_allclose(rotations._cross3(a, b), np.cross(a, b), atol=1e-14)
    np.testing.assert_allclose(rotations._cross3(a[0], b[0]), np.cross(a[0], b[0]), atol=1e-14)


def test_quat_rotate_scalar_path_equals_batched_path() -> None:
    rng = np.random.default_rng(1)
    q = rotations.random_quat(rng, 20)
    v = rng.normal(size=(20, 3))
    batched = rotations.quat_rotate(q, v)
    for i in range(20):
        np.testing.assert_allclose(rotations.quat_rotate(q[i], v[i]), batched[i], atol=1e-13)
        np.testing.assert_allclose(rotations.quat_to_rotmat(q[i]) @ v[i], batched[i], atol=1e-12)
