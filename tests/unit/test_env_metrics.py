"""T14 (ADR-0006 §9): the G4-G6 metric functions, on signals with known answers."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.fast

RATE = 100.0
AMP = 0.1


def _sine(freq_hz: float, seconds: float = 5.0, offset: float = 0.3) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    col = offset + AMP * np.sin(2.0 * np.pi * freq_hz * t)
    return np.tile(col[:, None], (1, 4))  # (n, 4 motors)


def test_hf_rms_is_zero_for_a_constant(api: ModuleType) -> None:
    assert api.hf_rms(np.full((500, 4), 0.3), RATE) == pytest.approx(0.0, abs=1e-12)


def test_hf_rms_is_zero_for_a_linear_ramp(api: ModuleType) -> None:
    ramp = np.tile(np.linspace(0.2, 0.4, 500)[:, None], (1, 4))
    assert api.hf_rms(ramp, RATE) == pytest.approx(0.0, abs=1e-9)  # the detrend removes it


@pytest.mark.parametrize("freq", [1.0, 5.0, 10.0, 19.0])
def test_hf_rms_ignores_sinusoids_below_the_cutoff(api: ModuleType, freq: float) -> None:
    # finite-window leakage after the linear detrend leaves ~1e-3 A at worst; the HF content of a
    # 30 Hz tone of the same amplitude is 0.7 A, so 2 % of A separates the cases by 35x
    assert api.hf_rms(_sine(freq), RATE) < 0.02 * AMP


@pytest.mark.parametrize("freq", [30.0, 40.0, 49.0])
def test_hf_rms_of_a_fast_sinusoid_is_amplitude_over_sqrt2(api: ModuleType, freq: float) -> None:
    assert api.hf_rms(_sine(freq), RATE) == pytest.approx(AMP / np.sqrt(2.0), rel=0.01)


def test_hf_rms_scales_with_amplitude_and_pools_motors(api: ModuleType) -> None:
    u = _sine(30.0)
    u[:, 2:] = 0.3  # two motors constant: the RMS over time AND motors drops by sqrt(2)
    assert api.hf_rms(u, RATE) == pytest.approx(AMP / np.sqrt(2.0) / np.sqrt(2.0), rel=0.01)
    assert api.hf_rms(0.5 * (_sine(30.0) - 0.3) + 0.3, RATE) == pytest.approx(
        0.5 * AMP / np.sqrt(2.0), rel=0.01
    )


def test_hf_rms_skips_the_first_second_by_default(api: ModuleType) -> None:
    u = np.full((600, 4), 0.3)
    u[:100] += 0.2 * np.sin(2.0 * np.pi * 30.0 * np.arange(100) / RATE)[:, None]
    assert api.hf_rms(u, RATE) == pytest.approx(0.0, abs=1e-6)  # the burst is inside t < 1 s
    assert api.hf_rms(u, RATE, t_start=0.0) > 0.05


def test_hf_rms_cutoff_is_a_parameter(api: ModuleType) -> None:
    u = _sine(10.0)
    assert api.hf_rms(u, RATE, f_cut=20.0) < 0.02 * AMP
    assert api.hf_rms(u, RATE, f_cut=5.0) == pytest.approx(AMP / np.sqrt(2.0), rel=0.01)


def test_hf_energy_fraction_is_nan_below_the_amplitude_floor(api: ModuleType) -> None:
    """Report-only metric: NaN when the detrended AC RMS is < 1e-3 (the PID's noise-driven 0.71)."""
    rng = np.random.default_rng(0)
    quiet = 0.3 + 1e-4 * rng.standard_normal((500, 4))
    assert np.isnan(api.hf_energy_fraction(quiet, RATE))
    assert np.isnan(api.hf_energy_fraction(np.full((500, 4), 0.3), RATE))
    loud = _sine(30.0)
    assert api.hf_energy_fraction(loud, RATE) == pytest.approx(1.0, abs=0.02)
    assert api.hf_energy_fraction(_sine(5.0), RATE) < 0.02


# --- success_recovery -----------------------------------------------------------------------------
DT = 0.01
FIVE_DEG = np.deg2rad(5.0)


def _err(recover_at_s: float, n: int = 400) -> np.ndarray:
    e = np.full(n, 0.5)
    e[round(recover_at_s / DT) :] = 0.01
    return e


def test_success_recovery_threshold_is_inclusive(api: ModuleType) -> None:
    assert api.success_recovery(_err(1.5), DT, 1.5, crashed=False)
    assert not api.success_recovery(_err(1.5 + DT), DT, 1.5, crashed=False)
    assert api.success_recovery(_err(1.0), DT, 1.5, crashed=False)


def test_success_recovery_requires_staying_below_to_the_end(api: ModuleType) -> None:
    e = _err(1.0)
    e[300] = 0.2  # leaves the 5 deg ball again at t = 3 s
    assert not api.success_recovery(e, DT, 2.5, crashed=False)
    e[300:] = 0.01
    e[399] = FIVE_DEG * 1.01
    assert not api.success_recovery(e, DT, 2.5, crashed=False)


def test_success_recovery_a_crash_is_never_a_success(api: ModuleType) -> None:
    assert not api.success_recovery(_err(0.5), DT, 2.5, crashed=True)


def test_success_recovery_never_recovering_or_already_recovered(api: ModuleType) -> None:
    assert not api.success_recovery(np.full(400, 0.3), DT, 2.5, crashed=False)
    assert api.success_recovery(np.full(400, 0.01), DT, 0.0, crashed=False)  # recovered at t = 0
    assert not api.success_recovery(np.full(400, 0.01), DT, 2.5, crashed=True)


def test_success_recovery_is_time_to_recover_le_t_max(api: ModuleType) -> None:
    """Same definition as ``time_to_recover`` (ADR: 'the existing function already requires...')."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        k = int(rng.integers(0, 400))
        e = _err(k * DT)
        t = api.time_to_recover(e, DT)
        for t_max in (0.5, 1.5, 2.5):
            assert api.success_recovery(e, DT, t_max, crashed=False) == bool(t <= t_max)
