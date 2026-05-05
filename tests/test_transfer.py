import numpy as np
import pytest

from microringlib import (
    compute_transmission,
    single_waveguide,
    single_mrr_thru,
    single_mrr_add_drop,
    compute_resonance_metrics,
    RingGeometry,
)
from microringlib.transfer import propagation_phase


def test_propagation_phase_valid():
    phase = propagation_phase(np.array([2.4, 2.4]), np.array([1550e-9, 1560e-9]), 10e-6)
    assert phase.shape == (2,)
    assert np.all(np.isfinite(phase))


def test_propagation_phase_rejects_negative_length():
    with pytest.raises(ValueError):
        propagation_phase(np.array([2.4]), np.array([1550e-9]), -1.0)


def test_single_waveguide_zero_length_has_unit_power(wavelengths, layers):
    out = single_waveguide(wavelengths, layers, T=30.0, polarization="TE", length=0.0)
    assert np.allclose(out.power, np.ones_like(wavelengths), atol=1e-8)


def test_single_waveguide_lossy_does_not_exceed_one(wavelengths, layers):
    out = single_waveguide(wavelengths, layers, T=30.0, polarization="TE", length=50e-6)
    assert np.all(out.power <= 1.0 + 1e-6)


def test_single_waveguide_rejects_negative_length(wavelengths, layers):
    with pytest.raises(ValueError):
        single_waveguide(wavelengths, layers, T=30.0, polarization="TE", length=-1.0)


def test_single_mrr_thru_shape_and_nonnegative(wavelengths, layers, ring):
    out = single_mrr_thru(wavelengths, ring, layers, T=30.0, polarization="TE", t=0.9)
    assert out.field.shape == wavelengths.shape
    assert out.power.shape == wavelengths.shape
    assert np.all(out.power >= 0)


def test_single_mrr_add_drop_shape_and_nonnegative(wavelengths, layers, ring):
    kappa = np.sqrt(1 - 0.9**2)
    out = single_mrr_add_drop(
        wavelengths, ring, layers, T=30.0, polarization="TE", t1=0.9, kappa1=kappa
    )
    assert out.field.shape == (2, wavelengths.size)
    assert out.power.shape == (2, wavelengths.size)
    assert np.all(out.power >= 0)


def test_single_mrr_add_drop_rejects_invalid_coupler(wavelengths, layers, ring):
    with pytest.raises(ValueError):
        single_mrr_add_drop(
            wavelengths, ring, layers, T=30.0, polarization="TE", t1=0.9, kappa1=0.9
        )


def test_compute_transmission_dispatch_waveguide(wavelengths, layers):
    out = compute_transmission(
        case="waveguide",
        wavelengths=wavelengths,
        layers=layers,
        T=30.0,
        polarization="TE",
        length=10e-6,
    )
    assert out.power.shape == wavelengths.shape


def test_compute_transmission_dispatch_ring(wavelengths, layers, ring):
    out = compute_transmission(
        case="mrr_thru",
        wavelengths=wavelengths,
        resonator=ring,
        layers=layers,
        T=30.0,
        polarization="TE",
        coupling={"t": 0.9},
    )
    assert out.power.shape == wavelengths.shape


def test_compute_transmission_rejects_unknown_case(wavelengths):
    with pytest.raises(ValueError):
        compute_transmission(case="unknown", wavelengths=wavelengths)


def test_resonance_metrics_basic():
    wl = np.linspace(1.0, 2.0, 4001)
    T = 1.0 - 0.8 * (np.sin(2 * np.pi * wl / 0.2) ** 2)
    metrics = compute_resonance_metrics(wl, T)
    assert metrics["min_transmission"] <= metrics["max_transmission"]
    assert np.isfinite(metrics["fsr"])