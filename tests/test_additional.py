# tests/test_additional.py
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import microringlib.transfer as tr
from microringlib import (
    Layer,
    RingGeometry,
    Coupler,
    compute_group_index,
    compute_resonance_metrics,
    ring_circumference,
    single_mrr_thru,
    compute_transmission,
    plot_transmission,
    plot_mode_profile,
)


class FakeModeResult:
    def __init__(self, wavelength, n_eff, field=None, polarization="TE"):
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.n_eff = np.asarray(n_eff, dtype=float)
        if field is None:
            field = np.ones((self.wavelength.size, 1, 5), dtype=np.complex128)
        self.field = field
        self.polarization = polarization


@pytest.fixture
def wavelengths():
    return np.linspace(1548e-9, 1552e-9, 401)


@pytest.fixture
def layers():
    return [
        Layer("silica", 2e-6, 1.45, dn_dT=1e-5, alpha=0.0),
        Layer("silicon", 220e-9, 3.48, dn_dT=1.8e-4, alpha=0.0),
        Layer("silica", 2e-6, 1.45, dn_dT=1e-5, alpha=0.0),
    ]


@pytest.fixture
def ring():
    return RingGeometry(kind="circular", radius=10e-6)


def test_ring_circumference_circular():
    r = 10e-6
    ring = RingGeometry(kind="circular", radius=r)
    assert np.isclose(ring_circumference(ring), 2 * np.pi * r)


def test_ring_circumference_racetrack():
    ring = RingGeometry(kind="racetrack", radius=10e-6, straight_length=20e-6)
    expected = 2 * np.pi * 10e-6 + 2 * 20e-6
    assert np.isclose(ring_circumference(ring), expected)


def test_group_index_constant_neff_is_constant():
    wl = np.linspace(1500e-9, 1600e-9, 100)
    n_eff = np.full_like(wl, 2.4)
    ng = compute_group_index(wl, n_eff)
    assert np.allclose(ng, 2.4)


def test_compute_resonance_metrics_simple_sinusoid():
    wl = np.linspace(1.0, 2.0, 4001)
    T = 1.0 - 0.8 * np.sin(2 * np.pi * wl / 0.2) ** 2
    m = compute_resonance_metrics(wl, T)
    assert m["min_transmission"] <= m["max_transmission"]
    assert np.isfinite(m["fsr"])


def test_single_mrr_thru_matches_manual_formula(monkeypatch, wavelengths, layers, ring):
    # Make the mode solver deterministic and very fast.
    fake_neff = np.full((wavelengths.size, 1), 2.4, dtype=float)
    fake_mode = FakeModeResult(wavelengths, fake_neff)

    monkeypatch.setattr(tr, "solve_waveguide_modes", lambda *args, **kwargs: fake_mode)

    t = 0.9
    out = single_mrr_thru(
        wavelengths,
        ring,
        layers,
        T=25.0,
        polarization="TE",
        t=t,
    )

    L = ring_circumference(ring)
    n_eff = 2.4
    alpha = 0.0
    A = np.exp(-alpha * L / 2.0)
    phi = 2.0 * np.pi * n_eff * L / wavelengths
    E_rt = A * np.exp(-1j * phi)
    expected = (t - E_rt) / (1.0 - t * E_rt)

    assert np.allclose(out.field, expected)
    assert np.allclose(out.power, np.abs(expected) ** 2)


def test_compute_transmission_dispatch_waveguide(monkeypatch, wavelengths, layers):
    fake_neff = np.full((wavelengths.size, 1), 2.4, dtype=float)
    fake_mode = FakeModeResult(wavelengths, fake_neff)

    monkeypatch.setattr(tr, "solve_waveguide_modes", lambda *args, **kwargs: fake_mode)

    out = compute_transmission(
        case="waveguide",
        wavelengths=wavelengths,
        layers=layers,
        T=25.0,
        polarization="TE",
        length=50e-6,
    )
    assert out.field.shape == wavelengths.shape
    assert out.power.shape == wavelengths.shape
    assert np.all(out.power <= 1.0 + 1e-12)


def test_compute_transmission_rejects_unknown_case(wavelengths):
    with pytest.raises(ValueError):
        compute_transmission(case="unknown", wavelengths=wavelengths)


def test_single_mrr_thru_rejects_invalid_coupling(wavelengths, layers, ring):
    with pytest.raises(ValueError):
        single_mrr_thru(
            wavelengths,
            ring,
            layers,
            T=25.0,
            polarization="TE",
            t=1.2,
        )


def test_plot_functions_do_not_crash(tmp_path, wavelengths, layers):
    power = 1.0 - 0.2 * np.exp(-((wavelengths - wavelengths.mean()) / 0.2e-9) ** 2)

    plot_transmission(wavelengths, power, labels=["Through"])
    plt.savefig(tmp_path / "plot_transmission.png", dpi=150)
    plt.close("all")

    mode_field = np.vstack(
        [
            np.sin(np.linspace(0, np.pi, 80)),
            np.cos(np.linspace(0, np.pi, 80)),
        ]
    )
    plot_mode_profile(mode_field, layers, T=25.0, labels=["Mode 1", "Mode 2"])
    plt.savefig(tmp_path / "plot_mode_profile.png", dpi=150)
    plt.close("all")

    assert (tmp_path / "plot_transmission.png").exists()
    assert (tmp_path / "plot_mode_profile.png").exists()