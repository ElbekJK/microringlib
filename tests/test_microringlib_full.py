import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import microringlib.transfer as transfer

from microringlib import (
    Layer,
    RingGeometry,
    Coupler,
    ModeResult,
    TransmissionResult,
    solve_waveguide_modes,
    compute_group_index,
    compute_resonance_metrics,
    ring_circumference,
    single_waveguide,
    single_mrr_thru,
    single_mrr_add_drop,
    cascaded_mrrs_add_drop,
    compute_transmission,
    plot_transmission,
    plot_mode_profile,
)


class FakeModeResult:
    def __init__(self, wavelength, n_eff, field=None, polarization="TE"):
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.n_eff = np.asarray(n_eff, dtype=float)
        if field is None:
            field = np.ones((self.wavelength.size, 1, 8), dtype=np.complex128)
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


def test_public_api_types_exist():
    assert Layer is not None
    assert RingGeometry is not None
    assert Coupler is not None
    assert ModeResult is not None
    assert TransmissionResult is not None


def test_ring_circumference_circular():
    ring = RingGeometry(kind="circular", radius=10e-6)
    assert np.isclose(ring_circumference(ring), 2.0 * np.pi * 10e-6)


def test_ring_circumference_elliptical():
    ring = RingGeometry(kind="elliptical", a=12e-6, b=8e-6)
    h = ((12e-6 - 8e-6) ** 2) / ((12e-6 + 8e-6) ** 2)
    expected = np.pi * (12e-6 + 8e-6) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h)))
    assert np.isclose(ring_circumference(ring), expected)


def test_ring_circumference_racetrack():
    ring = RingGeometry(kind="racetrack", radius=10e-6, straight_length=20e-6)
    expected = 2.0 * np.pi * 10e-6 + 2.0 * 20e-6
    assert np.isclose(ring_circumference(ring), expected)


def test_ring_circumference_rejects_invalid_kind():
    with pytest.raises(ValueError):
        ring_circumference(RingGeometry(kind="invalid"))


def test_coupler_validate_lossless():
    c = Coupler(t=0.9 + 0j, kappa=np.sqrt(1.0 - 0.9**2))
    c.validate_lossless()


def test_coupler_validate_lossless_rejects_bad_values():
    c = Coupler(t=1.2 + 0j, kappa=0.0)
    with pytest.raises(ValueError):
        c.validate_lossless()


def test_compute_group_index_constant():
    wl = np.linspace(1.0, 2.0, 51)
    n_eff = np.full_like(wl, 2.4)
    ng = compute_group_index(wl, n_eff)
    assert np.allclose(ng, 2.4)


def test_compute_group_index_linear_dispersion():
    wl = np.linspace(1.0, 2.0, 51)
    n_eff = 2.0 - 0.1 * wl
    ng = compute_group_index(wl, n_eff)
    assert np.allclose(ng, 2.0, atol=1e-10)


def test_compute_group_index_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        compute_group_index(np.array([1.0, 2.0]), np.array([1.0]))


def test_compute_group_index_rejects_too_few_samples():
    with pytest.raises(ValueError):
        compute_group_index(np.array([1.0]), np.array([1.0]))


def test_mode_solver_scalar_wavelength(layers):
    mode = solve_waveguide_modes(1550e-9, layers, T=25.0, polarization="TE", num_modes=1)
    assert mode.wavelength.shape == (1,)
    assert mode.n_eff.shape == (1,)
    assert mode.field.ndim == 2
    assert np.isfinite(mode.n_eff).all()


def test_mode_solver_array_wavelengths(layers):
    wl = np.linspace(1548e-9, 1552e-9, 5)
    mode = solve_waveguide_modes(wl, layers, T=25.0, polarization="TM", num_modes=1)
    assert mode.wavelength.shape == (wl.size,)
    assert mode.n_eff.shape == (wl.size, 1)
    assert mode.field.shape[0] == wl.size
    assert mode.field.shape[1] == 1


def test_mode_solver_deterministic(layers):
    wl = np.linspace(1548e-9, 1552e-9, 5)
    m1 = solve_waveguide_modes(wl, layers, T=25.0, polarization="TE", num_modes=1)
    m2 = solve_waveguide_modes(wl, layers, T=25.0, polarization="TE", num_modes=1)
    assert np.allclose(m1.n_eff, m2.n_eff)
    assert np.allclose(m1.field, m2.field)


def test_mode_solver_rejects_invalid_inputs(layers):
    wl = np.linspace(1548e-9, 1552e-9, 5)

    with pytest.raises(ValueError):
        solve_waveguide_modes(wl, [], T=25.0)

    with pytest.raises(ValueError):
        solve_waveguide_modes(wl, layers, T=25.0, polarization="X")

    with pytest.raises(ValueError):
        solve_waveguide_modes(wl, layers, T=25.0, num_modes=0)

    with pytest.raises(ValueError):
        solve_waveguide_modes(-wl, layers, T=25.0)

    with pytest.raises(ValueError):
        solve_waveguide_modes(wl, layers, T=25.0, dx=0.0)


def test_single_waveguide_zero_length_is_unit_power(monkeypatch, wavelengths, layers):
    fake_neff = np.full((wavelengths.size, 1), 2.4, dtype=float)
    fake_mode = FakeModeResult(wavelengths, fake_neff)

    monkeypatch.setattr(transfer, "solve_waveguide_modes", lambda *args, **kwargs: fake_mode)

    out = single_waveguide(wavelengths, layers, T=25.0, polarization="TE", length=0.0)
    assert out.field.shape == wavelengths.shape
    assert out.power.shape == wavelengths.shape
    assert np.allclose(out.power, 1.0, atol=1e-12)


def test_single_waveguide_dispatch(monkeypatch, wavelengths, layers):
    expected = TransmissionResult(
        wavelength=wavelengths,
        field=np.ones_like(wavelengths, dtype=np.complex128),
        power=np.ones_like(wavelengths, dtype=float),
        metadata={"case": "waveguide"},
    )

    def fake_single_waveguide(*args, **kwargs):
        return expected

    monkeypatch.setattr(transfer, "single_waveguide", fake_single_waveguide)

    out = compute_transmission(
        case="waveguide",
        wavelengths=wavelengths,
        layers=layers,
        T=25.0,
        polarization="TE",
        length=10e-6,
    )
    assert out is expected


def test_single_mrr_thru_matches_manual_formula(monkeypatch, wavelengths, layers, ring):
    fake_neff = np.full((wavelengths.size, 1), 2.4, dtype=float)
    fake_mode = FakeModeResult(wavelengths, fake_neff)
    monkeypatch.setattr(transfer, "solve_waveguide_modes", lambda *args, **kwargs: fake_mode)

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
    alpha = np.zeros_like(wavelengths)
    A = np.exp(-alpha * L / 2.0)
    phi = 2.0 * np.pi * n_eff * L / wavelengths
    E_rt = A * np.exp(-1j * phi)
    expected = (t - E_rt) / (1.0 - t * E_rt)

    assert out.field.shape == wavelengths.shape
    assert np.allclose(out.field, expected)
    assert np.allclose(out.power, np.abs(expected) ** 2)


def test_single_mrr_thru_rejects_invalid_t():
    wl = np.linspace(1548e-9, 1552e-9, 11)
    ring = RingGeometry(kind="circular", radius=10e-6)
    layers = [
        Layer("silica", 2e-6, 1.45),
        Layer("silicon", 220e-9, 3.48),
        Layer("silica", 2e-6, 1.45),
    ]
    with pytest.raises(ValueError):
        single_mrr_thru(wl, ring, layers, T=25.0, polarization="TE", t=1.2)


def test_single_mrr_add_drop_shapes_and_nonnegative(monkeypatch, wavelengths, layers, ring):
    fake_neff = np.full((wavelengths.size, 1), 2.4, dtype=float)
    fake_mode = FakeModeResult(wavelengths, fake_neff)
    monkeypatch.setattr(transfer, "solve_waveguide_modes", lambda *args, **kwargs: fake_mode)

    t = 0.9
    kappa = np.sqrt(1.0 - t**2)

    out = single_mrr_add_drop(
        wavelengths,
        ring,
        layers,
        T=25.0,
        polarization="TE",
        t1=t,
        kappa1=kappa,
    )

    assert out.field.shape == (2, wavelengths.size)
    assert out.power.shape == (2, wavelengths.size)
    assert np.all(out.power >= 0.0)


def test_single_mrr_add_drop_rejects_bad_coupler(monkeypatch, wavelengths, layers, ring):
    fake_neff = np.full((wavelengths.size, 1), 2.4, dtype=float)
    fake_mode = FakeModeResult(wavelengths, fake_neff)
    monkeypatch.setattr(transfer, "solve_waveguide_modes", lambda *args, **kwargs: fake_mode)

    with pytest.raises(ValueError):
        single_mrr_add_drop(
            wavelengths,
            ring,
            layers,
            T=25.0,
            polarization="TE",
            t1=0.9,
            kappa1=0.9,
        )


def test_compute_transmission_dispatch_mrr_thru(monkeypatch, wavelengths, layers, ring):
    expected = TransmissionResult(
        wavelength=wavelengths,
        field=np.ones_like(wavelengths, dtype=np.complex128),
        power=np.ones_like(wavelengths, dtype=float),
        metadata={"case": "mrr_thru"},
    )

    monkeypatch.setattr(transfer, "single_mrr_thru", lambda *args, **kwargs: expected)

    out = compute_transmission(
        case="mrr_thru",
        wavelengths=wavelengths,
        resonator=ring,
        layers=layers,
        T=25.0,
        polarization="TE",
        coupling={"t": 0.9},
    )
    assert out is expected


def test_compute_transmission_dispatch_mrr_add_drop(monkeypatch, wavelengths, layers, ring):
    expected = TransmissionResult(
        wavelength=wavelengths,
        field=np.vstack(
            [
                np.ones_like(wavelengths, dtype=np.complex128),
                np.ones_like(wavelengths, dtype=np.complex128),
            ]
        ),
        power=np.vstack([np.ones_like(wavelengths), np.ones_like(wavelengths)]),
        metadata={"case": "mrr_add_drop"},
    )

    monkeypatch.setattr(transfer, "single_mrr_add_drop", lambda *args, **kwargs: expected)

    out = compute_transmission(
        case="mrr_add_drop",
        wavelengths=wavelengths,
        resonator=ring,
        layers=layers,
        T=25.0,
        polarization="TE",
        coupling={"t1": 0.9, "kappa1": np.sqrt(1.0 - 0.9**2)},
    )
    assert out is expected


def test_compute_transmission_dispatch_multi_mrr(monkeypatch, wavelengths):
    expected_bus = np.exp(-1j * np.zeros_like(wavelengths))
    expected_power = np.ones_like(wavelengths)

    def fake_cascade(*args, **kwargs):
        return expected_bus, expected_power, [expected_power]

    monkeypatch.setattr(transfer, "cascaded_mrrs_add_drop", fake_cascade)

    out = compute_transmission(
        case="multi_mrr",
        wavelengths=wavelengths,
        params_list=[{"resonator": None}],
    )
    assert isinstance(out, tuple)
    assert np.allclose(out[0], expected_bus)
    assert np.allclose(out[1], expected_power)


def test_compute_transmission_rejects_unknown_case(wavelengths):
    with pytest.raises(ValueError):
        compute_transmission(case="unknown", wavelengths=wavelengths)


def test_compute_transmission_rejects_negative_wavelengths(layers, ring):
    wl = np.linspace(1548e-9, 1552e-9, 11)
    with pytest.raises(ValueError):
        compute_transmission(
            case="waveguide",
            wavelengths=-wl,
            layers=layers,
            T=25.0,
            polarization="TE",
            length=10e-6,
        )


def test_compute_resonance_metrics_basic():
    wl = np.linspace(1.0, 2.0, 4001)
    T = 1.0 - 0.8 * (np.sin(2.0 * np.pi * wl / 0.2) ** 2)
    metrics = compute_resonance_metrics(wl, T)

    assert metrics["min_transmission"] <= metrics["max_transmission"]
    assert np.isfinite(metrics["fwhm"])
    assert np.isfinite(metrics["fsr"])
    assert np.isfinite(metrics["quality_factor"])
    assert np.isfinite(metrics["extinction_ratio_db"])
    assert np.isfinite(metrics["intensity_enhancement"])


def test_compute_resonance_metrics_rejects_bad_inputs():
    with pytest.raises(ValueError):
        compute_resonance_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError):
        compute_resonance_metrics(np.array([1.0]), np.array([1.0]))


def test_cascaded_mrrs_add_drop_with_bus_segment(monkeypatch, wavelengths):
    # Make the internal ring responses fully deterministic.
    def fake_single_mrr_add_drop(wl, resonator, layers, T, polarization, t1, kappa1, t2=None, kappa2=None):
        thru = np.full_like(wl, 0.5 + 0.0j, dtype=np.complex128)
        drop = np.full_like(wl, 0.25 + 0.0j, dtype=np.complex128)
        field = np.vstack([thru, drop])
        power = np.abs(field) ** 2
        return TransmissionResult(wl, field, power, {"case": "mrr_add_drop"})

    monkeypatch.setattr(transfer, "single_mrr_add_drop", fake_single_mrr_add_drop)

    params_list = [
        {
            "resonator": RingGeometry(kind="circular", radius=10e-6),
            "layers": [Layer("a", 1.0, 1.5)],
            "T": 25.0,
            "polarization": "TE",
            "t1": 0.9,
            "kappa1": np.sqrt(1.0 - 0.9**2),
        },
        {
            "resonator": RingGeometry(kind="circular", radius=10e-6),
            "layers": [Layer("a", 1.0, 1.5)],
            "T": 25.0,
            "polarization": "TE",
            "t1": 0.9,
            "kappa1": np.sqrt(1.0 - 0.9**2),
        },
    ]

    bus_segments = [
        {"length": 10e-6, "alpha": 0.0, "n_eff": 2.0},
    ]

    bus_field, bus_power, drop_powers = cascaded_mrrs_add_drop(
        wavelengths,
        params_list=params_list,
        bus_segments=bus_segments,
    )

    phi = 2.0 * np.pi * 2.0 * 10e-6 / wavelengths
    expected_bus = (0.5 * np.exp(-1j * phi)) * 0.5

    assert np.allclose(bus_field, expected_bus)
    assert np.allclose(bus_power, np.abs(expected_bus) ** 2)
    assert len(drop_powers) == 2
    assert np.allclose(drop_powers[0], np.full_like(wavelengths, 0.0625))
    assert np.allclose(drop_powers[1], np.full_like(wavelengths, 0.015625))


def test_plotting_smoke(tmp_path, wavelengths, layers):
    power = 1.0 - 0.3 * np.exp(-((wavelengths - wavelengths.mean()) / 0.05e-9) ** 2)

    plot_transmission(wavelengths, power, labels=["Through"])
    plt.savefig(tmp_path / "through.png", dpi=150)
    plt.close("all")

    mode = solve_waveguide_modes(1550e-9, layers, T=25.0, polarization="TE", num_modes=1)
    plot_mode_profile(mode.field, layers, T=25.0, labels=["Mode 0"])
    plt.savefig(tmp_path / "mode.png", dpi=150)
    plt.close("all")

    assert (tmp_path / "through.png").exists()
    assert (tmp_path / "mode.png").exists()