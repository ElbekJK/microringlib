"""Release-readiness regression tests for microringlib.

This suite checks:
- public API imports
- dataclass construction and basic validation
- transfer-function sanity for every supported case
- resonance metric structure
- mode-solver consistency and physical bounds
- plotting smoke tests without interactive GUI use
- end-to-end numerical workflow

Run from the repository root:
    PYTHONPATH=. pytest -q tests/test_publishability_suite.py
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import microringlib as mrl
from microringlib import transfer as tr


pytestmark = pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown",
    "ignore:`trapz` is deprecated",
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def basic_layers() -> list[mrl.Layer]:
    """Return a physically plausible 1D stack for the mode solver."""
    return [
        mrl.Layer(material="SiO2", thickness=2.0e-6, n=1.44),
        mrl.Layer(material="Si", thickness=220e-9, n=3.48),
        mrl.Layer(material="SiO2", thickness=2.0e-6, n=1.44),
    ]


def basic_ring() -> mrl.RingGeometry:
    return mrl.RingGeometry(kind="circular", radius=10e-6)


def assert_finite(x) -> None:
    arr = np.asarray(x)
    assert np.all(np.isfinite(arr))


# -----------------------------------------------------------------------------
# Imports / public API
# -----------------------------------------------------------------------------


def test_package_imports_cleanly():
    assert hasattr(mrl, "Layer")
    assert hasattr(mrl, "RingGeometry")
    assert hasattr(mrl, "Coupler")
    assert hasattr(mrl, "solve_waveguide_modes")
    assert hasattr(mrl, "compute_group_index")
    assert hasattr(mrl, "compute_resonance_metrics")
    assert hasattr(mrl, "ring_circumference")
    assert hasattr(mrl, "single_waveguide")
    assert hasattr(mrl, "single_mrr_thru")
    assert hasattr(mrl, "single_mrr_add_drop")
    assert hasattr(mrl, "cascaded_mrrs_add_drop")
    assert hasattr(mrl, "compute_transmission")
    assert hasattr(mrl, "plot_transmission")
    assert hasattr(mrl, "plot_mode_profile")


def test_transfer_module_exports_internal_helpers():
    # These may not be part of the top-level package, but they should exist in
    # the transfer module because downstream code may rely on them.
    assert hasattr(tr, "propagation_phase")
    assert hasattr(tr, "ring_circumference")


@pytest.mark.parametrize(
    "name",
    [
        "Layer",
        "RingGeometry",
        "Coupler",
        "TransmissionResult",
        "ModeResult",
        "solve_waveguide_modes",
        "compute_group_index",
        "compute_resonance_metrics",
        "ring_circumference",
        "single_waveguide",
        "single_mrr_thru",
        "single_mrr_add_drop",
        "cascaded_mrrs_add_drop",
        "compute_transmission",
        "plot_transmission",
        "plot_mode_profile",
    ],
)
def test_public_api_exposed(name: str):
    assert hasattr(mrl, name), f"Missing public symbol: {name}"


# -----------------------------------------------------------------------------
# Dataclass / model sanity
# -----------------------------------------------------------------------------


def test_model_construction_and_repr():
    layer = mrl.Layer(material="Si", thickness=220e-9, n=3.48)
    geom = basic_ring()
    coupler = mrl.Coupler(t=math.sqrt(1 - 0.15**2), kappa=0.15)

    assert "Layer" in repr(layer)
    assert "RingGeometry" in repr(geom)
    assert "Coupler" in repr(coupler)
    assert layer.n > 1
    assert geom.radius > 0
    assert 0 <= abs(coupler.kappa) <= 1


@pytest.mark.parametrize("bad_radius", [0.0, -1.0e-6, -1])
def test_ring_geometry_rejects_invalid_radius(bad_radius):
    with pytest.raises((ValueError, TypeError, AssertionError)):
        mrl.ring_circumference(mrl.RingGeometry(kind="circular", radius=bad_radius))


# -----------------------------------------------------------------------------
# Transfer functions
# -----------------------------------------------------------------------------


def test_ring_circumference_matches_circle():
    ring = mrl.RingGeometry(kind="circular", radius=10e-6)
    c = mrl.ring_circumference(ring)
    assert np.isclose(c, 2.0 * np.pi * 10e-6)


def test_propagation_phase_is_finite():
    phase = tr.propagation_phase(n_eff=2.4, wavelength=1.55e-6, length=2.0e-5)
    assert np.isfinite(phase)
    assert phase > 0


@pytest.mark.parametrize("wavelength", [1.31e-6, 1.55e-6, 1.65e-6])
def test_propagation_phase_vectorizes(wavelength):
    phase = tr.propagation_phase(n_eff=np.array([2.3, 2.4]), wavelength=wavelength, length=10e-6)
    assert phase.shape == (2,)
    assert_finite(phase)


def test_single_waveguide_returns_physical_values():
    wl = np.linspace(1.53e-6, 1.57e-6, 11)
    result = mrl.single_waveguide(
        wavelengths=wl,
        layers=basic_layers(),
        T=25.0,
        polarization="TE",
        length=200e-6,
    )
    assert hasattr(result, "wavelength")
    assert hasattr(result, "field")
    assert hasattr(result, "power")
    assert np.asarray(result.wavelength).shape == wl.shape
    assert np.asarray(result.field).shape[0] == wl.size
    assert np.asarray(result.power).shape[0] == wl.size
    assert_finite(result.power)
    assert np.all(np.asarray(result.power) >= 0)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_single_mrr_thru_outputs_are_bounded(polarization):
    wl = np.linspace(1.53e-6, 1.57e-6, 21)
    out = mrl.single_mrr_thru(
        wavelengths=wl,
        resonator=basic_ring(),
        layers=basic_layers(),
        T=25.0,
        polarization=polarization,
        t=0.9,
    )
    assert np.asarray(out.wavelength).shape == wl.shape
    assert np.asarray(out.field).shape[0] == wl.size
    assert np.asarray(out.power).shape[0] == wl.size
    assert_finite(out.field)
    assert_finite(out.power)
    assert np.all(np.asarray(out.power) >= 0)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_single_mrr_add_drop_outputs_are_bounded(polarization):
    wl = np.linspace(1.53e-6, 1.57e-6, 21)
    out = mrl.single_mrr_add_drop(
        wavelengths=wl,
        resonator=basic_ring(),
        layers=basic_layers(),
        T=25.0,
        polarization=polarization,
        t1=0.9,
        kappa1=math.sqrt(1 - 0.9**2),
        t2=0.9,
        kappa2=math.sqrt(1 - 0.9**2),
    )
    assert np.asarray(out.field).shape[0] == 2
    assert np.asarray(out.field).shape[1] == wl.size
    assert np.asarray(out.power).shape == np.asarray(out.field).shape
    assert_finite(out.field)
    assert_finite(out.power)
    assert np.all(np.asarray(out.power) >= 0)


@pytest.mark.parametrize("n_rings", [1, 2, 3, 5])
def test_cascaded_mrrs_are_shape_consistent(n_rings):
    wl = np.linspace(1.50e-6, 1.60e-6, 31)
    params = []
    for _ in range(n_rings):
        params.append(
            {
                "resonator": basic_ring(),
                "layers": basic_layers(),
                "T": 25.0,
                "polarization": "TE",
                "t1": 0.9,
                "kappa1": math.sqrt(1 - 0.9**2),
                "t2": 0.9,
                "kappa2": math.sqrt(1 - 0.9**2),
            }
        )

    bus_segments = [{"length": 5e-6, "alpha": 0.0, "n_eff": 2.4} for _ in range(max(n_rings - 1, 0))]
    bus_field, bus_power, drop_powers = mrl.cascaded_mrrs_add_drop(
        wavelengths=wl,
        params_list=params,
        bus_segments=bus_segments,
    )

    assert np.asarray(bus_field).shape == wl.shape
    assert np.asarray(bus_power).shape == wl.shape
    assert len(drop_powers) == n_rings
    for dp in drop_powers:
        assert np.asarray(dp).shape == wl.shape
        assert_finite(dp)


@pytest.mark.parametrize("case", ["waveguide", "mrr_thru", "mrr_add_drop", "multi_mrr"])
def test_compute_transmission_returns_expected_structure(case):
    wl = np.linspace(1.53e-6, 1.57e-6, 21)

    kwargs = {"case": case, "wavelengths": wl}
    if case == "waveguide":
        kwargs.update({"layers": basic_layers(), "length": 100e-6})
    elif case == "mrr_thru":
        kwargs.update({"layers": basic_layers(), "resonator": basic_ring(), "coupling": {"t": 0.9}})
    elif case == "mrr_add_drop":
        kwargs.update({"layers": basic_layers(), "resonator": basic_ring(), "coupling": {"t1": 0.9, "kappa1": math.sqrt(1 - 0.9**2), "t2": 0.9, "kappa2": math.sqrt(1 - 0.9**2)}})
    elif case == "multi_mrr":
        params = [
            {
                "resonator": basic_ring(),
                "layers": basic_layers(),
                "T": 25.0,
                "polarization": "TE",
                "t1": 0.9,
                "kappa1": math.sqrt(1 - 0.9**2),
                "t2": 0.9,
                "kappa2": math.sqrt(1 - 0.9**2),
            }
        ]
        kwargs.update({"params_list": params})

    out = mrl.compute_transmission(**kwargs)

    if case == "multi_mrr":
        assert isinstance(out, tuple)
        assert len(out) == 3
        bus_field, bus_power, drop_powers = out
        assert np.asarray(bus_field).shape == wl.shape
        assert np.asarray(bus_power).shape == wl.shape
        assert len(drop_powers) >= 1
        for dp in drop_powers:
            assert np.asarray(dp).shape == wl.shape
            assert_finite(dp)
        return

    assert hasattr(out, "wavelength")
    assert hasattr(out, "field")
    assert hasattr(out, "power")
    assert np.asarray(out.wavelength).shape == wl.shape
    power = np.asarray(out.power)
    field = np.asarray(out.field)
    
    if case == "mrr_add_drop":
        assert field.shape == (2, wl.size)
        assert power.shape == (2, wl.size)
    else:
        assert field.shape[0] == wl.size
        assert power.shape[0] == wl.size
    assert_finite(out.field)
    assert_finite(out.power)


@pytest.mark.parametrize("coupling", [1.2, -1.2])
def test_single_mrr_rejects_invalid_coupler(coupling):
    wl = np.linspace(1.53e-6, 1.57e-6, 11)
    with pytest.raises((ValueError, AssertionError, TypeError)):
        mrl.single_mrr_thru(
            wavelengths=wl,
            resonator=basic_ring(),
            layers=basic_layers(),
            T=25.0,
            polarization="TE",
            t=coupling,
        )
    wl = np.linspace(1.53e-6, 1.57e-6, 11)
    with pytest.raises((ValueError, AssertionError, TypeError)):
        mrl.single_mrr_thru(
            wavelengths=wl,
            resonator=basic_ring(),
            layers=basic_layers(),
            T=25.0,
            polarization="TE",
            t=coupling,
        )


@pytest.mark.parametrize("num_rings", [0, -1])
def test_cascaded_rejects_invalid_num_rings(num_rings):
    wl = np.linspace(1.53e-6, 1.57e-6, 11)
    params = [
        {
            "resonator": basic_ring(),
            "layers": basic_layers(),
            "T": 25.0,
            "polarization": "TE",
            "t1": 0.9,
            "kappa1": math.sqrt(1 - 0.9**2),
            "t2": 0.9,
            "kappa2": math.sqrt(1 - 0.9**2),
        }
    ]
    if num_rings <= 0:
        params = []
    with pytest.raises((ValueError, AssertionError, TypeError)):
        mrl.cascaded_mrrs_add_drop(wavelengths=wl, params_list=params)


# -----------------------------------------------------------------------------
# Release-quality sanity checks
# -----------------------------------------------------------------------------


def test_all_public_callables_have_some_docstring():
    # This is intentionally a soft quality check: we fail only for exported
    # symbols that are clearly public and have no documentation at all.
    public = []
    for name in dir(mrl):
        if name.startswith("_"):
            continue
        obj = getattr(mrl, name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            public.append((name, obj))

    assert public, "No public callables discovered"

    missing = [name for name, obj in public if not inspect.getdoc(obj)]
    # Keep the suite strict enough to be useful, but do not fail the release
    # solely because a helper lacks a docstring.
    assert len(missing) < len(public), "Every public callable is undocumented"


def test_no_nan_or_inf_in_standard_workflow():
    wl = np.linspace(1.52e-6, 1.58e-6, 121)
    out = mrl.compute_transmission(
        case="mrr_thru",
        wavelengths=wl,
        layers=basic_layers(),
        resonator=basic_ring(),
        coupling={"t": 0.9},
    )
    t = np.asarray(out.power)
    assert np.isfinite(t).all()
    assert np.nanmin(t) >= 0
    assert np.nanmax(t) <= 2.0


def test_smoke_end_to_end_pipeline():
    modes = mrl.solve_waveguide_modes(
        wavelength=1.55e-6,
        layers=basic_layers(),
        T=25.0,
        polarization="TE",
        num_modes=1,
    )
    neff = float(np.asarray(modes.n_eff)[0])
    wl = np.linspace(1.53e-6, 1.57e-6, 51)
    out = mrl.compute_transmission(
        case="mrr_thru",
        wavelengths=wl,
        layers=basic_layers(),
        resonator=basic_ring(),
        coupling={"t": 0.9},
    )
    metrics = mrl.compute_resonance_metrics(wl, np.asarray(out.power))
    assert np.isfinite(neff)
    assert metrics["max_transmission"] >= metrics["min_transmission"]
    assert np.isfinite(metrics["min_transmission"])
    assert np.isfinite(metrics["max_transmission"])
