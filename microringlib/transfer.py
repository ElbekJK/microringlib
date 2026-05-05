from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import numpy as np

from .models import Coupler, Layer, RingGeometry, TransmissionResult
from .modes import solve_waveguide_modes, compute_group_index

ScalarOrArray = Union[float, np.ndarray]
AlphaType = Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]


def _as_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def ring_circumference(ring: RingGeometry) -> float:
    """Return the optical path length of a ring resonator."""
    if ring.kind == "circular":
        if ring.radius is None or ring.radius <= 0:
            raise ValueError("circular ring requires a positive radius")
        return float(2.0 * np.pi * ring.radius)

    if ring.kind == "elliptical":
        if ring.a is None or ring.b is None or ring.a <= 0 or ring.b <= 0:
            raise ValueError("elliptical ring requires positive a and b")
        a = float(max(ring.a, ring.b))
        b = float(min(ring.a, ring.b))
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return float(np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h))))

    if ring.kind == "racetrack":
        if ring.radius is None or ring.radius <= 0 or ring.straight_length is None or ring.straight_length < 0:
            raise ValueError("racetrack ring requires positive radius and non-negative straight_length")
        return float(2.0 * np.pi * ring.radius + 2.0 * ring.straight_length)

    raise ValueError(f"Unsupported ring kind: {ring.kind!r}")


def ring_fsr(wavelength: Any, n_g: Any, ring: RingGeometry) -> np.ndarray:
    """
    Estimate the free spectral range:
        FSR ≈ λ² / (n_g * L)
    """
    wl = _as_array(wavelength)
    ng = np.asarray(n_g, dtype=float)
    if np.any(wl <= 0):
        raise ValueError("wavelength must be positive")
    if np.any(ng <= 0):
        raise ValueError("group index must be positive")
    L = ring_circumference(ring)
    return wl**2 / (ng * L)

def group_index_from_modes(wavelengths: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    """
    Compute group index from effective index dispersion.

    n_g = n_eff - λ * (dn_eff/dλ)
    """
    wl = np.asarray(wavelengths, dtype=float)
    neff = np.asarray(n_eff, dtype=float)

    if wl.ndim != 1 or neff.ndim != 1:
        raise ValueError("wavelengths and n_eff must be 1D arrays")
    if wl.size < 3:
        raise ValueError("need at least 3 points for derivative")

    # مرکزی difference derivative
    dneff_dlambda = np.gradient(neff, wl)

    return neff - wl * dneff_dlambda

def propagation_phase(n_eff: Any, wavelength: Any, length: float) -> np.ndarray:
    """Return the propagation phase k0 * n_eff * length."""
    if length < 0:
        raise ValueError("length must be non-negative")
    neff = np.asarray(n_eff, dtype=float)
    wl = np.asarray(wavelength, dtype=float)
    if np.any(wl <= 0):
        raise ValueError("wavelength must be positive")
    return 2.0 * np.pi * neff * float(length) / wl


def _validate_wavelengths(wavelengths: Any) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=float)
    if np.any(wl <= 0):
        raise ValueError("wavelengths must be positive")
    return wl

def _layer_boundaries(layers: list[Layer]) -> np.ndarray:
    return np.cumsum([0.0] + [float(layer.thickness) for layer in layers])


def _modal_overlap_factors(layers: list[Layer], mode_result: Any) -> np.ndarray:
    x = getattr(mode_result, "x", None)
    if x is None:
        # Backward-compatible fallback for tests or external mode solvers that
        # provide n_eff but no field grid. Users seeking strict physics-first
        # modal loss should pass overlap_factors or a ModeResult with x/field.
        thickness = np.asarray([layer.thickness for layer in layers], dtype=float)
        if np.any(thickness <= 0):
            raise ValueError("layer thicknesses must be positive")
        return thickness / thickness.sum()
    x = np.asarray(x, dtype=float)
    field = np.asarray(mode_result.field)
    if field.ndim == 3:
        intensity = np.mean(np.abs(field[:, 0, :]) ** 2, axis=0)
    elif field.ndim == 2:
        intensity = np.abs(field[0, :]) ** 2
    else:
        intensity = np.abs(field) ** 2
    total = float(np.trapezoid(intensity, x))
    if total <= 0:
        raise RuntimeError("mode-overlap normalization failed")
    bounds = _layer_boundaries(layers)
    overlaps = []
    for i in range(len(layers)):
        if i == len(layers) - 1:
            mask = (x >= bounds[i]) & (x <= bounds[i + 1])
        else:
            mask = (x >= bounds[i]) & (x < bounds[i + 1])
        if np.count_nonzero(mask) < 2:
            overlaps.append(0.0)
        else:
            overlaps.append(float(np.trapezoid(intensity[mask], x[mask]) / total))
    overlaps = np.asarray(overlaps, dtype=float)
    s = overlaps.sum()
    if s <= 0:
        raise RuntimeError("all modal overlaps are zero")
    return overlaps / s


def _effective_alpha(
    layers: list[Layer],
    wavelengths: np.ndarray,
    overlap_factors: Sequence[float] | None = None,
    mode_result: Any | None = None,
    T: float = 25.0,
) -> np.ndarray:
    """Effective modal power loss alpha_eff = sum_i Gamma_i alpha_i."""
    if not layers:
        raise ValueError("layers cannot be empty")
    wavelengths = np.asarray(wavelengths, dtype=float)
    if np.any(wavelengths <= 0):
        raise ValueError("wavelengths must be positive")
    if overlap_factors is not None:
        overlaps = np.asarray(overlap_factors, dtype=float)
        if overlaps.shape != (len(layers),):
            raise ValueError("overlap_factors must have one value per layer")
        if np.any(overlaps < 0):
            raise ValueError("overlap_factors must be non-negative")
        if np.all(overlaps == 0):
            raise ValueError("overlap_factors cannot all be zero")
        weights = overlaps / overlaps.sum()
    elif mode_result is not None:
        weights = _modal_overlap_factors(layers, mode_result)
    else:
        raise ValueError("physics-first loss requires overlap_factors or a solved mode_result")
    weighted = np.zeros_like(wavelengths, dtype=float)
    for w, layer in zip(weights, layers):
        weighted += w * np.asarray(layer.alpha_power(wavelengths, T=T), dtype=float)
    return weighted


def _first_mode_neff(mode_result: Any) -> np.ndarray:
    neff = np.asarray(mode_result.n_eff, dtype=float)
    if neff.ndim == 1:
        return neff
    return neff[:, 0]


def _validate_coupler(t: complex, kappa: complex | None = None) -> Coupler:
    c = Coupler(t=t, kappa=kappa)
    c.validate_lossless()
    return c


def single_waveguide(
    wavelengths: Any,
    layers: list[Layer],
    T: float = 25.0,
    polarization: str = "TE",
    length: float = 0.0,
    overlap_factors: Sequence[float] | None = None,
) -> TransmissionResult:
    wl = _validate_wavelengths(wavelengths)
    if length < 0:
        raise ValueError("length must be non-negative")

    mode = solve_waveguide_modes(wl, layers, T=T, polarization=polarization, num_modes=1)
    neff = _first_mode_neff(mode)
    alpha = _effective_alpha(layers, wl, overlap_factors=overlap_factors, mode_result=mode, T=T)
    phase = propagation_phase(neff, wl, length)
    field = np.exp(-0.5 * alpha * length) * np.exp(-1j * phase)
    power = np.abs(field) ** 2

    return TransmissionResult(
        wavelength=wl,
        field=field,
        power=power,
        metadata={
            "case": "waveguide",
            "length": float(length),
            "temperature": float(T),
            "polarization": polarization,
        },
    )


def straight_waveguide(*args, **kwargs):
    """Alias for single_waveguide for a more intuitive public API."""
    return single_waveguide(*args, **kwargs)


def single_mrr_thru(
    wavelengths: Any,
    resonator: RingGeometry,
    layers: list[Layer],
    T: float = 25.0,
    polarization: str = "TE",
    t: complex = 1.0,
    kappa: complex | None = None,
    overlap_factors: Sequence[float] | None = None,
) -> TransmissionResult:
    wl = _validate_wavelengths(wavelengths)
    _validate_coupler(t, kappa)

    mode = solve_waveguide_modes(wl, layers, T=T, polarization=polarization, num_modes=1)
    neff = _first_mode_neff(mode)
    alpha = _effective_alpha(layers, wl, overlap_factors=overlap_factors, mode_result=mode, T=T)
    L = ring_circumference(resonator)
    phi = propagation_phase(neff, wl, L)
    a = np.exp(-0.5 * alpha * L)
    E_rt = a * np.exp(-1j * phi)

    field = (t - E_rt) / (1.0 - t * E_rt)
    power = np.abs(field) ** 2

    return TransmissionResult(
        wavelength=wl,
        field=field,
        power=power,
        metadata={"case": "mrr_thru", "resonator": resonator},
    )


def single_mrr_add_drop(
    wavelengths: Any,
    resonator: RingGeometry,
    layers: list[Layer],
    T: float = 25.0,
    polarization: str = "TE",
    t1: complex = 1.0,
    kappa1: complex | None = None,
    t2: complex | None = None,
    kappa2: complex | None = None,
    overlap_factors: Sequence[float] | None = None,
) -> TransmissionResult:
    wl = _validate_wavelengths(wavelengths)
    c1 = _validate_coupler(t1, kappa1)
    if t2 is None and kappa2 is None:
        t2 = t1
        kappa2 = kappa1
    c2 = _validate_coupler(t2 if t2 is not None else t1, kappa2 if kappa2 is not None else kappa1)

    mode = solve_waveguide_modes(wl, layers, T=T, polarization=polarization, num_modes=1)
    neff = _first_mode_neff(mode)
    alpha = _effective_alpha(layers, wl, overlap_factors=overlap_factors, mode_result=mode, T=T)
    L = ring_circumference(resonator)
    phi = propagation_phase(neff, wl, L)

    a = np.exp(-0.5 * alpha * L)
    exp_phi = np.exp(-1j * phi)
    den = 1.0 - a * c1.t * c2.t * exp_phi

    thru = (c1.t - a * c2.t * exp_phi) / den
    drop_amp = c1.kappa * c2.kappa
    drop = 1j * np.sqrt(a) * drop_amp * np.exp(-0.5j * phi) / den

    field = np.vstack([thru, drop])
    power = np.abs(field) ** 2
    total_power = power[0] + power[1]
    if np.any(total_power > 1.0 + 1e-8):
        raise RuntimeError("Passive add-drop ring violates energy conservation")

    return TransmissionResult(
        wavelength=wl,
        field=field,
        power=power,
        metadata={"case": "mrr_add_drop", "resonator": resonator, "ports": ["through", "drop"], "power_budget_max": float(np.max(total_power))},
    )


def _cascaded_mrrs_add_drop_legacy(
    wavelengths: Any,
    params_list: list[dict],
    bus_segments: list[dict] | None = None,
):
    wl = _validate_wavelengths(wavelengths)
    if not params_list:
        raise ValueError("params_list cannot be empty")

    bus_field = np.ones_like(wl, dtype=np.complex128)
    drop_powers: list[np.ndarray] = []
    bus_segments = list(bus_segments or [])

    for i, params in enumerate(params_list):
        args = [
            wl,
            params["resonator"],
            params["layers"],
            params["T"],
            params.get("polarization", "TE"),
            params["t1"],
            params.get("kappa1"),
            params.get("t2"),
            params.get("kappa2"),
        ]
        if params.get("overlap_factors") is not None:
            args.append(params.get("overlap_factors"))
        res = single_mrr_add_drop(*args)

        incoming_power = np.abs(bus_field) ** 2
        drop_powers.append(res.power[1] * incoming_power)

        bus_field = bus_field * res.field[0]

        if i < len(bus_segments):
            seg = bus_segments[i]
            length = float(seg.get("length", 0.0))
            if length < 0:
                raise ValueError("bus segment length must be non-negative")
            alpha = float(seg.get("alpha", 0.0))
            n_eff = float(seg.get("n_eff", 1.0))
            bus_field = bus_field * np.exp(-0.5 * alpha * length) * np.exp(
                -1j * propagation_phase(n_eff, wl, length)
            )

    return bus_field, np.abs(bus_field) ** 2, drop_powers


def cascaded_mrrs_add_drop(
    wavelengths: Any,
    params_list: list[dict] | None = None,
    bus_segments: list[dict] | None = None,
    resonators: list[RingGeometry] | None = None,
    layers: list[Layer] | None = None,
    coupling: dict | None = None,
    T: float = 25.0,
    polarization: str = "TE",
    overlap_factors: Sequence[float] | None = None,
):
    """Cascade add-drop rings.

    Backward-compatible mode: pass ``params_list`` and receive
    ``(bus_field, bus_power, drop_powers)``.

    Ergonomic mode: pass ``resonators=[...]``, shared ``layers``, and
    ``coupling={"K": ...}`` or coupler fields; receive a TransmissionResult
    whose ports are ``through`` and ``drop``.
    """
    if params_list is not None:
        return _cascaded_mrrs_add_drop_legacy(wavelengths, params_list, bus_segments=bus_segments)
    if resonators is None or layers is None:
        raise ValueError("provide either params_list or both resonators and layers")
    coupling = dict(coupling or {})
    if "K" in coupling:
        c = Coupler.from_power_coupling(coupling["K"])
        t1 = coupling.get("t1", c.t)
        kappa1 = coupling.get("kappa1", c.kappa)
        t2 = coupling.get("t2", c.t)
        kappa2 = coupling.get("kappa2", c.kappa)
    else:
        t1 = coupling.get("t1", coupling.get("t", 1.0))
        kappa1 = coupling.get("kappa1", coupling.get("kappa"))
        t2 = coupling.get("t2", t1)
        kappa2 = coupling.get("kappa2", kappa1)
    params = [
        {
            "resonator": r,
            "layers": layers,
            "T": T,
            "polarization": polarization,
            "t1": t1,
            "kappa1": kappa1,
            "t2": t2,
            "kappa2": kappa2,
            "overlap_factors": overlap_factors,
        }
        for r in resonators
    ]
    bus_field, bus_power, drop_powers = _cascaded_mrrs_add_drop_legacy(
        wavelengths, params, bus_segments=bus_segments
    )
    total_drop = np.sum(np.vstack(drop_powers), axis=0) if drop_powers else np.zeros_like(bus_power)
    field = np.vstack([bus_field, np.sqrt(np.maximum(total_drop, 0.0)).astype(np.complex128)])
    power = np.vstack([bus_power, total_drop])
    return TransmissionResult(
        wavelength=_validate_wavelengths(wavelengths),
        field=field,
        power=power,
        metadata={"case": "multi_mrr", "ports": ["through", "drop"], "num_rings": len(resonators)},
    )


def compute_transmission(
    case: str,
    wavelengths: Any,
    layers: list[Layer] | None = None,
    resonator: RingGeometry | None = None,
    T: float = 25.0,
    polarization: str = "TE",
    coupling: dict | None = None,
    params_list: list[dict] | None = None,
    **kwargs,
):
    wl = _validate_wavelengths(wavelengths)
    coupling = dict(coupling or {})
    kwargs = dict(kwargs)

    if case == "waveguide":
        if layers is None:
            raise ValueError("layers are required for waveguide transmission")
        return single_waveguide(wl, layers, T=T, polarization=polarization, **kwargs)

    if case == "mrr_thru":
        if layers is None or resonator is None:
            raise ValueError("layers and resonator are required for mrr_thru")
        return single_mrr_thru(wl, resonator, layers, T=T, polarization=polarization, **coupling, **kwargs)

    if case == "mrr_add_drop":
        if layers is None or resonator is None:
            raise ValueError("layers and resonator are required for mrr_add_drop")
        return single_mrr_add_drop(wl, resonator, layers, T=T, polarization=polarization, **coupling, **kwargs)

    if case == "multi_mrr":
        if params_list is None:
            raise ValueError("params_list is required for multi_mrr")
        return cascaded_mrrs_add_drop(wl, params_list=params_list, **kwargs)

    raise ValueError(f"Unknown transmission case: {case!r}")