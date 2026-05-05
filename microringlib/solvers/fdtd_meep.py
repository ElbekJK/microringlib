from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import require_module


@dataclass
class FDTDResult:
    wavelengths: np.ndarray
    transmission: np.ndarray
    backend: str = "meep"
    raw: object | None = None


def is_available() -> bool:
    try:
        require_module("meep", "conda install -c conda-forge pymeep")
        return True
    except ImportError:
        return False


def simulate_straight_waveguide_transmission(
    *,
    wavelength_center: float = 1.55e-6,
    wavelength_span: float = 0.04e-6,
    n_core: float = 3.48,
    n_clad: float = 1.444,
    waveguide_width: float = 0.5e-6,
    cell_length: float = 6.0e-6,
    cell_height: float = 3.0e-6,
    resolution: int = 15,
    runtime: float = 80,
    nfreq: int = 101,
) -> FDTDResult:
    """
    Small 2D MEEP straight-waveguide smoke simulation.

    Parameters are SI units. Internally, geometry is represented in microns.
    """
    require_module("meep", "conda install -c conda-forge pymeep")

    import meep as mp

    um = 1e-6

    wl0 = float(wavelength_center) / um
    span = float(wavelength_span) / um

    fcen = 1.0 / wl0
    df = abs(1.0 / (wl0 - span / 2.0) - 1.0 / (wl0 + span / 2.0))

    sx = float(cell_length) / um
    sy = float(cell_height) / um
    wg_w = float(waveguide_width) / um

    cell = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(1.0)]

    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, wg_w, mp.inf),
            center=mp.Vector3(),
            material=mp.Medium(index=float(n_core)),
        )
    ]

    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(-0.4 * sx, 0),
            size=mp.Vector3(0, wg_w),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        default_material=mp.Medium(index=float(n_clad)),
        resolution=int(resolution),
    )

    flux_region = mp.FluxRegion(
        center=mp.Vector3(0.4 * sx, 0),
        size=mp.Vector3(0, 2.0 * wg_w),
    )

    trans = sim.add_flux(fcen, df, int(nfreq), flux_region)

    sim.run(until_after_sources=float(runtime))

    fluxes = np.asarray(mp.get_fluxes(trans), dtype=float)
    freqs = np.asarray(mp.get_flux_freqs(trans), dtype=float)

    wavelengths = (1.0 / freqs) * um

    order = np.argsort(wavelengths)

    return FDTDResult(
        wavelengths=wavelengths[order],
        transmission=fluxes[order],
        backend="meep",
        raw=sim,
    )


def simulate_ring_resonator_2d(
    *,
    wavelength_center: float = 1.55e-6,
    wavelength_span: float = 0.08e-6,
    n_core: float = 3.48,
    n_clad: float = 1.444,
    ring_radius: float = 10e-6,
    waveguide_width: float = 0.5e-6,
    gap: float = 0.2e-6,
    resolution: int = 20,
    runtime: float = 300,
    nfreq: int = 201,
) -> FDTDResult:
    """
    Simple 2D MEEP ring-resonator validation model.

    This is intentionally a smoke/validation backend, not the fast default model.
    """
    require_module("meep", "conda install -c conda-forge pymeep")

    import meep as mp

    um = 1e-6

    wl0 = float(wavelength_center) / um
    span = float(wavelength_span) / um

    fcen = 1.0 / wl0
    df = abs(1.0 / (wl0 - span / 2.0) - 1.0 / (wl0 + span / 2.0))

    r = float(ring_radius) / um
    wg_w = float(waveguide_width) / um
    g = float(gap) / um

    pad = 3.0
    sx = 2.0 * r + 2.0 * pad
    sy = 2.0 * r + 2.0 * pad

    bus_y = -(r + g + wg_w)

    cell = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(1.0)]

    core = mp.Medium(index=float(n_core))
    clad = mp.Medium(index=float(n_clad))

    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, wg_w, mp.inf),
            center=mp.Vector3(0, bus_y),
            material=core,
        ),
        mp.Cylinder(
            radius=r + wg_w / 2.0,
            height=mp.inf,
            center=mp.Vector3(),
            material=core,
        ),
        mp.Cylinder(
            radius=r - wg_w / 2.0,
            height=mp.inf,
            center=mp.Vector3(),
            material=clad,
        ),
    ]

    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(-0.45 * sx, bus_y),
            size=mp.Vector3(0, wg_w),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        default_material=clad,
        resolution=int(resolution),
    )

    flux_region = mp.FluxRegion(
        center=mp.Vector3(0.45 * sx, bus_y),
        size=mp.Vector3(0, 2.0 * wg_w),
    )

    trans = sim.add_flux(fcen, df, int(nfreq), flux_region)

    sim.run(until_after_sources=float(runtime))

    fluxes = np.asarray(mp.get_fluxes(trans), dtype=float)
    freqs = np.asarray(mp.get_flux_freqs(trans), dtype=float)

    wavelengths = (1.0 / freqs) * um
    order = np.argsort(wavelengths)

    return FDTDResult(
        wavelengths=wavelengths[order],
        transmission=fluxes[order],
        backend="meep",
        raw=sim,
    )


def simulate_straight_waveguide_3d(
    *,
    wavelength_center: float = 1.55e-6,
    wavelength_span: float = 0.04e-6,
    n_core: float = 2.60,
    n_clad: float = 1.444,
    waveguide_width: float = 0.85e-6,
    waveguide_thickness: float = 0.50e-6,
    cell_length: float = 5.0e-6,
    cell_height: float = 3.0e-6,
    cell_z: float = 2.4e-6,
    resolution: int = 10,
    runtime: float = 80,
    nfreq: int = 61,
) -> FDTDResult:
    """Small true-3D MEEP straight-waveguide calibration cell.

    This is intended for optional publication signoff / field-handoff tests,
    not large sweeps.  Set MICRORINGLIB_RUN_MEEP_3D=1 in the demos to use it.
    Parameters are SI; MEEP geometry is in microns.
    """
    require_module("meep", "conda install -c conda-forge pymeep")
    import meep as mp

    um = 1e-6
    wl0 = float(wavelength_center) / um
    span = float(wavelength_span) / um
    fcen = 1.0 / wl0
    df = abs(1.0 / (wl0 - span / 2.0) - 1.0 / (wl0 + span / 2.0))

    sx = float(cell_length) / um
    sy = float(cell_height) / um
    sz = float(cell_z) / um
    wg_w = float(waveguide_width) / um
    wg_h = float(waveguide_thickness) / um

    cell = mp.Vector3(sx, sy, sz)
    pml_layers = [mp.PML(0.8)]
    core = mp.Medium(index=float(n_core))
    clad = mp.Medium(index=float(n_clad))

    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, wg_w, wg_h), center=mp.Vector3(), material=core)
    ]
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            component=mp.Ey,
            center=mp.Vector3(-0.42 * sx, 0, 0),
            size=mp.Vector3(0, wg_w, wg_h),
        )
    ]
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        default_material=clad,
        resolution=int(resolution),
    )
    flux_region = mp.FluxRegion(center=mp.Vector3(0.42 * sx, 0, 0), size=mp.Vector3(0, 1.8 * wg_w, 1.8 * wg_h))
    trans = sim.add_flux(fcen, df, int(nfreq), flux_region)
    sim.run(until_after_sources=float(runtime))
    fluxes = np.asarray(mp.get_fluxes(trans), dtype=float)
    freqs = np.asarray(mp.get_flux_freqs(trans), dtype=float)
    wavelengths = (1.0 / freqs) * um
    order = np.argsort(wavelengths)
    return FDTDResult(wavelengths=wavelengths[order], transmission=fluxes[order], backend="meep-3D-straight-waveguide", raw=sim)


def simulate_ring_resonator_3d(
    *,
    wavelength_center: float = 1.55e-6,
    wavelength_span: float = 0.04e-6,
    n_core: float = 2.60,
    n_clad: float = 1.444,
    ring_radius: float = 6e-6,
    waveguide_width: float = 0.85e-6,
    waveguide_thickness: float = 0.50e-6,
    gap: float = 0.25e-6,
    resolution: int = 6,
    runtime: float = 80,
    nfreq: int = 61,
) -> FDTDResult:
    """True-3D MEEP microring simulation cell.

    v28 uses this as a full-radius 3D microring simulation when demos pass the
    design radius. Large radii are computationally expensive, so resolution and
    runtime should be chosen carefully. A separate local-hook mode can still be
    requested by higher-level demos when explicitly enabled.
    """
    require_module("meep", "conda install -c conda-forge pymeep")
    import meep as mp

    um = 1e-6
    wl0 = float(wavelength_center) / um
    span = float(wavelength_span) / um
    fcen = 1.0 / wl0
    df = abs(1.0 / (wl0 - span / 2.0) - 1.0 / (wl0 + span / 2.0))

    r = float(ring_radius) / um
    wg_w = float(waveguide_width) / um
    wg_h = float(waveguide_thickness) / um
    g = float(gap) / um
    pad = 2.2
    sx = 2.0 * r + 2.0 * pad
    sy = 2.0 * r + 2.0 * pad
    sz = max(2.2, 4.0 * wg_h)
    bus_y = -(r + g + wg_w)

    cell = mp.Vector3(sx, sy, sz)
    pml_layers = [mp.PML(0.8)]
    core = mp.Medium(index=float(n_core))
    clad = mp.Medium(index=float(n_clad))
    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, wg_w, wg_h), center=mp.Vector3(0, bus_y, 0), material=core),
        mp.Cylinder(radius=r + wg_w / 2.0, height=wg_h, center=mp.Vector3(0, 0, 0), axis=mp.Vector3(0, 0, 1), material=core),
        mp.Cylinder(radius=max(1e-6 / um, r - wg_w / 2.0), height=wg_h * 1.05, center=mp.Vector3(0, 0, 0), axis=mp.Vector3(0, 0, 1), material=clad),
    ]
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            component=mp.Ey,
            center=mp.Vector3(-0.42 * sx, bus_y, 0),
            size=mp.Vector3(0, wg_w, wg_h),
        )
    ]
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        default_material=clad,
        resolution=int(resolution),
    )
    flux_region = mp.FluxRegion(center=mp.Vector3(0.42 * sx, bus_y, 0), size=mp.Vector3(0, 1.8 * wg_w, 1.8 * wg_h))
    trans = sim.add_flux(fcen, df, int(nfreq), flux_region)
    sim.run(until_after_sources=float(runtime))
    fluxes = np.asarray(mp.get_fluxes(trans), dtype=float)
    freqs = np.asarray(mp.get_flux_freqs(trans), dtype=float)
    wavelengths = (1.0 / freqs) * um
    order = np.argsort(wavelengths)
    return FDTDResult(wavelengths=wavelengths[order], transmission=fluxes[order], backend="meep-3D-microring-full-radius", raw=sim)
