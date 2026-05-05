#!/usr/bin/env python3
"""
FDTD + microringlib core comparison demo.

Purpose
-------
This demo shows the intended validation workflow:

    MEEP/FDTD ring simulation -> raw full-wave transmission
    microringlib core model   -> fast analytical ring transmission

The two are not expected to match perfectly because:
- the MEEP example is a compact 2D smoke/validation model,
- the analytical model uses a reduced all-pass ring formula,
- coupling/loss are approximate.

The goal is to demonstrate that FDTD can be used as an optional validation
backend while microringlib remains the fast analytical engine.

Run:
    python examples/demo_fdtd_meep_vs_microring_core.py

Optional dependency:
    conda install -c conda-forge pymeep
"""

from __future__ import annotations

import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import microringlib as mrl
from microringlib.solvers.fdtd_meep import simulate_ring_resonator_2d


def normalize_unit_interval(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)

    ymin = np.nanmin(y)
    ymax = np.nanmax(y)

    if not np.isfinite(ymin) or not np.isfinite(ymax):
        raise RuntimeError("Cannot normalize non-finite array.")

    if ymax <= ymin:
        return np.zeros_like(y)

    return (y - ymin) / (ymax - ymin)


def main() -> int:
    if importlib.util.find_spec("meep") is None:
        print("SKIP: meep is not installed.")
        print("Install with: conda install -c conda-forge pymeep")
        return 0

    outdir = pathlib.Path("figures")
    outdir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. FDTD ring simulation
    # -------------------------------------------------------------------------
    wavelength_center = 1550e-9
    wavelength_span = 80e-9

    radius = 10e-6
    waveguide_width = 500e-9
    gap = 200e-9

    n_core = 3.48
    n_clad = 1.444

    print("\n=== MEEP FDTD ring simulation ===")
    print("This may take a while depending on your MEEP installation and CPU.")

    fdtd = simulate_ring_resonator_2d(
        wavelength_center=wavelength_center,
        wavelength_span=wavelength_span,
        n_core=n_core,
        n_clad=n_clad,
        ring_radius=radius,
        waveguide_width=waveguide_width,
        gap=gap,
        resolution=20,
        runtime=300,
        nfreq=201,
    )

    fdtd_wl = np.asarray(fdtd.wavelengths, dtype=float)
    fdtd_flux = np.asarray(fdtd.transmission, dtype=float)

    if fdtd_wl.size == 0:
        raise RuntimeError("MEEP returned no wavelength samples.")

    if not np.all(np.isfinite(fdtd_flux)):
        raise RuntimeError("MEEP returned non-finite flux values.")

    fdtd_norm = normalize_unit_interval(fdtd_flux)

    print(f"FDTD backend: {fdtd.backend}")
    print(f"FDTD wavelength points: {fdtd_wl.size}")
    print(f"FDTD wavelength range: {fdtd_wl.min() * 1e9:.2f} - {fdtd_wl.max() * 1e9:.2f} nm")
    print(f"FDTD raw flux min/max: {fdtd_flux.min():.6e} / {fdtd_flux.max():.6e}")

    # -------------------------------------------------------------------------
    # 2. microringlib analytical response on the same wavelength grid
    # -------------------------------------------------------------------------
    # Approximate effective index for the same 2D-like high-index waveguide.
    # For a more rigorous result, use the FDE demo to extract this value.
    n_eff = 2.6

    alpha_dbcm = 2.0
    coupling_power = 0.08

    _, mrl_power, t, kappa = mrl.single_mrr_thru_fast(
        fdtd_wl,
        radius=radius,
        n_eff=n_eff,
        alpha_dbcm=alpha_dbcm,
        K=coupling_power,
    )

    mrl_norm = normalize_unit_interval(1.0 - mrl_power)

    metrics = mrl.resonance_metrics_fast(
        fdtd_wl,
        mrl_power,
        target_wavelength=wavelength_center,
        kind="dips",
    )

    print("\n=== microringlib core analytical comparison ===")
    print(f"n_eff used: {n_eff:.6f}")
    print(f"|t|^2: {abs(t) ** 2:.6f}")
    print(f"|kappa|^2: {abs(kappa) ** 2:.6f}")
    print(f"max analytical through power: {np.max(mrl_power):.6f}")

    print("\n=== Analytical resonance metrics ===")
    print(f"resonance wavelength: {metrics['resonance_wavelength'] * 1e9:.4f} nm")
    print(f"FWHM: {metrics['fwhm'] * 1e12:.4f} pm")
    print(f"FSR: {metrics['fsr'] * 1e9:.4f} nm")
    print(f"loaded Q: {metrics['loaded_Q']:.3g}")

    if np.max(mrl_power) > 1.0 + 1e-8:
        raise RuntimeError("microringlib passive through power exceeded 1.")

    # -------------------------------------------------------------------------
    # 3. Save comparison figure
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(fdtd_wl * 1e9, fdtd_norm, label="MEEP FDTD raw flux, normalized")
    plt.plot(fdtd_wl * 1e9, mrl_norm, "--", label="microringlib analytical notch, normalized")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized response")
    plt.title("FDTD validation vs microringlib analytical ring model")
    plt.legend()
    plt.tight_layout()

    fig_path = outdir / "fdtd_meep_vs_microring_core.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"\nSaved: {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
