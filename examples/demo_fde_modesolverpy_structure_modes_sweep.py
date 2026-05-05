#!/usr/bin/env python3
"""
Publication FDE demo using modesolverpy + microringlib.

This demo shows:
1. waveguide cross-section structure,
2. modesolverpy effective-index extraction,
3. wavelength sweep of n_eff,
4. FDE-calibrated microring response using microringlib.

Generated figures:
    fde_waveguide_structure.png
    fde_neff_wavelength_sweep.png
    fde_calibrated_ring_response.png

Notes:
    The actual mode-field extraction depends on the installed modesolverpy API.
    This demo focuses on robust structure + effective-index + ring-response output.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

import microringlib as mrl
from microringlib.solvers.fde_modesolverpy import solve_waveguide_mode, solve_dispersion


def plot_waveguide_structure(
    *,
    core_width: float,
    core_thickness: float,
    n_core: float,
    n_clad: float,
    filename: str,
) -> None:
    x_um = np.linspace(-2.0, 2.0, 600)
    y_um = np.linspace(-1.5, 1.5, 450)

    X, Y = np.meshgrid(x_um, y_um)
    n = np.full_like(X, n_clad, dtype=float)

    core_mask = (
        (np.abs(X) <= core_width * 1e6 / 2.0)
        & (np.abs(Y) <= core_thickness * 1e6 / 2.0)
    )
    n[core_mask] = n_core

    plt.figure(figsize=(6.8, 4.8))
    im = plt.imshow(
        n,
        extent=[x_um.min(), x_um.max(), y_um.min(), y_um.max()],
        origin="lower",
        aspect="auto",
    )
    plt.colorbar(im, label="Refractive index")
    plt.xlabel("x (um)")
    plt.ylabel("y (um)")
    plt.title("FDE Waveguide Cross-Section")
    plt.tight_layout()
    plt.savefig(filename, dpi=250)
    plt.close()


def main() -> int:
    if importlib.util.find_spec("modesolverpy") is None:
        print("SKIP: modesolverpy is not installed.")
        print("Install with: pip install modesolverpy")
        return 0

    core_width = 500e-9
    core_thickness = 220e-9
    n_core = 3.48
    n_clad = 1.444

    wavelength0 = 1550e-9
    wavelengths = np.linspace(1520e-9, 1580e-9, 9)

    print("\n=== FDE modesolverpy structure + wavelength sweep ===")

    plot_waveguide_structure(
        core_width=core_width,
        core_thickness=core_thickness,
        n_core=n_core,
        n_clad=n_clad,
        filename="fde_waveguide_structure.png",
    )

    mode0 = solve_waveguide_mode(
        wavelength=wavelength0,
        core_width=core_width,
        core_thickness=core_thickness,
        n_core=n_core,
        n_clad=n_clad,
        dx=40e-9,
        dy=40e-9,
        x_span=3.0e-6,
        y_span=3.0e-6,
        mode_index=0,
        allow_fallback=True,
    )

    print(f"Single-wavelength backend: {mode0.backend}")
    print(f"n_eff at {wavelength0 * 1e9:.2f} nm: {mode0.neff.real:.6f}")
    print(f"imag(n_eff): {mode0.neff.imag:.3e}")

    sweep = solve_dispersion(
        wavelengths=wavelengths,
        core_width=core_width,
        core_thickness=core_thickness,
        n_core=n_core,
        n_clad=n_clad,
        dx=50e-9,
        dy=50e-9,
        x_span=3.0e-6,
        y_span=3.0e-6,
        mode_index=0,
        allow_fallback=True,
    )

    neff = np.array([r.neff.real for r in sweep])
    neff_imag = np.array([r.neff.imag for r in sweep])

    print("\nWavelength sweep:")
    print("lambda_nm, neff_real, neff_imag, backend")
    for wl, r in zip(wavelengths, sweep):
        print(f"{wl * 1e9:.2f}, {r.neff.real:.6f}, {r.neff.imag:.3e}, {r.backend}")

    plt.figure(figsize=(6.8, 4.5))
    plt.plot(wavelengths * 1e9, neff, "o-")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Effective index")
    plt.title("FDE Effective Index Wavelength Sweep")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fde_neff_wavelength_sweep.png", dpi=250)
    plt.close()

    # Feed FDE n_eff into microringlib core.
    wl_ring = np.linspace(1520e-9, 1580e-9, 20001)
    radius = 10e-6
    alpha_dbcm = 2.0
    K = 0.08

    _, power, t, kappa = mrl.single_mrr_thru_fast(
        wl_ring,
        radius=radius,
        n_eff=float(mode0.neff.real),
        alpha_dbcm=alpha_dbcm,
        K=K,
    )

    metrics = mrl.resonance_metrics_fast(
        wl_ring,
        power,
        target_wavelength=wavelength0,
        kind="dips",
    )

    print("\n=== FDE-calibrated microringlib ring ===")
    print(f"radius: {radius * 1e6:.3f} um")
    print(f"|t|^2: {abs(t) ** 2:.6f}")
    print(f"|kappa|^2: {abs(kappa) ** 2:.6f}")
    print(f"resonance: {metrics['resonance_wavelength'] * 1e9:.4f} nm")
    print(f"FWHM: {metrics['fwhm'] * 1e12:.4f} pm")
    print(f"loaded Q: {metrics['loaded_Q']:.3g}")
    print(f"passive: {np.all(power <= 1.0 + 1e-8)}")

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(wl_ring * 1e9, power)
    plt.axvline(wavelength0 * 1e9, linestyle="--", label="target")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Through power")
    plt.title(f"FDE-Calibrated Ring Response, n_eff={mode0.neff.real:.4f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fde_calibrated_ring_response.png", dpi=250)
    plt.close()

    print("\nSaved:")
    print("  fde_waveguide_structure.png")
    print("  fde_neff_wavelength_sweep.png")
    print("  fde_calibrated_ring_response.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())