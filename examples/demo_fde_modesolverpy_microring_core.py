#!/usr/bin/env python3
"""
FDE + microringlib core demo.

Workflow:
    modesolverpy/FDE backend -> effective index -> microringlib ring model

This demo proves that the optional FDE backend can feed the core analytical
microring model.
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
from microringlib.solvers.fde_modesolverpy import solve_waveguide_mode


def main() -> int:
    if importlib.util.find_spec("modesolverpy") is None:
        print("SKIP: modesolverpy is not installed.")
        print("Install with: pip install modesolverpy")
        return 0

    outdir = pathlib.Path("figures")
    outdir.mkdir(exist_ok=True)

    wavelength0 = 1550e-9

    fde = solve_waveguide_mode(
        wavelength=wavelength0,
        core_width=500e-9,
        core_thickness=220e-9,
        n_core=3.48,
        n_clad=1.444,
        dx=40e-9,
        dy=40e-9,
        x_span=3.0e-6,
        y_span=3.0e-6,
        mode_index=0,
        allow_fallback=True,
    )

    n_eff = float(np.real(fde.neff))

    print("\n=== FDE effective-index extraction ===")
    print(f"backend: {fde.backend}")
    print(f"wavelength: {wavelength0 * 1e9:.2f} nm")
    print(f"n_eff: {n_eff:.6f}")

    if not np.isfinite(n_eff):
        raise RuntimeError("FDE returned non-finite n_eff.")

    if not (1.0 < n_eff < 4.5):
        raise RuntimeError(f"FDE n_eff looks unphysical: {n_eff}")

    wavelengths = np.linspace(1520e-9, 1580e-9, 20001)

    radius = 10e-6
    alpha_dbcm = 2.0
    coupling_power = 0.08

    field, power, t, kappa = mrl.single_mrr_thru_fast(
        wavelengths,
        radius=radius,
        n_eff=n_eff,
        alpha_dbcm=alpha_dbcm,
        K=coupling_power,
    )

    metrics = mrl.resonance_metrics_fast(
        wavelengths,
        power,
        target_wavelength=wavelength0,
        kind="dips",
    )

    print("\n=== microringlib core ring response using FDE n_eff ===")
    print(f"radius: {radius * 1e6:.3f} um")
    print(f"|t|^2: {abs(t) ** 2:.6f}")
    print(f"|kappa|^2: {abs(kappa) ** 2:.6f}")
    print(f"passive check max(P_thru): {np.max(power):.6f}")

    print("\n=== Extracted resonance metrics ===")
    print(f"resonance wavelength: {metrics['resonance_wavelength'] * 1e9:.4f} nm")
    print(f"FWHM: {metrics['fwhm'] * 1e12:.4f} pm")
    print(f"FSR: {metrics['fsr'] * 1e9:.4f} nm")
    print(f"loaded Q: {metrics['loaded_Q']:.3g}")
    print(f"extinction ratio: {metrics['extinction_ratio_db']:.3f} dB")

    if not np.all(np.isfinite(power)):
        raise RuntimeError("microringlib returned non-finite power.")

    if np.max(power) > 1.0 + 1e-8:
        raise RuntimeError("Passive through power exceeded 1.")

    plt.figure(figsize=(7, 4))
    plt.plot(wavelengths * 1e9, power)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Through power")
    plt.title(f"FDE-calibrated microring response, n_eff={n_eff:.4f}")
    plt.tight_layout()

    fig_path = outdir / "fde_modesolverpy_microring_core.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"\nSaved: {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
