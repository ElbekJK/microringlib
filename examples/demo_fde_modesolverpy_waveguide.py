#!/usr/bin/env python3
"""
FDE demo using modesolverpy.

This demo is optional. It requires:
    pip install modesolverpy

It is meant to validate the optional microringlib FDE backend.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np


def main() -> int:
    if importlib.util.find_spec("modesolverpy") is None:
        print("SKIP: modesolverpy is not installed.")
        return 0

    from microringlib.solvers.fde_modesolverpy import solve_dispersion

    wavelengths = np.linspace(1.52e-6, 1.58e-6, 3)

    results = solve_dispersion(
        wavelengths=wavelengths,
        core_width=500e-9,
        core_thickness=220e-9,
        n_core=3.48,
        n_clad=1.444,
        dx=40e-9,
        dy=40e-9,
        x_span=3.0e-6,
        y_span=3.0e-6,
    )

    neff = np.array([r.neff.real for r in results])

    print("FDE modesolverpy demo")
    print("wavelength_nm, neff_real")
    for wl, n in zip(wavelengths, neff):
        print(f"{wl * 1e9:.2f}, {n:.6f}")

    if not np.all(np.isfinite(neff)):
        raise RuntimeError("FDE returned non-finite neff values.")

    if not np.all((neff > 1.0) & (neff < 4.5)):
        raise RuntimeError(f"FDE neff values look unphysical: {neff}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
