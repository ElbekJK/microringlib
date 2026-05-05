import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)

from microringlib import plot_transmission, plot_mode_profile, solve_waveguide_modes


def test_plotting_smoke(layers):
    wl = np.linspace(1540e-9, 1560e-9, 51)
    T = 1.0 - 0.5 * np.exp(-((wl - wl.mean()) / 2e-9) ** 2)

    plot_transmission(wl, T, labels=["Through"])
    mode = solve_waveguide_modes(1550e-9, layers, T=30.0, polarization="TE", num_modes=1)
    plot_mode_profile(mode.field, layers, T=30.0, labels=["Mode 0"])