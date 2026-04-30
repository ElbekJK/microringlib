import numpy as np
import pytest

from microringlib import solve_waveguide_modes, compute_group_index


def test_mode_solver_scalar_wavelength(layers):
    mode = solve_waveguide_modes(1550e-9, layers, T=30.0, polarization="TE", num_modes=1)
    assert mode.wavelength.shape == (1,)
    assert mode.field.ndim == 2
    assert mode.n_eff.shape == (1,)


def test_mode_solver_array_wavelengths(layers):
    wl = np.linspace(1548e-9, 1552e-9, 7)
    mode = solve_waveguide_modes(wl, layers, T=30.0, polarization="TM", num_modes=2)
    assert mode.field.shape[0] == wl.size
    assert mode.field.shape[1] == 2
    assert mode.n_eff.shape == (wl.size, 2)


def test_mode_solver_deterministic(layers):
    wl = np.linspace(1548e-9, 1552e-9, 7)
    m1 = solve_waveguide_modes(wl, layers, T=30.0, polarization="TM", num_modes=2)
    m2 = solve_waveguide_modes(wl, layers, T=30.0, polarization="TM", num_modes=2)
    assert np.allclose(m1.n_eff, m2.n_eff)
    assert np.allclose(m1.field, m2.field)


def test_mode_solver_rejects_empty_layers():
    wl = np.linspace(1548e-9, 1552e-9, 7)
    with pytest.raises(ValueError):
        solve_waveguide_modes(wl, [], T=30.0)


def test_group_index_for_linear_dispersion():
    wl = np.linspace(1.5e-6, 1.6e-6, 51)
    n_eff = 2.0 - 0.1 * wl
    ng = compute_group_index(wl, n_eff)
    assert np.allclose(ng, np.full_like(wl, 2.0), atol=1e-8)