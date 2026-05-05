import numpy as np
import microringlib as mrl
from microringlib import tmm


def test_propagation_transfer_matches_waveguide_smatrix():
    wl = np.linspace(1.54e-6, 1.56e-6, 101)
    neff = 2.4
    alpha = tmm.dbcm_to_npm(2.0)
    L = 250e-6
    M = tmm.propagation_transfer(wl, neff, L, alpha)
    S = tmm.waveguide_smatrix(wl, neff, L, alpha)
    expected = tmm.propagation_amplitude(wl, neff, L, alpha)
    assert np.allclose(M.through, expected)
    assert np.allclose(S.response("in", "out"), expected)
    assert S.is_passive()


def test_lossless_directional_coupler_is_unitary_reciprocal():
    wl = np.linspace(1.5e-6, 1.6e-6, 11)
    c = tmm.directional_coupler_smatrix(wl, 0.25)
    assert c.is_unitary(atol=1e-12)
    assert c.is_reciprocal(atol=1e-12)
    assert c.is_passive(atol=1e-12)


def test_cascade_two_waveguides_equals_length_sum():
    wl = np.linspace(1.54e-6, 1.56e-6, 101)
    neff = 2.2
    a = tmm.waveguide_smatrix(wl, neff, 100e-6)
    b = tmm.waveguide_smatrix(wl, neff, 150e-6)
    casc = tmm.cascade_two_ports([a, b])
    ref = tmm.waveguide_smatrix(wl, neff, 250e-6)
    assert np.allclose(casc.response(0, 1), ref.response(0, 1))
    assert casc.is_unitary(atol=1e-12)


def test_mzi_has_expected_notches_and_passivity():
    wl = np.linspace(1.50e-6, 1.60e-6, 1001)
    mzi = tmm.mzi_smatrix(wl, 2.4, 1.0e-3, 1.03e-3, K1=0.5, K2=0.5)
    p = mzi.power(0, 0)
    assert np.nanmin(p) < 1e-3
    assert np.nanmax(p) <= 1.0 + 1e-10
    assert mzi.is_unitary(atol=1e-10)


def test_ring_allpass_is_passive_and_matches_public_tmm_export():
    wl = np.linspace(1545e-9, 1555e-9, 1001)
    h = mrl.ring_allpass_field(wl, 2.5, 10e-6, 0.12, alpha_power=tmm.dbcm_to_npm(3.0))
    S = mrl.ring_allpass_smatrix(wl, 2.5, 10e-6, 0.12, alpha_power=tmm.dbcm_to_npm(3.0))
    assert np.allclose(S.response(0, 1), h)
    assert S.is_passive(atol=1e-8)
    assert np.all(np.abs(h) <= 1.0 + 1e-8)


def test_ring_add_drop_energy_budget():
    wl = np.linspace(1545e-9, 1555e-9, 1001)
    res = tmm.ring_add_drop_fields(wl, 2.45, 10e-6, 0.08, 0.08, alpha_power=tmm.dbcm_to_npm(2.0))
    total = res["through_power"] + res["drop_power"]
    assert np.all(total <= 1.0 + 1e-8)


def test_fabry_perot_has_resonant_transmission():
    wl = np.linspace(1540e-9, 1560e-9, 2001)
    fp = tmm.fabry_perot_field(wl, 2.0, 20e-6, r1=0.6)
    assert np.max(np.abs(fp) ** 2) > 0.5
    assert np.all(np.isfinite(fp))
