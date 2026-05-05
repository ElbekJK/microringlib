import numpy as np
import microringlib as mrl


def test_fast_batch_shapes_and_passivity():
    wl = np.linspace(1540e-9, 1560e-9, 101)
    fields, powers, t, k = mrl.single_mrr_thru_fast_batch(wl, 10e-6, 3.476, 2.0, [0.01, 0.02])
    assert fields.shape == (2, wl.size)
    assert powers.shape == (2, wl.size)
    assert np.all(powers <= 1.0 + 1e-8)
    assert np.allclose(np.abs(t)**2 + np.abs(k)**2, 1.0)


def test_fast_sfwm_normalizes():
    P = np.linspace(0, 1e-3, 10)
    r = mrl.sfwm_pair_rate_relative_fast(P, gamma=2.0, loaded_Q=10000, ring_radius=10e-6)
    assert np.isclose(r[-1], 1.0)
    assert np.all(r >= 0)
