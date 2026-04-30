import numpy as np
import pytest

from microringlib import compute_group_index, compute_transmission, single_mrr_thru, RingGeometry


def test_group_index_rejects_too_few_samples():
    with pytest.raises(ValueError):
        compute_group_index(np.array([1.0]), np.array([1.0]))


def test_compute_transmission_rejects_negative_wavelengths(layers):
    wl = np.linspace(1540e-9, 1560e-9, 51)
    with pytest.raises(ValueError):
        compute_transmission(
            case="waveguide",
            wavelengths=-wl,
            layers=layers,
            T=30.0,
            polarization="TE",
            length=10e-6,
        )


def test_single_mrr_thru_rejects_invalid_t(layers):
    wl = np.linspace(1540e-9, 1560e-9, 51)
    ring = RingGeometry(kind="circular", radius=10e-6)
    with pytest.raises(Exception):
        single_mrr_thru(wl, ring, layers, T=30.0, polarization="TE", t=1.2)