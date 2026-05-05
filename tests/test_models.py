import numpy as np
import pytest

from microringlib import Coupler, RingGeometry, Layer, ring_circumference


def test_layer_temperature_shift():
    layer = Layer("silica", 2e-6, 1.45, dn_dT=1e-5)
    assert layer.n_at(25.0) == pytest.approx(1.45)
    assert layer.n_at(35.0) == pytest.approx(1.45 + 1e-5 * 10.0)


def test_coupler_validation_accepts_lossless():
    c = Coupler(t=0.9, kappa=np.sqrt(1 - 0.9**2))
    c.validate_lossless()


def test_coupler_validation_rejects_invalid():
    c = Coupler(t=0.9, kappa=0.9)
    with pytest.raises(ValueError):
        c.validate_lossless()


def test_ring_circumference_circular():
    ring = RingGeometry(kind="circular", radius=10e-6)
    assert ring_circumference(ring) == pytest.approx(2 * np.pi * 10e-6)


def test_ring_circumference_racetrack():
    ring = RingGeometry(kind="racetrack", radius=10e-6, straight_length=5e-6)
    assert ring_circumference(ring) == pytest.approx(2 * np.pi * 10e-6 + 2 * 5e-6)


def test_ring_circumference_elliptical_positive():
    ring = RingGeometry(kind="elliptical", a=12e-6, b=8e-6)
    assert ring_circumference(ring) > 0