import numpy as np
import pytest

from microringlib.utils import evaluate_alpha, layers_signature


def test_evaluate_alpha_scalar():
    wl = np.linspace(1540e-9, 1560e-9, 11)
    a = evaluate_alpha(10.0, wl)
    assert np.allclose(a, np.full_like(wl, 10.0))


def test_evaluate_alpha_array():
    wl = np.linspace(1540e-9, 1560e-9, 11)
    arr = np.linspace(1.0, 2.0, wl.size)
    a = evaluate_alpha(arr, wl)
    assert np.allclose(a, arr)


def test_evaluate_alpha_callable():
    wl = np.linspace(1540e-9, 1560e-9, 11)

    def alpha_fn(x):
        return 5.0 + 2.0 * (x - x.min()) / (x.max() - x.min())

    a = evaluate_alpha(alpha_fn, wl)
    assert a[0] == pytest.approx(5.0)
    assert a[-1] == pytest.approx(7.0)


def test_evaluate_alpha_rejects_negative():
    wl = np.linspace(1540e-9, 1560e-9, 11)
    with pytest.raises(ValueError):
        evaluate_alpha(-1.0, wl)


def test_layers_signature_stable(layers):
    sig1 = layers_signature(layers)
    sig2 = layers_signature(layers)
    assert sig1 == sig2