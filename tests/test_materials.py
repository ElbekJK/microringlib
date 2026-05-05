import numpy as np
import pytest
import microringlib as mrl


def test_constant_material_complex_index_and_alpha():
    mat = mrl.ConstantMaterial("test", n=2.0, k=1e-4, dn_dT=1e-5)
    wl = np.array([1.5e-6, 1.6e-6])
    nc = mat.n_complex(wl, T=35.0)
    assert np.allclose(np.real(nc), 2.0 + 1e-4)
    assert np.allclose(np.imag(nc), 1e-4)
    assert np.all(mat.alpha_power(wl) > 0)


def test_tabulated_material_interpolates():
    mat = mrl.TabulatedMaterial("tab", [1.5e-6, 1.6e-6], [2.0, 2.2], [0.0, 1e-4])
    nc = mat.n_complex(1.55e-6)
    assert np.real(nc) == pytest.approx(2.1)
    assert np.imag(nc) == pytest.approx(5e-5)


def test_layer_material_model_backward_compatible():
    mat = mrl.ConstantMaterial("Si", n=3.47, dn_dT=1e-4)
    layer = mrl.Layer("core", 220e-9, material_model=mat, alpha=10.0)
    assert layer.n_at(35.0) == pytest.approx(3.471)
    assert layer.alpha_power(1.55e-6) == pytest.approx(10.0)


def test_waveguide_accepts_material_models():
    wl = np.linspace(1540e-9, 1560e-9, 11)
    sio2 = mrl.ConstantMaterial("SiO2", 1.444)
    si = mrl.ConstantMaterial("Si", 3.476)
    layers = [
        mrl.Layer("lower", 2e-6, material_model=sio2),
        mrl.Layer("core", 220e-9, material_model=si, alpha=mrl.Layer.dbcm_to_npm(1.0)),
        mrl.Layer("upper", 2e-6, material_model=sio2),
    ]
    res = mrl.single_waveguide(wl, layers, length=10e-6)
    assert res.power.shape == wl.shape
    assert np.all(res.power <= 1.0 + 1e-8)
