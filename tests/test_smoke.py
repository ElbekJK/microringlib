import numpy as np
from microringlib import Layer, RingGeometry, compute_transmission, compute_resonance_metrics

def test_smoke():
    wl = np.linspace(1540e-9, 1560e-9, 101)
    layers = [
        Layer("silica", 2e-6, 1.45, dn_dT=1e-5, alpha=10.0),
        Layer("silicon", 220e-9, 3.48, dn_dT=1.8e-4, alpha=100.0),
        Layer("silica", 2e-6, 1.45, dn_dT=1e-5, alpha=10.0),
    ]
    ring = RingGeometry(kind="circular", radius=10e-6)
    out = compute_transmission(
        case="mrr_thru",
        wavelengths=wl,
        resonator=ring,
        layers=layers,
        T=30.0,
        polarization="TE",
        coupling={"t": 0.9},
    )
    assert out.power.shape == wl.shape
    metrics = compute_resonance_metrics(wl, out.power)
    assert "quality_factor" in metrics
