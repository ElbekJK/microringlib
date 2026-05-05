import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "examples_publication"))
sys.path.insert(0, str(PROJECT))

from shared.sic_experiment_helpers import ring_q_proxy


def test_ring_q_proxy_bounded_for_broad_residual():
    wl = np.linspace(1530e-9, 1570e-9, 281)
    power = (
        0.95
        + 0.03 * np.sin(np.linspace(0, 8 * np.pi, wl.size))
        + 0.015 * np.cos(np.linspace(0, 3 * np.pi, wl.size))
    )
    power -= 0.02 * np.exp(-0.5 * ((wl - 1550e-9) / (0.18e-9)) ** 2)

    _lam, q, _baseline, _dip, method, linewidth = ring_q_proxy(wl, power, 1550e-9)

    assert np.isfinite(q)
    assert q > 10.0
    assert np.isfinite(linewidth)
    assert linewidth <= np.ptp(wl) / 3.0 + 1e-18
    assert method in {
        "half_depth_crossing",
        "local_equivalent_notch_area",
        "bounded_compact_prior",
        "bounded_resolution_floor",
    }
