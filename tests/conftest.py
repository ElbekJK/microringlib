import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)

import pytest

from microringlib import Layer


@pytest.fixture
def layers():
    return [
        Layer("silica", 2e-6, 1.45, dn_dT=1e-5, alpha=10.0),
        Layer("silicon", 220e-9, 3.48, dn_dT=1.8e-4, alpha=100.0),
        Layer("silica", 2e-6, 1.45, dn_dT=1e-5, alpha=10.0),
    ]


@pytest.fixture
def wavelengths():
    return np.linspace(1540e-9, 1560e-9, 101)


@pytest.fixture
def ring():
    from microringlib import RingGeometry
    return RingGeometry(kind="circular", radius=10e-6)