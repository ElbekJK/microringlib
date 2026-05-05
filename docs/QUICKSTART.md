# Quick start

This page shows a minimal workflow for a through-port microring.

```python
import numpy as np
import microringlib as mrl

wl = np.linspace(1520e-9, 1580e-9, 2001)

layers = [
    mrl.Layer(material="Si core", thickness=220e-9, n=3.476, alpha=0.0),
    mrl.Layer(material="SiO2 cladding", thickness=2e-6, n=1.444, alpha=0.0),
]

ring = mrl.RingGeometry(radius=10e-6)
coupler = mrl.Coupler.from_power_coupling(0.08)

out = mrl.single_mrr_thru(
    wavelengths=wl,
    resonator=ring,
    layers=layers,
    t=coupler.t,
    kappa=coupler.kappa,
)

print("minimum power:", out.power.min())
```

## Key conventions

- Wavelengths are in meters.
- Ring radius is in meters.
- Power coupling `K` means `|kappa|^2`.
- Passive checks should satisfy `P_through + P_drop <= 1` when loss is included.
- `n_eff` controls resonance phase.
- `n_g` controls free spectral range.

## Transfer/scattering matrix example

```python
import numpy as np
from microringlib import tmm

wl = np.linspace(1520e-9, 1580e-9, 2001)
neff = 2.42
alpha = tmm.dbcm_to_npm(2.0)

wg = tmm.waveguide_smatrix(wl, neff, length=500e-6, alpha_power=alpha)
ring = tmm.ring_allpass_smatrix(wl, neff, radius=10e-6, K=0.1, alpha_power=alpha)

circuit = tmm.Cascade([wg, ring], name="wg_plus_ring").solve()
through_power = circuit.power("in", "out")
assert circuit.is_passive()
```
