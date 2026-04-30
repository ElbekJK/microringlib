# microringlib

**microringlib** is a physics-first Python library for integrated-photonics microring resonators. It provides physically constrained couplers, passive power checks, resonance metrics, material-dispersion backends, accelerated analytical sweeps, nonlinear reduced models, and quantum-photonics scaling demos.

Author: **Elbek J Keskinoglu**  
Email: **ejkeskinoglu@connect.ust.hk**

---

## Why this library?

Many research scripts can generate plausible microring spectra, but they often leave important physical checks to post-processing. `microringlib` tries to make the safe behavior the default:

- lossless couplers satisfy `|t|^2 + |kappa|^2 = 1`,
- passive add-drop rings satisfy `P_thru + P_drop <= 1`,
- field and power responses are kept separate,
- resonance metrics have explicit physical meaning,
- parameter sweeps can track the same resonance order,
- fast analytical helpers are available for large sweeps and Monte Carlo studies.

The library is intentionally lightweight. It is not a replacement for full-wave tools such as Meep or Tidy3D, nor for layout/PDK tools such as gdsfactory. Instead, it sits between analytic theory and large simulation frameworks: fast enough for design exploration, explicit enough for physical interpretation.

---

## Main features

### Core photonics

- straight waveguide transmission
- single-bus all-pass microring resonators
- add-drop microring resonators
- cascaded add-drop rings
- WDM ring filter banks
- group-delay / slow-light extraction
- thermal tuning via `dn_dT`

### Physics-first checks

- unitary coupler construction from power coupling `K`
- passive add-drop energy-budget validation
- field/power separation
- named output ports for through/drop devices

### Resonance metrics

- resonance wavelength
- FWHM linewidth
- FSR / peak spacing
- loaded Q
- finesse
- extinction ratio
- target-wavelength resonance tracking
- resonance counting for sanity checks

### Material models

- constant refractive index
- tabulated `n(lambda)` and `k(lambda)`
- function-backed complex index
- optional `refractiveindex` wrapper
- optional PyOptik object adapter
- material absorption from `alpha = 4*pi*k/lambda`

### Accelerated reduced models

The `microringlib.fast` helpers are vectorized analytical shortcuts for large sweeps:

- `single_mrr_thru_fast`
- `single_mrr_thru_fast_batch`
- `single_mrr_add_drop_fast`
- `resonance_metrics_fast`
- `monte_carlo_resonance_formula_fast`
- `sfwm_pair_rate_relative_fast`

These helpers use user-supplied `n_eff` and loss approximations. Use the full `transfer.py` APIs when you want the layer/mode workflow; use `fast.py` for rapid figure generation, parameter sweeps, and Monte Carlo studies.

### Research extensions

- reduced Kerr bistability model in `nonlinear.py`
- reduced SFWM photon-pair scaling model in `quantum.py`
- frequency-comb toy demo
- ring-modulator eye diagram demo
- fabrication-tolerance Monte Carlo demo

---

## Installation

From the project root:

```bash
pip install -e .
```

Optional material database support:

```bash
pip install -e ".[materials]"
```

Development install:

```bash
pip install -e ".[dev,materials]"
pytest -q
```

---

## Quick start: through-port microring

```python
import numpy as np
import microringlib as mrl

wl = np.linspace(1520e-9, 1580e-9, 20001)

layers = [
    mrl.Layer(material="SiO2 lower", thickness=2e-6, n=1.444, alpha=0),
    mrl.Layer(material="Si core", thickness=220e-9, n=3.476,
              dn_dT=1.86e-4, alpha=mrl.Layer.dbcm_to_npm(2.0)),
    mrl.Layer(material="SiO2 upper", thickness=2e-6, n=1.444, alpha=0),
]

ring = mrl.RingGeometry(radius=10e-6)
c = mrl.Coupler.from_power_coupling(0.01)

res = mrl.single_mrr_thru(
    wavelengths=wl,
    resonator=ring,
    layers=layers,
    t=c.t,
    kappa=c.kappa,
    polarization="TE",
)

metrics = mrl.compute_resonance_metrics(
    wl,
    res.power,
    target_wavelength=1550e-9,
)

print(metrics["resonance_wavelength"] * 1e9, "nm")
print(metrics["loaded_Q"])
```

---

## Fast sweep example

```python
import numpy as np
import microringlib as mrl

wl = np.linspace(1520e-9, 1580e-9, 20001)
K_values = np.array([0.005, 0.01, 0.02, 0.04, 0.08])

fields, powers, t_values, kappa_values = mrl.single_mrr_thru_fast_batch(
    wavelengths=wl,
    radius=10e-6,
    n_eff=3.476,
    alpha_dbcm=3.0,
    K_values=K_values,
)

for K, power in zip(K_values, powers):
    m = mrl.resonance_metrics_fast(
        wl,
        power,
        target_wavelength=1555e-9,
        kind="dips",
    )
    print(K, m["loaded_Q"], m["fwhm"] * 1e9)
```

---

## Add-drop passivity example

```python
c1 = mrl.Coupler.from_power_coupling(0.12)
c2 = mrl.Coupler.from_power_coupling(0.12)

out = mrl.single_mrr_add_drop(
    wavelengths=wl,
    resonator=ring,
    layers=layers,
    t1=c1.t,
    kappa1=c1.kappa,
    t2=c2.t,
    kappa2=c2.kappa,
    overlap_factors=[0.05, 0.90, 0.05],
)

thru = out.ports["through"]["power"]
drop = out.ports["drop"]["power"]

print(np.max(thru + drop))
```

A passive simulation should report a maximum total output power no larger than 1, up to numerical tolerance.

---

## Material backends

### Constant material

```python
si = mrl.ConstantMaterial("Si", n=3.476, k=0.0, dn_dT=1.86e-4)
sio2 = mrl.ConstantMaterial("SiO2", n=1.444, k=0.0, dn_dT=1.0e-5)

layers = [
    mrl.Layer("SiO2 lower", 2e-6, material_model=sio2),
    mrl.Layer("Si core", 220e-9, material_model=si,
              alpha=mrl.Layer.dbcm_to_npm(2.0)),
    mrl.Layer("SiO2 upper", 2e-6, material_model=sio2),
]
```

### Tabulated material

```python
wl_table = np.array([1.50, 1.55, 1.60]) * 1e-6
n_table = np.array([3.485, 3.476, 3.468])

si_tab = mrl.TabulatedMaterial(
    name="Si measured",
    wavelength_m=wl_table,
    n=n_table,
    dn_dT=1.86e-4,
)
```

### Optional refractive-index database adapter

```python
si = mrl.RefractiveIndexInfoMaterial(
    shelf="main",
    book="Si",
    page="Green-2008",
    dn_dT=1.86e-4,
)
```

The optional package is imported lazily. If it is not installed, the adapter raises a clear `ImportError` only when evaluated.

---

## Included research demos

Run from the repository root:

```bash
PYTHONPATH=. python demo2_critical_coupling_metrics.py
PYTHONPATH=. python demo_wdm_8ch_filter_bank_with_spacing.py
PYTHONPATH=. python demo_monte_carlo_tolerance.py
PYTHONPATH=. python demo_ring_modulator_eye.py
PYTHONPATH=. python demo_kerr_bistability.py
PYTHONPATH=. python demo_sic_sfwm_photon_pairs.py
```

Important generated figures include:

- `real_life_microring_response.png`
- `critical_coupling_sweep.png`
- `critical_coupling_metrics.png`
- `critical_coupling_secondary_metrics.png`
- `tracked_thermal_tuning.png`
- `real_life_group_delay.png`
- `wdm_8ch_filter_bank.png`
- `wdm_channel_centers.png`
- `wdm_channel_spacing.png`
- `monte_carlo_tolerance.png`
- `monte_carlo_q_vs_resonance.png`
- `ring_modulator_eye.png`
- `kerr_bistability.png`
- `sic_ring_drop_resonance.png`
- `sic_sfwm_pair_rate.png`

---

## Notes on nonlinear and quantum modules

The nonlinear and quantum modules are deliberately reduced models:

- `nonlinear.py` gives a steady-state single-mode Kerr cavity model, not a full Lugiato-Lefever solver.
- `quantum.py` gives SFWM scaling and toy joint-spectral-amplitude tools, not an absolute calibrated photon-pair-source simulator.

They are useful for design trends, teaching, and early-stage research exploration. Full calibrated predictions should be benchmarked against experiment or higher-fidelity solvers.

---

## Recommended external tools

`microringlib` is complementary to:

- **gdsfactory** for layout and PDK workflows,
- **Meep** for open-source FDTD,
- **Tidy3D** for Python/cloud FDTD workflows,
- **SAX / Simphony** for S-parameter circuit simulation.

---

## License

MIT License.

---

## Citation

If you use this package in research, please cite the GitHub repository and the standard microring/coupled-mode references relevant to your work, such as Yariv, Little et al., Bogaerts et al., Haus, Agrawal, and Helt/Liscidini/Sipe-style SFWM scaling literature.
