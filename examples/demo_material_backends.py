import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl

wl = np.linspace(1520e-9, 1580e-9, 20001)

# Example of a wavelength-dependent silicon fit using tabulated n(lambda).
si = mrl.TabulatedMaterial(
    name="Si tabulated demo",
    wavelength_m=np.array([1.50, 1.55, 1.60]) * 1e-6,
    n=np.array([3.485, 3.476, 3.468]),
    k=np.array([0.0, 0.0, 0.0]),
    dn_dT=1.86e-4,
)

sio2 = mrl.ConstantMaterial("SiO2", n=1.444, k=0.0, dn_dT=1.0e-5)

layers = [
    mrl.Layer("SiO2 lower", 2e-6, material_model=sio2),
    mrl.Layer("Si core", 220e-9, material_model=si, alpha=mrl.Layer.dbcm_to_npm(2.0)),
    mrl.Layer("SiO2 upper", 2e-6, material_model=sio2),
]

ring = mrl.RingGeometry(radius=10e-6)
c = mrl.Coupler.from_power_coupling(0.01)

print("\n=== Material backend demo ===")
print("Si n(1520 nm):", np.real(si.n_complex(1520e-9)))
print("Si n(1550 nm):", np.real(si.n_complex(1550e-9)))
print("Si n(1580 nm):", np.real(si.n_complex(1580e-9)))
print("Core extra alpha:", mrl.Layer.dbcm_to_npm(2.0), "Np/m")

res = mrl.single_mrr_thru(
    wavelengths=wl,
    resonator=ring,
    layers=layers,
    t=c.t,
    kappa=c.kappa,
    polarization="TE",
)

metrics = mrl.compute_resonance_metrics(wl, res.power, target_wavelength=1550e-9)
print("Tracked resonance:", metrics["resonance_wavelength"] * 1e9, "nm")
print("Loaded Q:", metrics["loaded_Q"])
print("Resonances detected:", metrics["num_resonances_detected"])

plt.plot(wl * 1e9, res.power)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Through power")
plt.title("Microring with TabulatedMaterial dispersion")
plt.grid(True)
plt.show()
