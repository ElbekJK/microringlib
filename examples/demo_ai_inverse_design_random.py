import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl

np.random.seed(7)

wl = np.linspace(1540e-9, 1560e-9, 10000)
target_lambda = 1550e-9
target_Q = 30000
target_ER = 10.0

n_iter = 150

best = None
history = []

for i in range(n_iter):
    R = np.random.uniform(8e-6, 14e-6)
    K = np.random.uniform(0.002, 0.05)
    alpha_dbcm = np.random.uniform(1.0, 5.0)

    layers = [
        mrl.Layer(material="Oxide lower", thickness=2e-6, n=1.444, alpha=0),
        mrl.Layer(material="Si core", thickness=220e-9, n=3.476, alpha=mrl.Layer.dbcm_to_npm(alpha_dbcm)),
        mrl.Layer(material="Oxide upper", thickness=2e-6, n=1.444, alpha=0),
    ]

    ring = mrl.RingGeometry(radius=R)
    c = mrl.Coupler.from_power_coupling(K)

    res = mrl.single_mrr_thru(
        wavelengths=wl,
        resonator=ring,
        layers=layers,
        t=c.t,
        kappa=c.kappa,
        polarization="TE",
    )

    m = mrl.compute_resonance_metrics(
        wl,
        res.power,
        target_wavelength=target_lambda,
        warn=False,
    )

    lam = m["resonance_wavelength"]
    Q = m["loaded_Q"]
    ER = m["extinction_ratio_db"]

    if not np.isfinite(Q) or not np.isfinite(ER):
        continue

    loss = (
        ((lam - target_lambda) / 0.2e-9) ** 2
        + ((Q - target_Q) / target_Q) ** 2
        + ((ER - target_ER) / target_ER) ** 2
    )

    history.append(loss)

    if best is None or loss < best["loss"]:
        best = {
            "loss": loss,
            "R": R,
            "K": K,
            "alpha_dbcm": alpha_dbcm,
            "lambda": lam,
            "Q": Q,
            "ER": ER,
            "spectrum": res.power,
        }

print("\n=== AI-style inverse design by random search ===")
print(f"Iterations: {n_iter}")
print(f"Best loss: {best['loss']:.6f}")
print(f"Radius: {best['R'] * 1e6:.4f} um")
print(f"K: {best['K']:.6f}")
print(f"Alpha: {best['alpha_dbcm']:.3f} dB/cm")
print(f"Resonance: {best['lambda'] * 1e9:.6f} nm")
print(f"Loaded Q: {best['Q']:.2f}")
print(f"Extinction ratio: {best['ER']:.2f} dB")

plt.figure()
plt.plot(history)
plt.xlabel("Accepted iteration")
plt.ylabel("Objective loss")
plt.title("Inverse Design Search History")
plt.grid(True)

plt.figure()
plt.plot(wl * 1e9, best["spectrum"])
plt.axvline(target_lambda * 1e9, linestyle="--", label="Target")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Through power")
plt.title("Best Designed Ring Spectrum")
plt.legend()
plt.grid(True)
plt.show()