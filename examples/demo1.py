#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import microringlib as mrl


def main():
    # Wavelength sweep around telecom C-band
    wl = np.linspace(1520e-9, 1580e-9, 20001)

    # Approximate SOI strip/rib stack.
    # alpha is power loss in Np/m.
    # Example: 3 dB/cm -> Np/m
    alpha_core = mrl.Layer.dbcm_to_npm(3.0)

    layers = [
        mrl.Layer("SiO2 bottom cladding", thickness=2.0e-6, n=1.444, alpha=0.0),
        mrl.Layer("Si core", thickness=220e-9, n=3.476, alpha=alpha_core),
        mrl.Layer("SiO2 top cladding", thickness=2.0e-6, n=1.444, alpha=0.0),
    ]

    ring = mrl.RingGeometry(kind="circular", radius=10e-6)

    # Physics-first coupler construction:
    # K = power coupling coefficient.
    c1 = mrl.Coupler.from_power_coupling(K=0.12)
    c2 = mrl.Coupler.from_power_coupling(K=0.12)

    # Modal confinement factors: most optical power in silicon core.
    # These should come from a real mode solver/measurement in serious work.
    overlap_factors = [0.05, 0.90, 0.05]

    result = mrl.single_mrr_add_drop(
        wavelengths=wl,
        resonator=ring,
        layers=layers,
        T=25.0,
        polarization="TE",
        t1=c1.t,
        kappa1=c1.kappa,
        t2=c2.t,
        kappa2=c2.kappa,
        overlap_factors=overlap_factors,
    )

    thru = result.ports["through"]["power"]
    drop = result.ports["drop"]["power"]
    total = thru + drop

    print("\n=== Passive energy check ===")
    print(f"max(P_thru + P_drop) = {np.max(total):.8f}")
    print(f"min(P_thru + P_drop) = {np.min(total):.8f}")
    print("Passes passive constraint:", np.all(total <= 1.0 + 1e-8))

    thru_metrics = mrl.compute_resonance_metrics(wl, thru)
    drop_metrics = mrl.compute_resonance_metrics(wl, drop, resonance_kind="peaks")

    fsr_est = mrl.ring_fsr(
        wavelength=1550e-9,
        n_g=4.2,
        ring=ring,
    )

    print("\n== = Ring design estimate ===")
    print(f"Radius: {ring.radius * 1e6:.2f} µm")
    print(f"Circumference: {mrl.ring_circumference(ring) * 1e6:.2f} µm")
    print(f"Estimated FSR using ng=4.2: {fsr_est * 1e9:.3f} nm")

    print("\n=== Through-port resonance metrics ===")
    for k, v in thru_metrics.items():
        if v is None:
            print(f"{k}: None")
        elif isinstance(v, np.ndarray):
            print(f"{k}: {v}")
        elif "wavelength" in k or k in {"fwhm", "fsr"}:
            print(f"{k}: {float(v) * 1e9:.6f} nm")
        elif isinstance(v, (float, int, np.floating, np.integer)):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")

    print("\n=== Coupler check ===")
    print(f"|t|^2 = {abs(c1.t)**2:.6f}")
    print(f"|kappa|^2 = {abs(c1.kappa)**2:.6f}")
    print(f"|t|^2 + |kappa|^2 = {abs(c1.t)**2 + abs(c1.kappa)**2:.6f}")
    print("Scattering matrix:")
    print(c1.scattering_matrix)

    tau_thru = mrl.compute_group_delay(wl, result.ports["through"]["field"])

    plt.figure()
    plt.plot(wl * 1e9, thru, label="Through")
    plt.plot(wl * 1e9, drop, label="Drop")
    plt.plot(wl * 1e9, total, "--", label="Through + Drop")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power transmission")
    plt.title("Add-drop microring response")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("real_life_microring_response.png", dpi=200)

    plt.figure()
    plt.plot(wl * 1e9, tau_thru * 1e12)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Group delay (ps)")
    plt.title("Through-port group delay")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("real_life_group_delay.png", dpi=200)

    print("\nSaved:")
    print("  real_life_microring_response.png")
    print("  real_life_group_delay.png")


if __name__ == "__main__":
    main()