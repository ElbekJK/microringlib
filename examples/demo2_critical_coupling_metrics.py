import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl


def main():
    wl = np.linspace(1520e-9, 1580e-9, 20001)
    radius = 10e-6
    n_eff = 3.476
    n_g = 4.2
    alpha_dbcm = 3.0
    K_values = np.array([0.005, 0.010, 0.020, 0.040, 0.060, 0.080])
    target_wavelength = 1555e-9

    print("\n=== Ring geometry ===")
    print(f"Radius: {radius * 1e6:.3f} um")
    print(f"Circumference: {mrl.ring_circumference_fast(radius) * 1e6:.3f} um")
    print(f"Estimated FSR at 1550 nm, ng=4.2: {mrl.ring_fsr_fast(1550e-9, n_g, radius) * 1e9:.3f} nm")

    fields, powers, t_values, kappa_values = mrl.single_mrr_thru_fast_batch(
        wavelengths=wl,
        radius=radius,
        n_eff=n_eff,
        alpha_dbcm=alpha_dbcm,
        K_values=K_values,
    )

    Tmins, Qs, FWHMs_nm, ERs = [], [], [], []

    plt.figure(figsize=(8, 5))
    for i, K in enumerate(K_values):
        power = powers[i]
        field = fields[i]
        metrics = mrl.resonance_metrics_fast(
            wl,
            power,
            target_wavelength=target_wavelength,
            kind="dips",
        )
        print(f"\n=== Through-ring coupling sweep: K = {K:.3f} ===")
        print("Coupler physics:")
        print(f"|t|^2 = {abs(t_values[i])**2:.6f}")
        print(f"|kappa|^2 = {abs(kappa_values[i])**2:.6f}")
        print(f"|t|^2 + |kappa|^2 = {abs(t_values[i])**2 + abs(kappa_values[i])**2:.6f}")
        print("\nTransmission:")
        print(f"Through min: {np.min(power):.6g}")
        print(f"Through max: {np.max(power):.6g}")
        print(f"Passive check P <= 1: {np.all(power <= 1 + 1e-8)}")
        phase = np.unwrap(np.angle(field))
        print(f"Phase span: {float(phase[-1] - phase[0]):.6g} rad")
        print("\nResonance metrics:")
        print(f"Resonance wavelength: {metrics['resonance_wavelength'] * 1e9:.6f} nm")
        print(f"FWHM: {metrics['fwhm'] * 1e9:.6f} nm")
        print(f"FSR: {metrics['fsr'] * 1e9:.6f} nm")
        print(f"Loaded Q: {metrics['loaded_Q']:.3f}")
        print(f"Finesse: {metrics['finesse']:.3f}")
        print(f"Extinction ratio: {metrics['extinction_ratio_db']:.3f} dB")
        print(f"Resonances detected: {metrics['num_resonances_detected']}")

        Tmins.append(np.min(power))
        Qs.append(metrics["loaded_Q"])
        FWHMs_nm.append(metrics["fwhm"] * 1e9)
        ERs.append(metrics["extinction_ratio_db"])
        plt.plot(wl * 1e9, power, label=f"K={K:.3f}")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Through power")
    plt.title("Critical Coupling Sweep")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("critical_coupling_sweep.png", dpi=200)

    Tmins = np.asarray(Tmins)
    Qs = np.asarray(Qs)
    FWHMs_nm = np.asarray(FWHMs_nm)
    ERs = np.asarray(ERs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(K_values, Tmins, "o-")
    axes[0].set_xlabel("Power coupling K")
    axes[0].set_ylabel("Minimum through power")
    axes[0].set_title("Critical-Coupling Trend")
    axes[0].grid(True)
    axes[1].plot(K_values, Qs, "o-")
    axes[1].set_xlabel("Power coupling K")
    axes[1].set_ylabel("Loaded Q")
    axes[1].set_title("Loaded Q vs Coupling")
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("critical_coupling_metrics.png", dpi=200)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(K_values, FWHMs_nm, "o-")
    axes[0].set_xlabel("Power coupling K")
    axes[0].set_ylabel("FWHM (nm)")
    axes[0].set_title("Linewidth Broadening")
    axes[0].grid(True)
    axes[1].plot(K_values, ERs, "o-")
    axes[1].set_xlabel("Power coupling K")
    axes[1].set_ylabel("Extinction ratio (dB)")
    axes[1].set_title("Extinction Ratio vs Coupling")
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("critical_coupling_secondary_metrics.png", dpi=200)

    print("\nSaved:")
    print("  critical_coupling_sweep.png")
    print("  critical_coupling_metrics.png")
    print("  critical_coupling_secondary_metrics.png")
    plt.show()


if __name__ == "__main__":
    main()
