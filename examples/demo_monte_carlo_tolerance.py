import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl


def main():
    np.random.seed(4)
    n_trials = 100_000
    R0 = 10e-6
    sigma_R = 5e-9
    n0 = 3.476
    sigma_n = 1e-4
    target_wavelength = 1550e-9

    R_samples = R0 + np.random.randn(n_trials) * sigma_R
    n_samples = n0 + np.random.randn(n_trials) * sigma_n

    out = mrl.monte_carlo_resonance_formula_fast(
        n_eff_samples=n_samples,
        radius_samples=R_samples,
        target_wavelength=target_wavelength,
        n_g=4.2,
        loaded_Q_nominal=5.884369e4,
        extinction_ratio_db_nominal=4.55,
    )
    resonances = out["resonance_wavelength_nm"]
    Qs = out["loaded_Q"]
    ERs = out["extinction_ratio_db"]

    print("\n=== Monte Carlo fabrication tolerance ===")
    print(f"Trials: {n_trials}")
    print(f"Resonance mean: {np.mean(resonances):.4f} nm")
    print(f"Resonance std:  {np.std(resonances):.4f} nm")
    print(f"Q mean: {np.nanmean(Qs):.2f}")
    print(f"Q std:  {np.nanstd(Qs):.2f}")
    print(f"ER mean: {np.nanmean(ERs):.2f} dB")
    print(f"ER std:  {np.nanstd(ERs):.2f} dB")

    plt.figure(figsize=(7, 4.5))
    plt.hist(resonances, bins=45)
    plt.xlabel("Tracked resonance wavelength (nm)")
    plt.ylabel("Count")
    plt.title("Fabrication-Induced Resonance Spread")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("monte_carlo_tolerance.png", dpi=200)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(resonances, Qs, s=2, alpha=0.2)
    plt.xlabel("Resonance wavelength (nm)")
    plt.ylabel("Loaded Q")
    plt.title("Q vs Resonance Shift")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("monte_carlo_q_vs_resonance.png", dpi=200)

    print("\nSaved:")
    print("  monte_carlo_tolerance.png")
    print("  monte_carlo_q_vs_resonance.png")
    plt.show()


if __name__ == "__main__":
    main()
