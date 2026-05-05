import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl


def main():
    wl = np.linspace(1520e-9, 1580e-9, 20001)
    n_eff = 2.60
    alpha_dbcm = 1.0
    radius = 25e-6
    K1 = 0.04
    K2 = 0.04

    _, drop_field, _, drop = mrl.single_mrr_add_drop_fast(
        wavelengths=wl,
        radius=radius,
        n_eff=n_eff,
        alpha_dbcm=alpha_dbcm,
        K1=K1,
        K2=K2,
    )

    metrics = mrl.resonance_metrics_fast(
        wavelengths=wl,
        power=drop,
        target_wavelength=1550e-9,
        kind="peaks",
    )

    Q = metrics["loaded_Q"]
    lam_p = metrics["resonance_wavelength"]

    gamma = 2.0  # 1/W/m placeholder for SiC-like waveguide nonlinearity

    # Include zero for the linear-scale plot.
    Pp = np.linspace(0, 20e-3, 200)

    relative_pair_rate = mrl.sfwm_pair_rate_relative_fast(
        pump_power=Pp,
        gamma=gamma,
        loaded_Q=Q,
        ring_radius=radius,
        normalize=True,
    )

    # Exclude zero power for log-log analysis.
    positive = (Pp > 0) & (relative_pair_rate > 0)
    Pp_pos = Pp[positive]
    rate_pos = relative_pair_rate[positive]

    # Fit log10(rate) = slope * log10(P) + intercept.
    slope, intercept = np.polyfit(
        np.log10(Pp_pos),
        np.log10(rate_pos),
        deg=1,
    )

    # Reference slope-2 curve normalized to match the final simulated point.
    reference_slope2 = rate_pos[-1] * (Pp_pos / Pp_pos[-1]) ** 2

    print("\n=== SiC ring SFWM photon-pair demo ===")
    print("Relative pair-rate model, not absolute calibrated quantum simulation.")
    print(f"Tracked pump resonance: {lam_p * 1e9:.4f} nm")
    print(f"Loaded Q: {Q:.2f}")
    print(f"Ring radius: {radius * 1e6:.2f} um")
    print(f"Relative pair rate at 20 mW: {relative_pair_rate[-1]:.3f}")

    print("\n=== SFWM scaling check ===")
    print("Expected spontaneous four-wave-mixing pair-rate scaling: R ∝ P_p^2")
    print(f"Fitted log-log slope: {slope:.4f}")
    print(f"Slope error from 2: {slope - 2.0:+.4e}")

    print("\n=== Fast model parameters ===")
    print(f"n_eff approximation: {n_eff:.4f}")
    print(f"Propagation loss: {alpha_dbcm:.3f} dB/cm")
    print(f"K1: {K1:.4f}")
    print(f"K2: {K2:.4f}")
    print(f"Detected resonances: {metrics['num_resonances_detected']}")
    print(f"FWHM: {metrics['fwhm'] * 1e9:.6f} nm")
    print(f"FSR: {metrics['fsr'] * 1e9:.6f} nm")
    print(f"Finesse: {metrics['finesse']:.3f}")
    print(f"Drop ER: {metrics['extinction_ratio_db']:.3f} dB")

    # ------------------------------------------------------------------
    # Plot 1: drop-port resonance
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    plt.plot(wl * 1e9, drop)
    plt.axvline(lam_p * 1e9, linestyle="--", label="tracked pump resonance")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Drop power")
    plt.title("SiC Ring Drop Resonance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sic_ring_drop_resonance.png", dpi=250)

    # ------------------------------------------------------------------
    # Plot 2: linear-scale pair-rate scaling
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    plt.plot(Pp * 1e3, relative_pair_rate)
    plt.xlabel("Pump power (mW)")
    plt.ylabel("Relative photon-pair rate")
    plt.title("SFWM Pair Generation Scaling")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sic_sfwm_pair_rate.png", dpi=250)

    # ------------------------------------------------------------------
    # Plot 3: log-log pair-rate scaling with slope-2 reference
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    plt.loglog(Pp_pos * 1e3, rate_pos, "o", markersize=4, label="microringlib fast model")
    plt.loglog(
        Pp_pos * 1e3,
        reference_slope2,
        "--",
        label=rf"reference $\propto P_p^2$",
    )
    plt.xlabel("Pump power (mW)")
    plt.ylabel("Relative photon-pair rate")
    plt.title(r"SFWM Log-Log Scaling: $R \propto P_p^2$")
    plt.grid(True, which="both")
    plt.legend(title=f"fit slope = {slope:.3f}")
    plt.tight_layout()
    plt.savefig("sic_sfwm_pair_rate_loglog.png", dpi=250)

    print("\nSaved:")
    print("  sic_ring_drop_resonance.png")
    print("  sic_sfwm_pair_rate.png")
    print("  sic_sfwm_pair_rate_loglog.png")

    plt.show()


if __name__ == "__main__":
    main()