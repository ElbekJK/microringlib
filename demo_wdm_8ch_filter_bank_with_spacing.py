import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl


def main():
    wl = np.linspace(1535e-9, 1565e-9, 20000)
    n_eff = 3.476
    n_g = 4.2
    alpha_dbcm = 2.0
    base_radius = 10e-6
    radii = base_radius + np.linspace(-0.35e-6, 0.35e-6, 8)
    K1 = 0.08
    K2 = 0.08

    print("\n=== WDM 8-channel ring filter bank ===")
    print("\nCoupler physics:")
    print(f"|t1|^2 = {1-K1:.6f}")
    print(f"|kappa1|^2 = {K1:.6f}")
    print(f"|t1|^2 + |kappa1|^2 = 1.000000")
    print(f"|t2|^2 = {1-K2:.6f}")
    print(f"|kappa2|^2 = {K2:.6f}")
    print(f"|t2|^2 + |kappa2|^2 = 1.000000")

    peaks_nm, Qs = [], []
    channel_ids = np.arange(1, 9)

    plt.figure(figsize=(9, 5))
    for i, R in enumerate(radii, start=1):
        target_nm = 1543.8 + (i - 1) * 1.66
        thru_field, drop_field, thru, drop = mrl.single_mrr_add_drop_fast(
            wavelengths=wl,
            radius=R,
            n_eff=n_eff,
            alpha_dbcm=alpha_dbcm,
            K1=K1,
            K2=K2,
        )
        total = thru + drop
        metrics = mrl.resonance_metrics_fast(
            wl,
            drop,
            target_wavelength=target_nm * 1e-9,
            kind="peaks",
        )
        peak_nm = metrics["resonance_wavelength"] * 1e9
        peaks_nm.append(peak_nm)
        Qs.append(metrics["loaded_Q"])
        print(
            f"CH{i}: R = {R * 1e6:.4f} um | "
            f"circ = {mrl.ring_circumference_fast(R) * 1e6:.4f} um | "
            f"FSR est = {mrl.ring_fsr_fast(1550e-9, n_g, R) * 1e9:.4f} nm | "
            f"peak = {peak_nm:.4f} nm | "
            f"FWHM = {metrics['fwhm'] * 1e9:.4f} nm | "
            f"Q = {metrics['loaded_Q']:.1f} | "
            f"finesse = {metrics['finesse']:.2f} | "
            f"peaks = {metrics['num_resonances_detected']} | "
            f"passive = {np.all(total <= 1 + 1e-8)}"
        )
        plt.plot(wl * 1e9, drop, label=f"CH{i}")

    peaks_nm = np.asarray(peaks_nm)
    Qs = np.asarray(Qs)
    spacing_nm = np.diff(peaks_nm)

    print("\n=== Channel spacing ===")
    for i, s in enumerate(spacing_nm, start=1):
        print(f"CH{i} -> CH{i + 1}: {s:.4f} nm")
    print("\n=== WDM summary ===")
    print(f"Mean channel spacing: {np.mean(spacing_nm):.4f} nm")
    print(f"Std channel spacing:  {np.std(spacing_nm):.4f} nm")
    print(f"Min channel spacing:  {np.min(spacing_nm):.4f} nm")
    print(f"Max channel spacing:  {np.max(spacing_nm):.4f} nm")
    print(f"Mean loaded Q:        {np.nanmean(Qs):.2f}")
    print(f"Q range:              {np.nanmin(Qs):.2f} to {np.nanmax(Qs):.2f}")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Drop power")
    plt.title("8-Channel WDM Ring Filter Bank")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("wdm_8ch_filter_bank.png", dpi=200)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(channel_ids, peaks_nm, "o-")
    plt.xlabel("Channel")
    plt.ylabel("Peak wavelength (nm)")
    plt.title("WDM Channel Center Wavelengths")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("wdm_channel_centers.png", dpi=200)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(np.arange(1, 8), spacing_nm, "o-")
    plt.xlabel("Adjacent channel pair")
    plt.ylabel("Spacing (nm)")
    plt.title("Adjacent WDM Channel Spacing")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("wdm_channel_spacing.png", dpi=200)

    print("\nSaved:")
    print("  wdm_8ch_filter_bank.png")
    print("  wdm_channel_centers.png")
    print("  wdm_channel_spacing.png")
    plt.show()


if __name__ == "__main__":
    main()
