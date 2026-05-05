#!/usr/bin/env python3
"""
8-channel WDM ring filter bank demo with explicit channel spacing.

This version is publication-friendly:
- uses a user-defined channel spacing,
- targets monotonic channel center wavelengths,
- computes ring radii from a fixed azimuthal mode number,
- searches each resonance only in a narrow local window,
- avoids accidental resonance-order hopping,
- reports target/actual spacing, Q, finesse estimate, center error, and passivity.

Generated figures:
    wdm_8ch_filter_bank_with_spacing.png
    wdm_channel_centers_with_spacing.png
    wdm_channel_spacing_with_spacing.png
    wdm_channel_center_error_with_spacing.png
"""

import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl


def main():
    # ------------------------------------------------------------------
    # Simulation grid
    # ------------------------------------------------------------------
    wl = np.linspace(1535e-9, 1565e-9, 20000)

    # Waveguide / ring model parameters
    n_eff = 3.476
    n_g = 4.2
    alpha_dbcm = 2.0
    base_radius = 10e-6
    K1 = 0.08
    K2 = 0.08

    # ------------------------------------------------------------------
    # Explicit WDM channel plan
    # ------------------------------------------------------------------
    channel_ids = np.arange(1, 9)

    start_wavelength_nm = 1544.0
    channel_spacing_nm = 1.60

    target_centers_nm = start_wavelength_nm + (channel_ids - 1) * channel_spacing_nm

    # Choose a fixed resonance order based on the base radius near mid-band.
    lambda_ref = np.mean(target_centers_nm) * 1e-9
    m0 = int(round(2.0 * np.pi * n_eff * base_radius / lambda_ref))

    # Compute radii so the same azimuthal order m0 lands on each target center.
    radii = m0 * (target_centers_nm * 1e-9) / (2.0 * np.pi * n_eff)

    print("\n=== WDM 8-channel ring filter bank with explicit spacing ===")
    print("\nCoupler physics:")
    print(f"|t1|^2 = {1 - K1:.6f}")
    print(f"|kappa1|^2 = {K1:.6f}")
    print("|t1|^2 + |kappa1|^2 = 1.000000")
    print(f"|t2|^2 = {1 - K2:.6f}")
    print(f"|kappa2|^2 = {K2:.6f}")
    print("|t2|^2 + |kappa2|^2 = 1.000000")

    print("\nDesign plan:")
    print(f"Start wavelength: {start_wavelength_nm:.4f} nm")
    print(f"Channel spacing:  {channel_spacing_nm:.4f} nm")
    print(f"Reference resonance order m0 = {m0}")
    print(f"Target centers (nm): {np.array2string(target_centers_nm, precision=4)}")

    peaks_nm = []
    Qs = []
    finesse_vals = []
    passivity_flags = []

    # Narrow local search avoids hopping to neighboring FSR orders.
    local_window_nm = min(1.2, 0.45 * channel_spacing_nm)

    plt.figure(figsize=(9, 5))

    for i, (R, target_nm) in enumerate(zip(radii, target_centers_nm), start=1):
        thru_field, drop_field, thru, drop = mrl.single_mrr_add_drop_fast(
            wavelengths=wl,
            radius=R,
            n_eff=n_eff,
            alpha_dbcm=alpha_dbcm,
            K1=K1,
            K2=K2,
        )

        total = thru + drop
        passive = bool(np.all(total <= 1.0 + 1e-8))
        passivity_flags.append(passive)

        # Local metric extraction around the designed target center.
        mask = np.abs(wl * 1e9 - target_nm) <= local_window_nm
        wl_local = wl[mask]
        drop_local = drop[mask]

        if wl_local.size < 10:
            raise RuntimeError(f"Local wavelength window too small for CH{i}")

        metrics = mrl.resonance_metrics_fast(
            wl_local,
            drop_local,
            target_wavelength=target_nm * 1e-9,
            kind="peaks",
        )

        peak_nm = float(metrics["resonance_wavelength"] * 1e9)
        fwhm_nm = float(metrics["fwhm"] * 1e9)
        fsr_est_nm = float(mrl.ring_fsr_fast(1550e-9, n_g, R) * 1e9)

        finesse_est = (
            fsr_est_nm / fwhm_nm
            if np.isfinite(fwhm_nm) and fwhm_nm > 0.0
            else np.nan
        )

        peaks_nm.append(peak_nm)
        Qs.append(metrics["loaded_Q"])
        finesse_vals.append(finesse_est)

        error_pm = (peak_nm - target_nm) * 1e3

        print(
            f"CH{i}: "
            f"target = {target_nm:.4f} nm | "
            f"R = {R * 1e6:.4f} um | "
            f"circ = {mrl.ring_circumference_fast(R) * 1e6:.4f} um | "
            f"FSR est = {fsr_est_nm:.4f} nm | "
            f"peak = {peak_nm:.4f} nm | "
            f"error = {error_pm:.2f} pm | "
            f"FWHM = {fwhm_nm:.4f} nm | "
            f"Q = {metrics['loaded_Q']:.1f} | "
            f"finesse est = {finesse_est:.2f} | "
            f"peaks in local window = {metrics['num_resonances_detected']} | "
            f"passive = {passive}"
        )

        plt.plot(wl * 1e9, drop, label=f"CH{i}")

    peaks_nm = np.asarray(peaks_nm, dtype=float)
    Qs = np.asarray(Qs, dtype=float)
    finesse_vals = np.asarray(finesse_vals, dtype=float)

    spacing_nm = np.diff(peaks_nm)
    target_spacing_nm = np.diff(target_centers_nm)
    center_error_pm = (peaks_nm - target_centers_nm) * 1e3
    spacing_error_pm = (spacing_nm - target_spacing_nm) * 1e3

    print("\n=== Channel spacing ===")
    for i, s in enumerate(spacing_nm, start=1):
        print(
            f"CH{i} -> CH{i + 1}: "
            f"{s:.4f} nm "
            f"(target {target_spacing_nm[i - 1]:.4f} nm, "
            f"error {spacing_error_pm[i - 1]:.2f} pm)"
        )

    print("\n=== WDM summary ===")
    print(f"Mean channel spacing:       {np.mean(spacing_nm):.4f} nm")
    print(f"Std channel spacing:        {np.std(spacing_nm):.4f} nm")
    print(f"Min channel spacing:        {np.min(spacing_nm):.4f} nm")
    print(f"Max channel spacing:        {np.max(spacing_nm):.4f} nm")
    print(f"Mean spacing error:         {np.mean(spacing_error_pm):.3f} pm")
    print(f"Max abs spacing error:      {np.max(np.abs(spacing_error_pm)):.3f} pm")
    print(f"Mean center error:          {np.mean(center_error_pm):.3f} pm")
    print(f"Max abs center error:       {np.max(np.abs(center_error_pm)):.3f} pm")
    print(f"Mean loaded Q:              {np.nanmean(Qs):.2f}")
    print(f"Q range:                    {np.nanmin(Qs):.2f} to {np.nanmax(Qs):.2f}")
    print(f"Mean finesse estimate:      {np.nanmean(finesse_vals):.2f}")
    print(f"Finesse estimate range:     {np.nanmin(finesse_vals):.2f} to {np.nanmax(finesse_vals):.2f}")
    print(f"All channels passive:       {all(passivity_flags)}")

    # ------------------------------------------------------------------
    # Plot 1: spectra
    # ------------------------------------------------------------------
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Drop power")
    plt.title(f"8-Channel WDM Ring Filter Bank ({channel_spacing_nm:.2f} nm spacing)")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("wdm_8ch_filter_bank_with_spacing.png", dpi=250)

    # ------------------------------------------------------------------
    # Plot 2: actual channel centers vs target centers
    # ------------------------------------------------------------------
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(channel_ids, target_centers_nm, "s--", label="Target centers")
    plt.plot(channel_ids, peaks_nm, "o-", label="Actual centers")
    plt.xlabel("Channel")
    plt.ylabel("Center wavelength (nm)")
    plt.title("WDM Channel Center Wavelengths")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("wdm_channel_centers_with_spacing.png", dpi=250)

    # ------------------------------------------------------------------
    # Plot 3: adjacent channel spacing
    # ------------------------------------------------------------------
    plt.figure(figsize=(6.5, 4.5))
    pair_ids = np.arange(1, 8)
    plt.plot(pair_ids, spacing_nm, "o-", label="Actual spacing")
    plt.plot(pair_ids, target_spacing_nm, "s--", label="Target spacing")
    plt.xlabel("Adjacent channel pair")
    plt.ylabel("Spacing (nm)")
    plt.title("Adjacent WDM Channel Spacing")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("wdm_channel_spacing_with_spacing.png", dpi=250)

    # ------------------------------------------------------------------
    # Plot 4: center error
    # ------------------------------------------------------------------
    plt.figure(figsize=(6.5, 4.5))
    plt.axhline(0.0, linestyle=":")
    plt.plot(channel_ids, center_error_pm, "o-")
    plt.xlabel("Channel")
    plt.ylabel("Center error (pm)")
    plt.title("WDM Channel Center Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("wdm_channel_center_error_with_spacing.png", dpi=250)

    print("\nSaved:")
    print("  wdm_8ch_filter_bank_with_spacing.png")
    print("  wdm_channel_centers_with_spacing.png")
    print("  wdm_channel_spacing_with_spacing.png")
    print("  wdm_channel_center_error_with_spacing.png")

    plt.show()


if __name__ == "__main__":
    main()