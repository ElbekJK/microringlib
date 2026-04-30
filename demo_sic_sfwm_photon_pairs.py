import numpy as np
import matplotlib.pyplot as plt


wl = np.linspace(1520e-9, 1580e-9, 20001)


def dbcm_to_npm(alpha_db_cm):
    """
    Convert power loss from dB/cm to Neper/m.
    """
    return alpha_db_cm * 100.0 / 4.343


def ring_circumference(radius):
    return 2.0 * np.pi * radius


def interpolate_crossing(x1, y1, x2, y2, y_target):
    """
    Linear interpolation of x where y crosses y_target.
    """
    if y2 == y1:
        return 0.5 * (x1 + x2)

    return x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)


def detect_resonance_maxima(wavelengths, power):
    """
    Detect local maxima for drop-port resonances.
    """
    if len(power) < 3:
        return np.array([], dtype=int)

    idx = np.where(
        (power[1:-1] > power[:-2]) &
        (power[1:-1] >= power[2:])
    )[0] + 1

    return idx


def compute_peak_metrics_fast(wavelengths, power, target_wavelength):
    """
    Fast resonance metric extraction for peak-type resonances.
    """
    maxima = detect_resonance_maxima(wavelengths, power)

    if len(maxima) == 0:
        return {
            "resonance_wavelength": np.nan,
            "resonance_index": -1,
            "fwhm": np.nan,
            "fsr": np.nan,
            "loaded_Q": np.nan,
            "finesse": np.nan,
            "extinction_ratio_db": np.nan,
            "num_resonances_detected": 0,
            "resonance_indices": maxima,
        }

    tracked_idx = maxima[np.argmin(np.abs(wavelengths[maxima] - target_wavelength))]
    lam0 = wavelengths[tracked_idx]
    peak_val = power[tracked_idx]

    pos = np.where(maxima == tracked_idx)[0][0]

    fsr_values = []
    if pos > 0:
        fsr_values.append(lam0 - wavelengths[maxima[pos - 1]])
    if pos < len(maxima) - 1:
        fsr_values.append(wavelengths[maxima[pos + 1]] - lam0)

    fsr = np.mean(fsr_values) if len(fsr_values) > 0 else np.nan

    left_bound = maxima[pos - 1] if pos > 0 else 0
    right_bound = maxima[pos + 1] if pos < len(maxima) - 1 else len(power) - 1

    local_min = np.min(power[left_bound:right_bound + 1])

    if local_min <= 0 or peak_val <= 0:
        er_db = np.nan
    else:
        er_db = 10.0 * np.log10(peak_val / local_min)

    half_level = local_min + 0.5 * (peak_val - local_min)

    left_cross = np.nan
    for i in range(tracked_idx - 1, left_bound - 1, -1):
        if power[i] <= half_level and power[i + 1] > half_level:
            left_cross = interpolate_crossing(
                wavelengths[i],
                power[i],
                wavelengths[i + 1],
                power[i + 1],
                half_level,
            )
            break

    right_cross = np.nan
    for i in range(tracked_idx, right_bound):
        if power[i] > half_level and power[i + 1] <= half_level:
            right_cross = interpolate_crossing(
                wavelengths[i],
                power[i],
                wavelengths[i + 1],
                power[i + 1],
                half_level,
            )
            break

    if np.isfinite(left_cross) and np.isfinite(right_cross) and right_cross > left_cross:
        fwhm = right_cross - left_cross
        loaded_Q = lam0 / fwhm
    else:
        fwhm = np.nan
        loaded_Q = np.nan

    finesse = fsr / fwhm if np.isfinite(fsr) and np.isfinite(fwhm) and fwhm > 0 else np.nan

    return {
        "resonance_wavelength": lam0,
        "resonance_index": int(tracked_idx),
        "fwhm": fwhm,
        "fsr": fsr,
        "loaded_Q": loaded_Q,
        "quality_factor": loaded_Q,
        "finesse": finesse,
        "extinction_ratio_db": er_db,
        "num_resonances_detected": len(maxima),
        "resonance_indices": maxima,
    }


def single_mrr_add_drop_fast(
    wavelengths,
    radius,
    n_eff,
    alpha_dbcm,
    K1,
    K2,
):
    """
    Fast analytical add-drop microring model.

    Returns:
        through_field, drop_field, through_power, drop_power
    """
    K1 = np.clip(K1, 0.0, 1.0)
    K2 = np.clip(K2, 0.0, 1.0)

    t1 = np.sqrt(1.0 - K1)
    t2 = np.sqrt(1.0 - K2)

    kappa1 = 1j * np.sqrt(K1)
    kappa2 = 1j * np.sqrt(K2)

    alpha_np_m = dbcm_to_npm(alpha_dbcm)
    L = ring_circumference(radius)

    # Round-trip field attenuation.
    a = np.exp(-0.5 * alpha_np_m * L)

    phi = 2.0 * np.pi * n_eff * L / wavelengths
    exp_phi = np.exp(-1j * phi)

    denominator = 1.0 - t1 * t2 * a * exp_phi

    through_field = (t1 - t2 * a * exp_phi) / denominator
    drop_field = 1j * np.sqrt(a) * kappa1 * kappa2 * np.exp(-0.5j * phi) / denominator

    through_power = np.abs(through_field) ** 2
    drop_power = np.abs(drop_field) ** 2

    return through_field, drop_field, through_power, drop_power


# ---------------------------------------------------------------------
# Fast SiC add-drop ring parameters
# ---------------------------------------------------------------------

n_eff = 2.60
alpha_dbcm = 1.0

radius = 25e-6
K1 = 0.04
K2 = 0.04

target_wavelength = 1550e-9

through_field, drop_field, through, drop = single_mrr_add_drop_fast(
    wavelengths=wl,
    radius=radius,
    n_eff=n_eff,
    alpha_dbcm=alpha_dbcm,
    K1=K1,
    K2=K2,
)

metrics = compute_peak_metrics_fast(
    wavelengths=wl,
    power=drop,
    target_wavelength=target_wavelength,
)

lam_p = metrics["resonance_wavelength"]
Q = metrics["loaded_Q"]
fwhm = metrics["fwhm"]
fsr = metrics["fsr"]
finesse = metrics["finesse"]
drop_er = metrics["extinction_ratio_db"]
maxima = metrics["resonance_indices"]

total = through + drop

print("\n=== SiC SFWM drop-resonance figure ===")
print(f"Tracked pump resonance: {lam_p * 1e9:.4f} nm")
print(f"Loaded Q: {Q:.2f}")
print(f"Detected resonances: {metrics['num_resonances_detected']}")
print(f"FWHM: {fwhm * 1e9:.6f} nm")
print(f"FSR: {fsr * 1e9:.6f} nm")
print(f"Finesse: {finesse:.3f}")
print(f"Drop ER: {drop_er:.3f} dB")
print(f"Max P_through + P_drop: {np.max(total):.8f}")
print(f"Passive check: {np.all(total <= 1 + 1e-8)}")

# ---------------------------------------------------------------------
# Plot 1: full drop-port sweep with detected resonance markers
# ---------------------------------------------------------------------

plt.figure(figsize=(8.5, 4.8))
plt.plot(wl * 1e9, drop, label="Drop power")
plt.plot(
    wl[maxima] * 1e9,
    drop[maxima],
    "o",
    markersize=4,
    label="Detected drop resonances",
)
plt.axvline(
    lam_p * 1e9,
    linestyle="--",
    label=f"Tracked pump resonance = {lam_p * 1e9:.3f} nm",
)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Drop power")
plt.title("SiC Add-Drop Ring: Drop-Port Resonances")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sic_sfwm_drop_resonance.png", dpi=250)

# ---------------------------------------------------------------------
# Plot 2: zoomed tracked drop resonance
# Optional but useful for checking narrow resonance shape.
# ---------------------------------------------------------------------

zoom_nm = 0.6
mask = np.abs(wl - lam_p) < zoom_nm * 1e-9

plt.figure(figsize=(7.2, 4.6))
plt.plot(wl[mask] * 1e9, drop[mask], label="Drop power")
plt.axvline(
    lam_p * 1e9,
    linestyle="--",
    label=f"Peak = {lam_p * 1e9:.4f} nm",
)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Drop power")
plt.title("Zoomed SiC Drop-Port Pump Resonance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sic_sfwm_drop_resonance_zoom.png", dpi=250)

# ---------------------------------------------------------------------
# Plot 3: add-drop passivity check
# Optional but useful for validating the fast model.
# ---------------------------------------------------------------------

plt.figure(figsize=(8.5, 4.8))
plt.plot(wl * 1e9, through, label="Through")
plt.plot(wl * 1e9, drop, label="Drop")
plt.plot(wl * 1e9, total, "--", label="Through + Drop")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power")
plt.title("Fast SiC Add-Drop Ring Response")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sic_add_drop_fast_response.png", dpi=250)

print("\nSaved:")
print("  sic_sfwm_drop_resonance.png")
print("  sic_sfwm_drop_resonance_zoom.png")
print("  sic_add_drop_fast_response.png")

plt.show()