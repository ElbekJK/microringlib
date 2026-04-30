from __future__ import annotations

import warnings
import numpy as np


C0 = 299_792_458.0


def _as_1d_pair(x, y, x_name="wavelength", y_name="transmission"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"{x_name} and {y_name} must be 1D arrays")
    if x.shape != y.shape:
        raise ValueError(f"{x_name} and {y_name} must have the same shape")
    if x.size < 3:
        raise ValueError("need at least 3 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{x_name} contains non-finite values")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{y_name} contains non-finite values")

    order = np.argsort(x)
    return x[order], y[order]


def _interp_crossing(x0, y0, x1, y1, level):
    if y1 == y0:
        return float(x0)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def find_resonances(
    wavelength,
    transmission,
    kind="dips",
    min_prominence=None,
):
    wl, T = _as_1d_pair(wavelength, transmission)

    if kind not in {"dips", "peaks"}:
        raise ValueError("kind must be 'dips' or 'peaks'")

    if kind == "dips":
        candidates = np.where((T[1:-1] <= T[:-2]) & (T[1:-1] <= T[2:]))[0] + 1
    else:
        candidates = np.where((T[1:-1] >= T[:-2]) & (T[1:-1] >= T[2:]))[0] + 1

    if min_prominence is None:
        return candidates

    kept = []
    for i in candidates:
        local_ref = 0.5 * (T[i - 1] + T[i + 1])
        prominence = abs(local_ref - T[i])
        if prominence >= min_prominence:
            kept.append(i)

    return np.asarray(kept, dtype=int)


def fit_lorentzian(
    wavelength,
    transmission,
    resonance_index=None,
    resonance_kind="dips",
):
    wl, T = _as_1d_pair(wavelength, transmission)

    if wl.size < 5:
        return {
            "center_wavelength": np.nan,
            "fwhm": np.nan,
            "loaded_Q": np.nan,
            "depth": np.nan,
            "half_depth_level": np.nan,
            "left_half_depth_wavelength": np.nan,
            "right_half_depth_wavelength": np.nan,
        }

    if resonance_kind not in {"dips", "peaks"}:
        raise ValueError("resonance_kind must be 'dips' or 'peaks'")

    if resonance_index is None:
        idx = int(np.argmin(T)) if resonance_kind == "dips" else int(np.argmax(T))
    else:
        idx = int(resonance_index)

    if idx < 0 or idx >= wl.size:
        raise ValueError("resonance_index is out of bounds")

    if resonance_kind == "dips":
        baseline = float(np.max(T))
        extremum = float(T[idx])
        half = 0.5 * (baseline + extremum)
    else:
        baseline = float(np.min(T))
        extremum = float(T[idx])
        half = 0.5 * (baseline + extremum)

    left = np.nan
    for j in range(idx - 1, -1, -1):
        if (T[j] - half) * (T[j + 1] - half) <= 0:
            left = _interp_crossing(wl[j], T[j], wl[j + 1], T[j + 1], half)
            break

    right = np.nan
    for j in range(idx, wl.size - 1):
        if (T[j] - half) * (T[j + 1] - half) <= 0:
            right = _interp_crossing(wl[j], T[j], wl[j + 1], T[j + 1], half)
            break

    if np.isfinite(left) and np.isfinite(right) and right > left:
        fwhm = float(right - left)
    else:
        fwhm = np.nan

    lam0 = float(wl[idx])
    loaded_Q = float(lam0 / fwhm) if np.isfinite(fwhm) and fwhm > 0 else np.nan

    return {
        "center_wavelength": lam0,
        "fwhm": fwhm,
        "loaded_Q": loaded_Q,
        "depth": abs(float(extremum - baseline)),
        "half_depth_level": float(half),
        "left_half_depth_wavelength": left,
        "right_half_depth_wavelength": right,
    }

def track_resonance_vs_parameter(
    wavelengths,
    spectra,
    parameter_values,
    initial_target_wavelength,
    resonance_kind="dips",
    min_prominence=None,
):
    tracked = []
    target = initial_target_wavelength

    for value, spectrum in zip(parameter_values, spectra):
        metrics = compute_resonance_metrics(
            wavelengths,
            spectrum,
            resonance_kind=resonance_kind,
            target_wavelength=target,
            min_prominence=min_prominence,
            warn=False,
        )

        tracked.append({
            "parameter": value,
            "resonance_wavelength": metrics["resonance_wavelength"],
            "fwhm": metrics["fwhm"],
            "loaded_Q": metrics["loaded_Q"],
            "fsr": metrics["fsr"],
            "finesse": metrics["finesse"],
            "extinction_ratio_db": metrics["extinction_ratio_db"],
            "num_resonances_detected": metrics["num_resonances_detected"],
            "resonance_index": metrics["resonance_index"],
        })

        target = metrics["resonance_wavelength"]

    return tracked

def compute_group_delay(wavelength, field):
    wl = np.asarray(wavelength, dtype=float)
    H = np.asarray(field, dtype=np.complex128)

    if wl.ndim != 1 or H.ndim != 1:
        raise ValueError("wavelength and field must be 1D arrays")
    if wl.shape != H.shape:
        raise ValueError("wavelength and field must have the same shape")
    if wl.size < 3:
        raise ValueError("need at least 3 samples")

    omega = 2.0 * np.pi * C0 / wl
    order = np.argsort(omega)

    phase = np.unwrap(np.angle(H[order]))
    tau_sorted = -np.gradient(phase, omega[order])

    tau = np.empty_like(tau_sorted)
    tau[order] = tau_sorted
    return tau


def compute_resonance_metrics(
    wavelength,
    transmission,
    resonance_kind="dips",
    warn=True,
    target_wavelength=None,
    min_prominence=None,
):
    """
    Compute physically useful resonance metrics.

    Parameters
    ----------
    wavelength:
        Wavelength array in meters.

    transmission:
        Power transmission array.

    resonance_kind:
        "dips" for through-port notches.
        "peaks" for drop-port peaks.

    target_wavelength:
        Optional wavelength in meters. If provided, the function tracks the
        resonance closest to this target instead of blindly choosing the
        deepest dip or highest peak. This is important for thermal tuning,
        where different resonance orders can otherwise be selected.

    min_prominence:
        Optional minimum local contrast for detecting resonances.
    """
    wl, T = _as_1d_pair(wavelength, transmission)

    if resonance_kind not in {"dips", "peaks"}:
        raise ValueError("resonance_kind must be 'dips' or 'peaks'")

    if np.any(T < -1e-12):
        raise ValueError("power transmission contains negative values")

    T = np.maximum(T, 0.0)

    idx_min = int(np.argmin(T))
    idx_max = int(np.argmax(T))
    Tmin = float(T[idx_min])
    Tmax = float(T[idx_max])

    resonance_indices = find_resonances(
        wl,
        T,
        kind=resonance_kind,
        min_prominence=min_prominence,
    )

    if len(resonance_indices) > 0 and target_wavelength is not None:
        target_wavelength = float(target_wavelength)
        resonance_idx = int(
            resonance_indices[
                np.argmin(np.abs(wl[resonance_indices] - target_wavelength))
            ]
        )
    elif len(resonance_indices) > 0:
        if resonance_kind == "dips":
            resonance_idx = int(resonance_indices[np.argmin(T[resonance_indices])])
        else:
            resonance_idx = int(resonance_indices[np.argmax(T[resonance_indices])])
    else:
        if warn:
            warnings.warn(
                "No local resonances detected. Falling back to global minimum/maximum.",
                RuntimeWarning,
                stacklevel=2,
            )
        resonance_idx = idx_min if resonance_kind == "dips" else idx_max

    lam0 = float(wl[resonance_idx])

    fit = fit_lorentzian(
        wl,
        T,
        resonance_index=resonance_idx,
        resonance_kind=resonance_kind,
    )

    fwhm = float(fit["fwhm"])
    loaded_Q = float(lam0 / fwhm) if np.isfinite(fwhm) and fwhm > 0 else np.nan

    fsr = np.nan
    if len(resonance_indices) >= 2:
        fsr = float(np.median(np.diff(np.sort(wl[resonance_indices]))))
    elif warn:
        warnings.warn(
            "FSR/finesse are NaN because fewer than two resonances were detected. "
            "Use a wider wavelength sweep.",
            RuntimeWarning,
            stacklevel=2,
        )

    finesse = (
        float(fsr / fwhm)
        if np.isfinite(fsr) and np.isfinite(fwhm) and fwhm > 0
        else np.nan
    )

    extinction_ratio_db = (
        float(10.0 * np.log10(Tmax / Tmin))
        if Tmin > 0 and Tmax > 0
        else np.nan
    )

    notch_depth_factor = float(Tmax / Tmin) if Tmin > 0 else np.nan

    return {
        "resonance_wavelength": lam0,
        "resonance_index": int(resonance_idx),
        "min_transmission": Tmin,
        "max_transmission": Tmax,
        "fwhm": fwhm,
        "fsr": fsr,
        "quality_factor": loaded_Q,
        "loaded_Q": loaded_Q,
        "finesse": finesse,
        "extinction_ratio_db": extinction_ratio_db,
        "notch_depth_factor": notch_depth_factor,

        # Backward-compatible alias.
        "intensity_enhancement": notch_depth_factor,
        "intensity_enhancement_note": (
            "Deprecated alias for notch_depth_factor = max_transmission / "
            "min_transmission. This is not intracavity intensity enhancement."
        ),

        "num_resonances_detected": int(len(resonance_indices)),
        "resonance_indices": resonance_indices,
        "target_wavelength": target_wavelength,
        "target_tracking_used": target_wavelength is not None,
    }