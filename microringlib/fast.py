"""Accelerated analytical utilities for high-volume microring sweeps.

The core :mod:`microringlib.transfer` functions solve the layer stack and keep the
full physics-first API. This module provides vectorized analytical shortcuts for
large parameter sweeps, Monte Carlo studies, and demo figure generation.

These helpers intentionally use user-supplied ``n_eff`` / loss approximations and
are best viewed as fast reduced models, not replacements for the full
physics-first transfer API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


C0 = 299_792_458.0


def dbcm_to_npm(alpha_db_cm: float | np.ndarray) -> np.ndarray:
    """Convert power loss from dB/cm to Np/m."""
    return np.asarray(alpha_db_cm, dtype=float) * np.log(10.0) / 10.0 * 100.0


def dbcm_to_npm_fast(alpha_db_cm: float | np.ndarray) -> np.ndarray:
    """Backward-compatible fast alias for :func:`dbcm_to_npm`."""
    return dbcm_to_npm(alpha_db_cm)


def ring_circumference_fast(radius: float | np.ndarray) -> np.ndarray:
    """Circular-ring circumference for scalar or vector radii."""
    R = np.asarray(radius, dtype=float)
    if np.any(R <= 0):
        raise ValueError("radius must be positive")
    return 2.0 * np.pi * R


def ring_fsr_fast(
    wavelength: float | np.ndarray,
    n_g: float | np.ndarray,
    radius: float | np.ndarray,
) -> np.ndarray:
    """Approximate wavelength-domain FSR: lambda^2/(n_g L)."""
    wl = np.asarray(wavelength, dtype=float)
    ng = np.asarray(n_g, dtype=float)

    if np.any(wl <= 0):
        raise ValueError("wavelength must be positive")
    if np.any(ng <= 0):
        raise ValueError("group index must be positive")

    return wl**2 / (ng * ring_circumference_fast(radius))


def _validate_wavelengths(wavelengths) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=float)

    if wl.ndim != 1:
        raise ValueError("wavelengths must be 1D")
    if wl.size < 3:
        raise ValueError("need at least 3 wavelength points")
    if np.any(wl <= 0):
        raise ValueError("wavelengths must be positive")
    if not np.all(np.isfinite(wl)):
        raise ValueError("wavelengths must be finite")

    return wl


def _interp_crossing(x1, y1, x2, y2, y):
    if y2 == y1:
        return 0.5 * (x1 + x2)
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)


def interpolate_crossing_fast(x1, y1, x2, y2, y_target):
    """Public fast linear crossing interpolation helper."""
    return _interp_crossing(x1, y1, x2, y2, y_target)


def find_local_extrema(power, kind: Literal["dips", "peaks"] = "dips") -> np.ndarray:
    """Fast local extrema detector for clean simulated spectra."""
    p = np.asarray(power, dtype=float)

    if p.ndim != 1:
        raise ValueError("power must be 1D")
    if p.size < 3:
        return np.array([], dtype=int)

    if kind == "dips":
        return np.where((p[1:-1] < p[:-2]) & (p[1:-1] <= p[2:]))[0] + 1

    if kind == "peaks":
        return np.where((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1

    raise ValueError("kind must be 'dips' or 'peaks'")


def detect_resonance_minima_fast(power) -> np.ndarray:
    """Detect local minima in a clean simulated transmission spectrum."""
    return find_local_extrema(power, kind="dips")


def detect_resonance_maxima_fast(power) -> np.ndarray:
    """Detect local maxima in a clean simulated transmission spectrum."""
    return find_local_extrema(power, kind="peaks")


def resonance_metrics_fast(
    wavelengths,
    power,
    target_wavelength=None,
    kind: Literal["dips", "peaks"] = "dips",
) -> dict:
    """Fast FWHM/FSR/Q extraction for clean simulated notch or peak spectra.

    This is intentionally lighter than :func:`compute_resonance_metrics` and is
    useful inside Monte Carlo loops. For noisy experimental data, prefer the full
    metrics module or fit routines.
    """
    wl = _validate_wavelengths(wavelengths)
    p = np.asarray(power, dtype=float)

    if p.ndim != 1:
        raise ValueError("power must be 1D")
    if p.shape != wl.shape:
        raise ValueError("power must match wavelength shape")
    if not np.all(np.isfinite(p)):
        raise ValueError("power must be finite")

    extrema = find_local_extrema(p, kind=kind)

    if extrema.size == 0:
        return {
            "resonance_wavelength": np.nan,
            "resonance_index": -1,
            "fwhm": np.nan,
            "fsr": np.nan,
            "loaded_Q": np.nan,
            "quality_factor": np.nan,
            "finesse": np.nan,
            "extinction_ratio_db": np.nan,
            "num_resonances_detected": 0,
            "resonance_indices": extrema,
        }

    if target_wavelength is None:
        idx = extrema[np.argmin(p[extrema])] if kind == "dips" else extrema[np.argmax(p[extrema])]
    else:
        idx = extrema[np.argmin(np.abs(wl[extrema] - float(target_wavelength)))]

    pos = int(np.where(extrema == idx)[0][0])
    lam0 = float(wl[idx])

    fsrs = []
    if pos > 0:
        fsrs.append(lam0 - float(wl[extrema[pos - 1]]))
    if pos < extrema.size - 1:
        fsrs.append(float(wl[extrema[pos + 1]]) - lam0)
    fsr = float(np.mean(fsrs)) if fsrs else np.nan

    left_bound = int(extrema[pos - 1]) if pos > 0 else 0
    right_bound = int(extrema[pos + 1]) if pos < extrema.size - 1 else p.size - 1
    local = p[left_bound:right_bound + 1]

    if kind == "dips":
        extremum = float(p[idx])
        baseline = float(np.max(local))
        half = extremum + 0.5 * (baseline - extremum)
        er = 10.0 * np.log10(baseline / extremum) if extremum > 0 and baseline > 0 else np.nan

        left_cross = np.nan
        for j in range(idx - 1, left_bound - 1, -1):
            if p[j] >= half and p[j + 1] < half:
                left_cross = _interp_crossing(wl[j], p[j], wl[j + 1], p[j + 1], half)
                break

        right_cross = np.nan
        for j in range(idx, right_bound):
            if p[j] < half and p[j + 1] >= half:
                right_cross = _interp_crossing(wl[j], p[j], wl[j + 1], p[j + 1], half)
                break

    else:
        extremum = float(p[idx])
        baseline = float(np.min(local))
        half = baseline + 0.5 * (extremum - baseline)
        er = 10.0 * np.log10(extremum / baseline) if extremum > 0 and baseline > 0 else np.nan

        left_cross = np.nan
        for j in range(idx - 1, left_bound - 1, -1):
            if p[j] <= half and p[j + 1] > half:
                left_cross = _interp_crossing(wl[j], p[j], wl[j + 1], p[j + 1], half)
                break

        right_cross = np.nan
        for j in range(idx, right_bound):
            if p[j] > half and p[j + 1] <= half:
                right_cross = _interp_crossing(wl[j], p[j], wl[j + 1], p[j + 1], half)
                break

    if np.isfinite(left_cross) and np.isfinite(right_cross) and right_cross > left_cross:
        fwhm = float(right_cross - left_cross)
        q_loaded = float(lam0 / fwhm)
    else:
        fwhm = np.nan
        q_loaded = np.nan

    finesse = float(fsr / fwhm) if np.isfinite(fsr) and np.isfinite(fwhm) and fwhm > 0 else np.nan

    return {
        "resonance_wavelength": lam0,
        "resonance_index": int(idx),
        "fwhm": fwhm,
        "fsr": fsr,
        "loaded_Q": q_loaded,
        "quality_factor": q_loaded,
        "finesse": finesse,
        "extinction_ratio_db": float(er) if np.isfinite(er) else np.nan,
        "num_resonances_detected": int(extrema.size),
        "resonance_indices": extrema,
    }


def compute_resonance_metrics_fast(wavelengths, power, target_wavelength=None) -> dict:
    """Fast metrics for dip/notch resonances."""
    return resonance_metrics_fast(
        wavelengths=wavelengths,
        power=power,
        target_wavelength=target_wavelength,
        kind="dips",
    )


def compute_peak_metrics_fast(wavelengths, power, target_wavelength=None) -> dict:
    """Fast metrics for peak/drop-port resonances."""
    return resonance_metrics_fast(
        wavelengths=wavelengths,
        power=power,
        target_wavelength=target_wavelength,
        kind="peaks",
    )


def single_mrr_thru_fast(
    wavelengths,
    radius: float,
    n_eff: float,
    alpha_dbcm: float,
    K: float,
):
    """Fast all-pass through-port field/power for one coupling value."""
    fields, powers, t, kappa = single_mrr_thru_fast_batch(
        wavelengths,
        radius,
        n_eff,
        alpha_dbcm,
        [K],
    )
    return fields[0], powers[0], t[0], kappa[0]


def single_mrr_thru_fast_batch(
    wavelengths,
    radius: float,
    n_eff: float,
    alpha_dbcm: float,
    K_values,
):
    """Vectorized all-pass ring through-port model for many coupling values.

    Returns ``fields, powers, t_values, kappa_values`` with shape ``(nK, nW)``
    for the spectral arrays.
    """
    wl = _validate_wavelengths(wavelengths)
    K = np.asarray(K_values, dtype=float)

    if K.ndim == 0:
        K = K[None]
    if K.ndim != 1:
        raise ValueError("K_values must be scalar or 1D")
    if np.any((K < 0) | (K > 1)):
        raise ValueError("K_values must be in [0, 1]")

    if n_eff <= 0:
        raise ValueError("n_eff must be positive")

    L = float(ring_circumference_fast(radius))
    alpha = float(dbcm_to_npm(alpha_dbcm))
    a = np.exp(-0.5 * alpha * L)

    phi = 2.0 * np.pi * float(n_eff) * L / wl
    e = np.exp(-1j * phi)[None, :]

    t = np.sqrt(1.0 - K)
    kappa = 1j * np.sqrt(K)

    tv = t[:, None]
    field = (tv - a * e) / (1.0 - tv * a * e)
    power = np.abs(field) ** 2

    return field, power, t, kappa


def single_mrr_add_drop_fast(
    wavelengths,
    radius: float,
    n_eff: float,
    alpha_dbcm: float,
    K1: float,
    K2: float,
):
    """Fast analytical add-drop microring model.

    Returns ``through_field, drop_field, through_power, drop_power``. The model
    uses a standard reduced add-drop denominator and is useful for fast demos
    and sweeps when ``n_eff`` is known.
    """
    wl = _validate_wavelengths(wavelengths)

    if n_eff <= 0:
        raise ValueError("n_eff must be positive")
    if not (0 <= K1 <= 1 and 0 <= K2 <= 1):
        raise ValueError("K1 and K2 must be in [0, 1]")

    t1 = np.sqrt(1.0 - float(K1))
    t2 = np.sqrt(1.0 - float(K2))
    k1 = 1j * np.sqrt(float(K1))
    k2 = 1j * np.sqrt(float(K2))

    L = float(ring_circumference_fast(radius))
    alpha = float(dbcm_to_npm(alpha_dbcm))
    a = np.exp(-0.5 * alpha * L)

    phi = 2.0 * np.pi * float(n_eff) * L / wl
    exp_phi = np.exp(-1j * phi)

    den = 1.0 - a * t1 * t2 * exp_phi

    through_field = (t1 - a * t2 * exp_phi) / den
    drop_field = 1j * np.sqrt(a) * k1 * k2 * np.exp(-0.5j * phi) / den

    through_power = np.abs(through_field) ** 2
    drop_power = np.abs(drop_field) ** 2

    return through_field, drop_field, through_power, drop_power


def sfwm_pair_rate_relative_fast(
    pump_power,
    gamma: float,
    loaded_Q: float,
    ring_radius: float,
    normalize: bool = True,
):
    """Fast reduced SFWM scaling: R ∝ gamma^2 P^2 Q^3/R^2."""
    P = np.asarray(pump_power, dtype=float)

    if np.any(P < 0):
        raise ValueError("pump_power must be nonnegative")
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    if loaded_Q <= 0:
        raise ValueError("loaded_Q must be positive")
    if ring_radius <= 0:
        raise ValueError("ring_radius must be positive")

    rate = (float(gamma) ** 2) * P**2 * (float(loaded_Q) ** 3) / (float(ring_radius) ** 2)

    if normalize and np.max(rate) > 0:
        rate = rate / np.max(rate)

    return rate


@dataclass(frozen=True)
class MonteCarloResult:
    resonances_nm: np.ndarray
    loaded_Q: np.ndarray
    extinction_ratio_db: np.ndarray
    radius_samples_m: np.ndarray
    n_eff_samples: np.ndarray

    def summary(self) -> dict:
        return {
            "trials": int(self.resonances_nm.size),
            "resonance_mean_nm": float(np.nanmean(self.resonances_nm)),
            "resonance_std_nm": float(np.nanstd(self.resonances_nm)),
            "Q_mean": float(np.nanmean(self.loaded_Q)),
            "Q_std": float(np.nanstd(self.loaded_Q)),
            "ER_mean_db": float(np.nanmean(self.extinction_ratio_db)),
            "ER_std_db": float(np.nanstd(self.extinction_ratio_db)),
        }


def monte_carlo_ring_tolerance_fast(
    wavelengths,
    n_trials: int,
    radius_mean: float,
    radius_sigma: float,
    n_eff_mean: float,
    n_eff_sigma: float,
    alpha_dbcm: float,
    K: float,
    target_wavelength: float,
    seed: int | None = None,
) -> MonteCarloResult:
    """Accelerated Monte Carlo tolerance analysis for a single-bus ring."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if radius_sigma < 0:
        raise ValueError("radius_sigma must be nonnegative")
    if n_eff_sigma < 0:
        raise ValueError("n_eff_sigma must be nonnegative")

    rng = np.random.default_rng(seed)
    wl = _validate_wavelengths(wavelengths)

    radius_samples = radius_mean + rng.normal(size=n_trials) * radius_sigma
    n_eff_samples = n_eff_mean + rng.normal(size=n_trials) * n_eff_sigma

    resonances = np.empty(n_trials)
    q_values = np.empty(n_trials)
    er_values = np.empty(n_trials)

    for i in range(n_trials):
        _, power, _, _ = single_mrr_thru_fast(
            wl,
            radius_samples[i],
            n_eff_samples[i],
            alpha_dbcm,
            K,
        )
        metrics = resonance_metrics_fast(
            wl,
            power,
            target_wavelength=target_wavelength,
            kind="dips",
        )
        resonances[i] = metrics["resonance_wavelength"] * 1e9
        q_values[i] = metrics["loaded_Q"]
        er_values[i] = metrics["extinction_ratio_db"]

    return MonteCarloResult(
        resonances_nm=resonances,
        loaded_Q=q_values,
        extinction_ratio_db=er_values,
        radius_samples_m=radius_samples,
        n_eff_samples=n_eff_samples,
    )


def monte_carlo_resonance_formula_fast(
    n_eff_samples,
    radius_samples,
    target_wavelength: float,
    n_g: float | None = None,
    loaded_Q_nominal: float | None = None,
    extinction_ratio_db_nominal: float | None = None,
) -> dict:
    """Ultra-fast resonance-order Monte Carlo estimate.

    Tracks the integer resonance order nearest ``target_wavelength`` using
    ``m = round(n_eff * 2*pi*R / target_wavelength)`` and returns the
    corresponding wavelength ``lambda_m = n_eff * 2*pi*R / m``.

    This is much faster than sweeping spectra and is appropriate for first-pass
    yield estimates.
    """
    n = np.asarray(n_eff_samples, dtype=float)
    radius = np.asarray(radius_samples, dtype=float)

    if n.shape != radius.shape:
        raise ValueError("n_eff_samples and radius_samples must have the same shape")
    if np.any(n <= 0):
        raise ValueError("n_eff_samples must be positive")
    if np.any(radius <= 0):
        raise ValueError("radius_samples must be positive")
    if target_wavelength <= 0:
        raise ValueError("target_wavelength must be positive")

    circumference = 2.0 * np.pi * radius
    mode_number = np.maximum(1, np.rint(n * circumference / target_wavelength)).astype(int)
    resonance_wavelength = n * circumference / mode_number

    out = {
        "resonance_wavelength_m": resonance_wavelength,
        "resonance_wavelength_nm": resonance_wavelength * 1e9,
        "mode_number": mode_number,
    }

    if n_g is not None:
        if n_g <= 0:
            raise ValueError("n_g must be positive")
        fsr_m = resonance_wavelength**2 / (float(n_g) * circumference)
        out["fsr_m"] = fsr_m
        out["fsr_nm"] = fsr_m * 1e9

    if loaded_Q_nominal is not None:
        out["loaded_Q"] = np.full_like(resonance_wavelength, float(loaded_Q_nominal), dtype=float)

    if extinction_ratio_db_nominal is not None:
        out["extinction_ratio_db"] = np.full_like(
            resonance_wavelength,
            float(extinction_ratio_db_nominal),
            dtype=float,
        )

    return out