from __future__ import annotations

from dataclasses import dataclass
import numpy as np


C0 = 299_792_458.0


@dataclass
class SFWMParams:
    gamma: float
    pump_power: float
    loaded_Q: float
    ring_radius: float
    pump_wavelength: float
    signal_wavelength: float | None = None
    idler_wavelength: float | None = None
    coupling_efficiency: float = 1.0

    @property
    def circumference(self) -> float:
        return 2.0 * np.pi * self.ring_radius


def sfwm_pair_rate_relative(
    pump_power,
    gamma: float,
    loaded_Q: float,
    ring_radius: float,
    normalize: bool = True,
):
    """
    Reduced relative SFWM pair-generation scaling.

    Uses a qualitative scaling:

        R_pair ∝ gamma^2 * P_pump^2 * Q^3 / R^2

    This is not an absolute calibrated quantum source model.
    """
    P = np.asarray(pump_power, dtype=float)

    if np.any(P < 0):
        raise ValueError("pump_power must be nonnegative")
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    if loaded_Q <= 0:
        raise ValueError("loaded_Q must be positive")
    if ring_radius <= 0:
        raise ValueError("ring_radius must be positive")

    rate = (gamma**2) * (P**2) * (loaded_Q**3) / (ring_radius**2)

    if normalize:
        max_rate = np.max(rate)
        if max_rate > 0:
            rate = rate / max_rate

    return rate


def sfwm_pair_rate_from_params(params: SFWMParams, pump_power=None, normalize=True):
    P = params.pump_power if pump_power is None else pump_power

    rate = sfwm_pair_rate_relative(
        pump_power=P,
        gamma=params.gamma,
        loaded_Q=params.loaded_Q,
        ring_radius=params.ring_radius,
        normalize=normalize,
    )

    return params.coupling_efficiency * rate


def wavelength_to_frequency(wavelength):
    wl = np.asarray(wavelength, dtype=float)
    if np.any(wl <= 0):
        raise ValueError("wavelength must be positive")
    return C0 / wl


def frequency_to_wavelength(frequency):
    f = np.asarray(frequency, dtype=float)
    if np.any(f <= 0):
        raise ValueError("frequency must be positive")
    return C0 / f


def energy_conserving_idler_wavelength(pump_wavelength, signal_wavelength):
    """
    For degenerate SFWM:

        2 f_p = f_s + f_i

    Returns idler wavelength.
    """
    fp = wavelength_to_frequency(pump_wavelength)
    fs = wavelength_to_frequency(signal_wavelength)
    fi = 2.0 * fp - fs

    if np.any(fi <= 0):
        raise ValueError("invalid signal wavelength gives nonphysical idler frequency")

    return frequency_to_wavelength(fi)


def lorentzian_amplitude(wavelength, center_wavelength, loaded_Q):
    """
    Normalized complex Lorentzian cavity amplitude response.
    """
    wl = np.asarray(wavelength, dtype=float)

    if loaded_Q <= 0:
        raise ValueError("loaded_Q must be positive")

    f = wavelength_to_frequency(wl)
    f0 = wavelength_to_frequency(center_wavelength)
    linewidth = f0 / loaded_Q

    x = 2.0 * (f - f0) / linewidth
    return 1.0 / (1.0 + 1j * x)


def lorentzian_power(wavelength, center_wavelength, loaded_Q):
    H = lorentzian_amplitude(wavelength, center_wavelength, loaded_Q)
    return np.abs(H) ** 2


def sfwm_joint_spectral_amplitude_toy(
    signal_wavelengths,
    idler_wavelengths,
    pump_wavelength: float,
    pump_bandwidth_nm: float,
    signal_center: float,
    idler_center: float,
    signal_Q: float,
    idler_Q: float,
):
    """
    Toy joint spectral amplitude for SFWM.

    JSA ≈ pump_envelope(νs + νi - 2νp)
          × cavity_s(signal)
          × cavity_i(idler)

    Returns complex JSA array with shape:
        (len(signal_wavelengths), len(idler_wavelengths))
    """
    ls = np.asarray(signal_wavelengths, dtype=float)
    li = np.asarray(idler_wavelengths, dtype=float)

    fs = wavelength_to_frequency(ls)[:, None]
    fi = wavelength_to_frequency(li)[None, :]
    fp = wavelength_to_frequency(pump_wavelength)

    pump_bw_hz = C0 * pump_bandwidth_nm * 1e-9 / pump_wavelength**2
    if pump_bw_hz <= 0:
        raise ValueError("pump_bandwidth_nm must be positive")

    energy_mismatch = fs + fi - 2.0 * fp
    pump_env = np.exp(-0.5 * (energy_mismatch / pump_bw_hz) ** 2)

    Hs = lorentzian_amplitude(ls, signal_center, signal_Q)[:, None]
    Hi = lorentzian_amplitude(li, idler_center, idler_Q)[None, :]

    JSA = pump_env * Hs * Hi
    norm = np.sqrt(np.sum(np.abs(JSA) ** 2))
    if norm > 0:
        JSA = JSA / norm

    return JSA


def schmidt_number_from_jsa(jsa):
    """
    Estimate Schmidt number from a discretized JSA.

    K = 1 / sum(lambda_i^4), where lambda_i are normalized singular values.
    """
    A = np.asarray(jsa, dtype=np.complex128)

    if A.ndim != 2:
        raise ValueError("jsa must be a 2D array")

    _, s, _ = np.linalg.svd(A, full_matrices=False)

    norm = np.sqrt(np.sum(s**2))
    if norm == 0:
        return np.nan

    lambdas = s / norm
    return float(1.0 / np.sum(lambdas**4))


def heralded_purity_from_jsa(jsa):
    K = schmidt_number_from_jsa(jsa)
    if not np.isfinite(K) or K <= 0:
        return np.nan
    return float(1.0 / K)


def coincidence_to_accidental_ratio(pair_rate, coincidence_window, dark_count_rate=0.0):
    """
    Simple CAR estimate.

    accidental_rate ≈ pair_rate^2 * coincidence_window + dark_count_rate
    CAR = pair_rate / accidental_rate
    """
    R = np.asarray(pair_rate, dtype=float)

    if np.any(R < 0):
        raise ValueError("pair_rate must be nonnegative")
    if coincidence_window <= 0:
        raise ValueError("coincidence_window must be positive")
    if dark_count_rate < 0:
        raise ValueError("dark_count_rate must be nonnegative")

    accidental = R**2 * coincidence_window + dark_count_rate
    return np.where(accidental > 0, R / accidental, np.inf)


def heralding_efficiency(
    signal_collection_efficiency,
    idler_collection_efficiency,
    detector_efficiency_signal=1.0,
    detector_efficiency_idler=1.0,
):
    eta_s = float(signal_collection_efficiency) * float(detector_efficiency_signal)
    eta_i = float(idler_collection_efficiency) * float(detector_efficiency_idler)

    if not (0 <= eta_s <= 1):
        raise ValueError("signal efficiency must be in [0, 1]")
    if not (0 <= eta_i <= 1):
        raise ValueError("idler efficiency must be in [0, 1]")

    return {
        "signal_arm_efficiency": eta_s,
        "idler_arm_efficiency": eta_i,
        "symmetric_heralding_efficiency": float(np.sqrt(eta_s * eta_i)),
        "pair_detection_efficiency": float(eta_s * eta_i),
    }


def brightness_summary(
    pump_power,
    relative_pair_rate,
    repetition_rate=None,
):
    P = np.asarray(pump_power, dtype=float)
    R = np.asarray(relative_pair_rate, dtype=float)

    if P.shape != R.shape:
        raise ValueError("pump_power and relative_pair_rate must have the same shape")

    summary = {
        "max_relative_pair_rate": float(np.max(R)),
        "pump_at_max_rate": float(P[int(np.argmax(R))]),
        "rate_at_highest_pump": float(R[-1]),
    }

    if repetition_rate is not None:
        if repetition_rate <= 0:
            raise ValueError("repetition_rate must be positive")
        summary["relative_pairs_per_pulse"] = R / repetition_rate

    return summary