from .models import Layer, RingGeometry, Coupler, TransmissionResult, ModeResult

from .materials import (
    ConstantMaterial,
    TabulatedMaterial,
    FunctionMaterial,
    RefractiveIndexInfoMaterial,
    PyOptikMaterial,
)

from .metrics import (
    compute_resonance_metrics,
    compute_group_delay,
    find_resonances,
    fit_lorentzian,
    track_resonance_vs_parameter,
)

from .nonlinear import (
    KerrCavityParams,
    solve_kerr_energy,
    solve_kerr_sweep,
    kerr_effective_detuning,
    kerr_through_field,
    kerr_through_power,
    kerr_hysteresis,
    kerr_params_from_Q,
)

from .quantum import (
    SFWMParams,
    sfwm_pair_rate_relative,
    sfwm_pair_rate_from_params,
    wavelength_to_frequency,
    frequency_to_wavelength,
    energy_conserving_idler_wavelength,
    lorentzian_amplitude,
    lorentzian_power,
    sfwm_joint_spectral_amplitude_toy,
    schmidt_number_from_jsa,
    heralded_purity_from_jsa,
    coincidence_to_accidental_ratio,
    heralding_efficiency,
    brightness_summary,
)

# Fast accelerated helpers.
# These names must match functions/classes actually defined in microringlib/fast.py.
from .fast import (
    dbcm_to_npm_fast,
    ring_circumference_fast,
    ring_fsr_fast,
    interpolate_crossing_fast,
    detect_resonance_minima_fast,
    detect_resonance_maxima_fast,
    compute_resonance_metrics_fast,
    compute_peak_metrics_fast,
    single_mrr_thru_fast_batch,
    single_mrr_add_drop_fast,
    sfwm_pair_rate_relative_fast,
)

_TRANSFER_EXPORTS = {
    "ring_fsr",
    "group_index_from_modes",
    "ring_circumference",
    "single_waveguide",
    "straight_waveguide",
    "single_mrr_thru",
    "single_mrr_add_drop",
    "cascaded_mrrs_add_drop",
    "compute_transmission",
}

_MODE_EXPORTS = {
    "solve_waveguide_modes",
    "compute_group_index",
}

_PLOT_EXPORTS = {
    "plot_transmission",
    "plot_mode_profile",
}

_FAST_EXPORTS = {
    "dbcm_to_npm_fast",
    "ring_circumference_fast",
    "ring_fsr_fast",
    "interpolate_crossing_fast",
    "detect_resonance_minima_fast",
    "detect_resonance_maxima_fast",
    "compute_resonance_metrics_fast",
    "compute_peak_metrics_fast",
    "single_mrr_thru_fast_batch",
    "single_mrr_add_drop_fast",
    "sfwm_pair_rate_relative_fast",
}


def __getattr__(name):
    # Keep heavier imports lazy while preserving the public API.
    if name in _MODE_EXPORTS:
        from . import modes
        return getattr(modes, name)

    if name in _TRANSFER_EXPORTS:
        from . import transfer
        return getattr(transfer, name)

    if name in _PLOT_EXPORTS:
        from . import plotting
        return getattr(plotting, name)

    if name in _FAST_EXPORTS:
        from . import fast
        return getattr(fast, name)

    raise AttributeError(f"module 'microringlib' has no attribute {name!r}")


__all__ = [
    # Core models
    "Layer",
    "RingGeometry",
    "Coupler",
    "TransmissionResult",
    "ModeResult",

    # Materials
    "ConstantMaterial",
    "TabulatedMaterial",
    "FunctionMaterial",
    "RefractiveIndexInfoMaterial",
    "PyOptikMaterial",

    # Modes
    "solve_waveguide_modes",
    "compute_group_index",

    # Transfer / simulation
    "ring_circumference",
    "ring_fsr",
    "group_index_from_modes",
    "single_waveguide",
    "straight_waveguide",
    "single_mrr_thru",
    "single_mrr_add_drop",
    "cascaded_mrrs_add_drop",
    "compute_transmission",

    # Metrics
    "compute_resonance_metrics",
    "compute_group_delay",
    "find_resonances",
    "fit_lorentzian",
    "track_resonance_vs_parameter",

    # Nonlinear
    "KerrCavityParams",
    "solve_kerr_energy",
    "solve_kerr_sweep",
    "kerr_effective_detuning",
    "kerr_through_field",
    "kerr_through_power",
    "kerr_hysteresis",
    "kerr_params_from_Q",

    # Quantum
    "SFWMParams",
    "sfwm_pair_rate_relative",
    "sfwm_pair_rate_from_params",
    "wavelength_to_frequency",
    "frequency_to_wavelength",
    "energy_conserving_idler_wavelength",
    "lorentzian_amplitude",
    "lorentzian_power",
    "sfwm_joint_spectral_amplitude_toy",
    "schmidt_number_from_jsa",
    "heralded_purity_from_jsa",
    "coincidence_to_accidental_ratio",
    "heralding_efficiency",
    "brightness_summary",

    # Plotting
    "plot_transmission",
    "plot_mode_profile",

    # Fast accelerated API
    "dbcm_to_npm_fast",
    "ring_circumference_fast",
    "ring_fsr_fast",
    "interpolate_crossing_fast",
    "detect_resonance_minima_fast",
    "detect_resonance_maxima_fast",
    "compute_resonance_metrics_fast",
    "compute_peak_metrics_fast",
    "single_mrr_thru_fast_batch",
    "single_mrr_add_drop_fast",
    "sfwm_pair_rate_relative_fast",
]