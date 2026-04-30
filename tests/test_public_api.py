from microringlib import (
    Layer,
    RingGeometry,
    Coupler,
    TransmissionResult,
    ModeResult,
    solve_waveguide_modes,
    compute_group_index,
    compute_resonance_metrics,
    ring_circumference,
    single_waveguide,
    single_mrr_thru,
    single_mrr_add_drop,
    cascaded_mrrs_add_drop,
    compute_transmission,
    plot_transmission,
    plot_mode_profile,
)


def test_public_imports_exist():
    assert Layer is not None
    assert RingGeometry is not None
    assert Coupler is not None
    assert TransmissionResult is not None
    assert ModeResult is not None
    assert callable(solve_waveguide_modes)
    assert callable(compute_group_index)
    assert callable(compute_resonance_metrics)
    assert callable(ring_circumference)
    assert callable(single_waveguide)
    assert callable(single_mrr_thru)
    assert callable(single_mrr_add_drop)
    assert callable(cascaded_mrrs_add_drop)
    assert callable(compute_transmission)
    assert callable(plot_transmission)
    assert callable(plot_mode_profile)