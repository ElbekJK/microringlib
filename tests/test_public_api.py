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
    TransferMatrix,
    ScatteringMatrix,
    Cascade,
    waveguide_smatrix,
    directional_coupler_smatrix,
    mzi_smatrix,
    ring_allpass_smatrix,
    ring_add_drop_fields,
    cascade_two_ports,
    is_passive,
    is_unitary,
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
    assert TransferMatrix is not None
    assert ScatteringMatrix is not None
    assert Cascade is not None
    assert callable(waveguide_smatrix)
    assert callable(directional_coupler_smatrix)
    assert callable(mzi_smatrix)
    assert callable(ring_allpass_smatrix)
    assert callable(ring_add_drop_fields)
    assert callable(cascade_two_ports)
    assert callable(is_passive)
    assert callable(is_unitary)
