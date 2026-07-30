"""Tests for DEM error decomposition."""

from __future__ import annotations

import pytest
import stim

from qec_loss.decompose_errors import (
    canonicalize_dem_components,
    decompose_errors,
    is_graphlike,
)


@pytest.mark.parametrize(
    "code_task",
    [
        "surface_code:rotated_memory_z",
        "color_code:memory_xyz",
    ],
)
def test_decompose_errors_matches_stim_on_generated_codes(code_task: str) -> None:
    """DEM-level decompose matches Stim's circuit-level ``decompose_errors=True``.

    Component order within a ``^``-split is canonicalized on both sides: Stim's
    local pass uses X-then-Z channel order, which is not recoverable from a DEM.
    """
    circuit = stim.Circuit.generated(
        code_task,
        distance=3,
        rounds=2,
        before_round_data_depolarization=0.01,
    )
    expected = canonicalize_dem_components(circuit.detector_error_model(decompose_errors=True))
    got = decompose_errors(circuit.detector_error_model(decompose_errors=False))
    assert expected.approx_equals(got, atol=1e-5)


@pytest.mark.parametrize(
    "code_task",
    [
        "surface_code:rotated_memory_z",
        "surface_code:rotated_memory_x",
        "color_code:memory_xyz",
    ],
)
def test_decompose_errors_is_graphlike_with_clifford_noise(code_task: str) -> None:
    circuit = stim.Circuit.generated(
        code_task,
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    dem = decompose_errors(circuit.detector_error_model(decompose_errors=False))
    for inst in dem.flattened():
        if inst.type == "error":
            assert is_graphlike(list(inst.targets_copy()))


def test_decompose_errors_handcrafted_hyperedge() -> None:
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0
        error(0.1) D1
        error(0.1) D0 D1 D2
        detector D0
        detector D1
        detector D2
        """
    )
    got = decompose_errors(dem)
    assert is_graphlike(list(next(i for i in got.flattened() if i.type == "error" and "D2" in str(i)).targets_copy()))
    # The hyperedge becomes D0 ^ D1 ^ D2 (or an equivalent graphlike split).
    hyper = [i for i in got.flattened() if i.type == "error" and any(t.is_separator() for t in i.targets_copy())]
    assert len(hyper) == 1
    assert abs(hyper[0].args_copy()[0] - 0.1) < 1e-12


def test_decompose_errors_identity_on_already_graphlike() -> None:
    dem = stim.DetectorErrorModel(
        """
        error(0.01) D0
        error(0.02) D0 D1
        error(0.03) D1 L0
        detector D0
        detector D1
        """
    )
    got = decompose_errors(dem)
    assert dem.approx_equals(got, atol=1e-15)


def test_decompose_errors_preserves_detector_metadata() -> None:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=2,
        before_round_data_depolarization=0.01,
    )
    dem0 = circuit.detector_error_model(decompose_errors=False)
    dem1 = circuit.detector_error_model(decompose_errors=True)
    got = decompose_errors(dem0)
    meta0 = [str(i) for i in dem0 if i.type != "error"]
    meta_g = [str(i) for i in got if i.type != "error"]
    meta1 = [str(i) for i in dem1 if i.type != "error"]
    assert meta_g == meta0 == meta1
