import stim
import pymatching
from qec_loss.frame.physical_frame_decoder import PhysicalFrameDecoder


def _surface_code_circuit(distance=5, rounds=5):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.01,
        after_reset_flip_probability=0.01,
        before_measure_flip_probability=0.01,
    )


def _assert_ground_truth(decoder: PhysicalFrameDecoder):
    mismatches = 0
    contribs = decoder.final_frame_contributions
    for i in range(decoder.num_errors):
        contrib = contribs[i]
        for obs_idx, group in enumerate(decoder.obs_groups):
            expected = int(decoder.observables_matrix[obs_idx, i])
            actual = 0
            for m in group:
                if decoder._final_meas_pos[m] in contrib:
                    actual ^= 1
            if actual != expected:
                mismatches += 1
    assert mismatches == 0, (
        f"final_frame_contributions vs DEM observable membership: "
        f"{mismatches}/{decoder.num_errors} mismatches"
    )


def _assert_decode_correctness(decoder: PhysicalFrameDecoder, circuit: stim.Circuit, shots=100):
    dem_ref = circuit.detector_error_model(decompose_errors=True)
    matching_ref = pymatching.Matching.from_detector_error_model(dem_ref)
    dets, _ = circuit.compile_detector_sampler().sample(
        shots=shots, separate_observables=True
    )

    n_self_consistent = 0
    n_matches_reference = 0
    for s in range(shots):
        fault_vector, frame, predicted_obs, final_correction = decoder.decode(dets[s])

        recombined = 0
        for m in decoder.obs_groups[0]:
            q = decoder.meas_to_qubit[m]
            recombined ^= final_correction[q]
        if recombined == predicted_obs[0]:
            n_self_consistent += 1

        ref_pred = matching_ref.decode(dets[s])
        if predicted_obs[0] == ref_pred[0]:
            n_matches_reference += 1

    assert n_self_consistent == shots, "final_correction XOR must match predicted_obs"
    assert n_matches_reference == shots, (
        "decoder must match reference pymatching decoder bit-for-bit"
    )


def test_per_fault_ground_truth_vs_dem_observables():
    """Eager mode: full table available immediately after construction."""
    circuit = _surface_code_circuit()
    decoder = PhysicalFrameDecoder(circuit, lazy=False)
    assert decoder._checkpoints is None
    _assert_ground_truth(decoder)


def test_physical_frame_decoder_correctness():
    circuit = _surface_code_circuit()
    decoder = PhysicalFrameDecoder(circuit, lazy=False)
    _assert_decode_correctness(decoder, circuit)


def test_lazy_decode_then_ground_truth():
    """Stage 3: decode first (resolving only selected faults), then full-table
    ground truth must still pass — laziness changes when work runs, not results."""
    circuit = _surface_code_circuit()
    decoder = PhysicalFrameDecoder(circuit, lazy=True)
    assert decoder._checkpoints is not None
    assert len(decoder._final_frame_contributions) == 0

    _assert_decode_correctness(decoder, circuit, shots=50)
    assert 0 < len(decoder._final_frame_contributions) < decoder.num_errors

    _assert_ground_truth(decoder)
    assert len(decoder._final_frame_contributions) == decoder.num_errors
