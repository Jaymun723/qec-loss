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


def test_per_fault_ground_truth_vs_dem_observables():
    """Stage 1 validation: each fault's simulated final-frame contribution
    must reproduce the DEM's recorded observable-membership bits (0/N)."""
    circuit = _surface_code_circuit()
    decoder = PhysicalFrameDecoder(circuit)

    mismatches = 0
    for i in range(decoder.num_errors):
        contrib = decoder.final_frame_contributions[i]
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


def test_physical_frame_decoder_correctness():
    circuit = _surface_code_circuit()
    decoder = PhysicalFrameDecoder(circuit)
    dem_ref = circuit.detector_error_model(decompose_errors=True)
    matching_ref = pymatching.Matching.from_detector_error_model(dem_ref)

    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=100, separate_observables=True)

    n_self_consistent = 0
    n_matches_reference = 0

    for s in range(100):
        fault_vector, frame, predicted_obs, final_correction = decoder.decode(dets[s])

        # (a) internal consistency
        recombined = 0
        for m in decoder.obs_groups[0]:
            q = decoder.meas_to_qubit[m]
            recombined ^= final_correction[q]
        if recombined == predicted_obs[0]:
            n_self_consistent += 1

        # (b) match with reference
        ref_pred = matching_ref.decode(dets[s])
        if predicted_obs[0] == ref_pred[0]:
            n_matches_reference += 1

    assert n_self_consistent == 100, "final_correction XOR must match predicted_obs"
    assert n_matches_reference == 100, "decoder must match reference pymatching decoder bit-for-bit"
