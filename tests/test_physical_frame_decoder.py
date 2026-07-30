import numpy as np
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


def _strip_observable_include(circuit: stim.Circuit) -> stim.Circuit:
    """Return a copy of `circuit` with every OBSERVABLE_INCLUDE removed."""
    out = stim.Circuit()
    for instr in circuit:
        if instr.name == "OBSERVABLE_INCLUDE":
            continue
        out.append(instr)
    return out


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


def test_decode_batch_matches_per_shot():
    """Stage 4: decode_batch must agree with the per-shot decode loop."""
    circuit = _surface_code_circuit()
    decoder = PhysicalFrameDecoder(circuit, lazy=False)
    dets, _ = circuit.compile_detector_sampler().sample(
        shots=200, separate_observables=True
    )

    fault_vectors, frames, predicted_obs, final_corrections = decoder.decode_batch(dets)

    for s in range(len(dets)):
        fv, frame, obs, final = decoder.decode(dets[s])
        assert np.array_equal(fv, fault_vectors[s])
        assert frame == frames[s]
        assert np.array_equal(obs, predicted_obs[s])
        assert final == final_corrections[s]


def test_apply_frame_to_measurements_recovers_observable():
    """End-to-end: strip OBSERVABLE_INCLUDE, sample, decode, apply frame.

    Workflow this locks in:
      1. Build a surface-code memory circuit (with OBSERVABLE_INCLUDE) and a
         PhysicalFrameDecoder from it — the include lines tell the decoder
         which final measurements form each logical.
      2. Strip OBSERVABLE_INCLUDE and sample raw measurement bits (no stim
         observable tracking at sample time).
      3. Convert measurements → detectors, decode → final_correction.
      4. XOR final_correction onto the obs-group measurement bits; their
         parity is the decoded logical value (0 for Z-memory when the
         decoder is correct).
    """
    circuit = _surface_code_circuit(distance=3, rounds=3)
    circuit_no_obs = _strip_observable_include(circuit)
    assert circuit_no_obs.num_observables == 0
    assert circuit_no_obs.num_measurements == circuit.num_measurements
    assert circuit_no_obs.num_detectors == circuit.num_detectors

    decoder = PhysicalFrameDecoder(circuit, lazy=False)
    assert len(decoder.obs_groups) == 1

    shots = 200
    measurements = circuit_no_obs.compile_sampler().sample(shots=shots)
    dets = circuit_no_obs.compile_m2d_converter().convert(
        measurements=measurements, append_observables=False
    )
    # Ground-truth observable flips from the original declaration, applied to
    # the same measurement record (OBSERVABLE_INCLUDE does not affect sampling).
    _, true_obs = circuit.compile_m2d_converter().convert(
        measurements=measurements, separate_observables=True
    )

    n_frame_recovers_prepared = 0
    n_decode_correct = 0
    for s in range(shots):
        _, _, predicted_obs, final_correction = decoder.decode(dets[s])

        # Raw logical from measurements alone (== stim's true_obs for Z-memory).
        raw_obs = 0
        # Frame-corrected logical: XOR final_correction onto each obs-group bit.
        corrected_obs = 0
        correction_parity = 0
        for m in decoder.obs_groups[0]:
            q = decoder.meas_to_qubit[m]
            raw_obs ^= int(measurements[s, m])
            corrected_obs ^= int(measurements[s, m]) ^ final_correction[q]
            correction_parity ^= final_correction[q]

        assert raw_obs == int(true_obs[s, 0])
        assert correction_parity == int(predicted_obs[0])

        if predicted_obs[0] == true_obs[s, 0]:
            n_decode_correct += 1
            # Successful decode ⇒ corrected measurements recover the prepared |0⟩.
            assert corrected_obs == 0
            n_frame_recovers_prepared += 1
        else:
            assert corrected_obs == 1

    assert n_decode_correct == n_frame_recovers_prepared
    assert n_decode_correct > shots * 0.5  # well above chance at this noise


def test_decoder_without_observable_include_all_data_qubits():
    """Build on a circuit with no OBSERVABLE_INCLUDE; correct all data qubits."""
    circuit = _surface_code_circuit(distance=3, rounds=3)
    circuit_no_obs = _strip_observable_include(circuit)
    assert circuit_no_obs.num_observables == 0

    # Final data layer of a d=3 rotated memory: M 1 3 5 8 10 12 15 17 19
    expected_data = [1, 3, 5, 8, 10, 12, 15, 17, 19]

    decoder = PhysicalFrameDecoder(circuit_no_obs, lazy=False)
    assert decoder.obs_groups == []
    assert decoder.num_observables == 0
    assert decoder.final_qubits == expected_data

    # Same data qubits when built from the circuit that still has includes.
    decoder_with_obs = PhysicalFrameDecoder(circuit, lazy=False)
    assert set(decoder_with_obs.final_qubits) == set(expected_data)
    assert len(decoder_with_obs.obs_groups) == 1
    # Obs support is a proper subset of all data qubits.
    obs_qubits = {decoder_with_obs.meas_to_qubit[m] for m in decoder_with_obs.obs_groups[0]}
    assert obs_qubits < set(expected_data)

    measurements = circuit_no_obs.compile_sampler().sample(shots=50)
    dets = circuit_no_obs.compile_m2d_converter().convert(
        measurements=measurements, append_observables=False
    )
    fault_vector, frame, predicted_obs, final_correction = decoder.decode(dets[0])
    assert len(predicted_obs) == 0
    assert set(final_correction) == set(expected_data)
    assert all(b in (0, 1) for b in final_correction.values())

    # data_qubits override keeps only the listed qubits (last non-reset M each).
    subset = [1, 5, 19]
    decoder_sub = PhysicalFrameDecoder(circuit_no_obs, data_qubits=subset)
    assert decoder_sub.final_qubits == subset
    _, _, _, corr_sub = decoder_sub.decode(dets[0])
    assert list(corr_sub.keys()) == subset
