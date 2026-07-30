"""Parity tests for the native `get_loss_segment_dem` fast path.

The native C++ function must reproduce the Python reference pipeline used by
the delayed erasure decoder:

    rewritten = qec_loss.monaka.get_loss_rewritten_circuits(circuit, segment)
    dems = [get_detector_error_model_gauge_observables(c) for c in rewritten]
    dem = qec_loss.combine_dems(dems, geometric_weights)
"""

import math
from collections import defaultdict

import numpy as np
import qec_loss
import stim
from qec_loss.utils import flattened_dem_instructions


def _get_detector_error_model_gauge_observables(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Python reference: surface_boss.utils.get_detector_error_model(decompose_errors=False).

    OBSERVABLE_INCLUDE instructions are converted to temporary detectors so
    that undeterministic observables are tolerated via allow_gauge_detectors,
    then mapped back to logical observables.
    """
    dets = circuit.num_detectors
    new_circuit = stim.Circuit()
    dets_to_obs = {}

    for op in circuit.flattened():
        if op.name == "OBSERVABLE_INCLUDE":
            args = op.gate_args_copy()
            assert len(args) == 1
            obs_idx = int(args[0])
            assert obs_idx not in dets_to_obs.values()
            dets_to_obs[dets - circuit.num_detectors] = obs_idx
            dets += 1
            new_circuit.append("DETECTOR", op.targets_copy())
        else:
            new_circuit.append(op)

    dem = new_circuit.detector_error_model(allow_gauge_detectors=True, decompose_errors=False).flattened()

    new_dem = stim.DetectorErrorModel()
    for instruction in flattened_dem_instructions(dem):
        if instruction.type == "error":
            new_targets = []
            for target in instruction.targets_copy():
                if target.is_relative_detector_id() and target.val >= circuit.num_detectors:
                    obs_idx = dets_to_obs[target.val - circuit.num_detectors]
                    target = stim.DemTarget.logical_observable_id(obs_idx)
                new_targets.append(target)
            new_dem.append("error", instruction.args_copy()[0], new_targets)
        elif instruction.type == "detector":
            (target,) = instruction.targets_copy()
            if target.is_relative_detector_id() and target.val >= circuit.num_detectors:
                obs_idx = dets_to_obs[target.val - circuit.num_detectors]
                new_dem.append("logical_observable", [], [stim.DemTarget.logical_observable_id(obs_idx)])
            else:
                new_dem.append(instruction)
        else:
            new_dem.append(instruction)
    return new_dem


def _reference_loss_segment_dem(
    circuit: qec_loss.LossyCircuit, life_segment, loss_2q: float
) -> stim.DetectorErrorModel:
    """Python reference pipeline (as in vqec's DelayedErasureDecoder)."""
    rewritten_circuits = qec_loss.monaka.get_loss_rewritten_circuits(circuit, life_segment)

    dems = []
    weights = np.ones(len(rewritten_circuits)) * loss_2q
    tot = 0.0
    for i, rewritten in enumerate(rewritten_circuits):
        dems.append(_get_detector_error_model_gauge_observables(rewritten))
        weights[i] *= (1 - loss_2q) ** i
        tot += weights[i]
    weights /= tot

    return qec_loss.combine_dems(dems, weights)


def _canonical_error_mechanisms(dem: stim.DetectorErrorModel) -> dict[tuple[tuple[str, int], ...], float]:
    """Order-insensitive physical error content of a DEM (no decomposition hints here)."""
    probabilities: dict[tuple[tuple[str, int], ...], list[float]] = defaultdict(list)
    for instruction in flattened_dem_instructions(dem):
        if instruction.type != "error":
            continue
        parity: set[tuple[str, int]] = set()
        for target in instruction.targets_copy():
            if target.is_separator():
                continue
            if target.is_relative_detector_id():
                key = ("D", target.val)
            elif target.is_logical_observable_id():
                key = ("L", target.val)
            else:
                raise ValueError(f"Unsupported DEM error target: {target}")
            if key in parity:
                parity.remove(key)
            else:
                parity.add(key)
        if parity:
            probabilities[tuple(sorted(parity))].append(float(instruction.args_copy()[0]))

    # The reference pipeline sums weighted probabilities for identical targets
    # (not bernoulli_xor), so compare with plain sums per signature.
    return {signature: sum(ps) for signature, ps in probabilities.items()}


def _assert_dems_equivalent(first: stim.DetectorErrorModel, second: stim.DetectorErrorModel) -> None:
    first_errors = _canonical_error_mechanisms(first)
    second_errors = _canonical_error_mechanisms(second)
    assert first_errors.keys() == second_errors.keys()
    for signature, p in first_errors.items():
        assert math.isclose(p, second_errors[signature], rel_tol=1e-12, abs_tol=1e-12), (
            f"Probability mismatch for {signature}: {p} vs {second_errors[signature]}"
        )


def _lossy_circuit_with_segments(stim_name: str, loss_2q: float, **kwargs):
    circuit = stim.Circuit.generated(stim_name, **kwargs)
    lossy = qec_loss.add_loss_noise(circuit, loss_before_2_qubit_gate=loss_2q)
    lcm = lossy.compile_monaka_builder().life_cycle_manager
    segments = [
        segment for qubit in range(lossy.num_qubits) for segment in lcm.get_life_cycle(qubit) if segment.loss_locations
    ]
    return lossy, segments


def test_get_loss_segment_dem_repetition_code():
    loss_2q = 0.01
    lossy, segments = _lossy_circuit_with_segments("repetition_code:memory", loss_2q, distance=3, rounds=3)
    assert segments, "expected at least one life segment with loss locations"
    for segment in segments:
        native = qec_loss.monaka.get_loss_segment_dem(lossy, segment)
        reference = _reference_loss_segment_dem(lossy, segment, loss_2q)
        _assert_dems_equivalent(native, reference)


def test_get_loss_segment_dem_surface_code():
    loss_2q = 0.005
    lossy, segments = _lossy_circuit_with_segments("surface_code:rotated_memory_z", loss_2q, distance=3, rounds=3)
    assert segments, "expected at least one life segment with loss locations"
    for segment in segments:
        native = qec_loss.monaka.get_loss_segment_dem(lossy, segment)
        reference = _reference_loss_segment_dem(lossy, segment, loss_2q)
        _assert_dems_equivalent(native, reference)


def test_get_loss_segment_dems_batch():
    loss_2q = 0.01
    lossy, segments = _lossy_circuit_with_segments("repetition_code:memory", loss_2q, distance=3, rounds=3)
    batch = qec_loss.monaka.get_loss_segment_dems(lossy, segments)
    assert len(batch) == len(segments)
    for native, segment in zip(batch, segments):
        reference = _reference_loss_segment_dem(lossy, segment, loss_2q)
        _assert_dems_equivalent(native, reference)
