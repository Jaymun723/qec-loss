import stim
import pytest

from qec_loss.observable import DetsRerouter


def test_rerouter_basic():
    # Repetition code circuit with proper detectors:
    # Qubits: 0 (data), 1 (ancilla), 2 (data)
    # Measurements:
    # 0: M 1 (ancilla)
    # 1: M 0 (data)
    # 2: M 2 (data)
    # Detector 0: M 1 (comparing to initial |000> state)
    # Detector 1: M 1 ^ M 0 ^ M 2 (comparing round 1 to final data measurements)
    # Observable: M 0
    circuit = stim.Circuit("""
        R 0 1 2
        CX 0 1
        CX 2 1
        M 1
        DETECTOR rec[-1]
        M 0 2
        DETECTOR rec[-3] rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)

    # Data qubits are 0 and 2
    rerouter = DetsRerouter(circuit, [0, 2])

    # 1. No lost qubits: should return the original observable [rec[-2]] (M 0)
    targets_no_loss = rerouter.reroute(0, [], optimize=False)
    assert len(targets_no_loss) == 1
    assert targets_no_loss[0] == stim.target_rec(-2)

    # 2. Lost qubit 0: M 0 (index 1 / rec[-2]) is bad.
    # Reconstructed observable should be M 2 (rec[-1] relative to the end of the circuit)
    targets_loss_0 = rerouter.reroute(0, [0], optimize=False)
    assert len(targets_loss_0) == 1
    assert targets_loss_0[0] == stim.target_rec(-1)

    # 3. Lost qubit 2: M 2 (index 2 / rec[-1]) is bad.
    # But the observable is M 0 (index 1 / rec[-2]).
    # Since qubit 2 is lost but the observable only depends on qubit 0 (which is not lost),
    # we don't need any changes! The original observable is already valid.
    targets_loss_2 = rerouter.reroute(0, [2], optimize=False)
    assert len(targets_loss_2) == 1
    assert targets_loss_2[0] == stim.target_rec(-2)

    # 4. Lost both qubits 0 and 2: impossible to reconstruct.
    targets_loss_both = rerouter.reroute(0, [0, 2], optimize=False)
    assert len(targets_loss_both) == 0


def test_rerouter_optimization():
    # We want to test weight optimization.
    # Measurements:
    # 0: M 0 (data)
    # 1: M 3 (data)
    # 2: M 4 (data)
    # 3: M 1 (ancilla)
    # 4: M 2 (ancilla)
    # Qubits: 0, 3, 4 (data), 1, 2 (ancillas)
    # Detector 0: M 1 + M 2
    # Detector 1: M 2
    # Detector 2: M 0 + M 3 + M 4
    # Observable: M 0 + M 3 + M 1 (which requires M 1 and M 2 to be canceled)
    # Particular solution to cancel M 1: add Detector 0 and Detector 1
    # L_new = (M 0 + M 3 + M 1) + (M 1 + M 2) + (M 2) = M 0 + M 3 (weight 2).
    # Since Detector 2 is a stabilizer (M 0 + M 3 + M 4), we can also add it:
    # L_new = M 0 + M 3 + (M 0 + M 3 + M 4) = M 4 (weight 1).
    # The optimizer should find the weight 1 solution (M 4).
    circuit = stim.Circuit("""
        R 0 1 2 3 4
        M 0 3 4 1 2
        DETECTOR rec[-2] rec[-1]
        DETECTOR rec[-1]
        DETECTOR rec[-5] rec[-4] rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4] rec[-2]
    """)

    rerouter = DetsRerouter(circuit, [0, 3, 4])

    # Without optimization: we might get the particular solution (weight 2: M 0 + M 3)
    # With optimization: we must get the optimized solution (weight 1: M 4 / rec[-3])
    targets_optimized = rerouter.reroute(0, [], optimize=True)
    assert len(targets_optimized) == 1
    assert targets_optimized[0] == stim.target_rec(-3)  # M 4


def test_real_world():
    repetition = stim.Circuit.generated("repetition_code:memory", rounds=3, distance=3).flattened()

    rerouter_repetition = DetsRerouter(repetition)

    # Losing qubits 0 and 4 should still allow us to reconstruct using qubit 2.
    targets_loss_0_4 = rerouter_repetition.reroute(0, [0, 4], optimize=True)
    assert len(targets_loss_0_4) == 1
    assert targets_loss_0_4[0] == stim.target_rec(-2)  # M 2

    surface = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=3, distance=3).flattened()
    rerouter_surface = DetsRerouter(surface)

    # Losing qubit 3 and 10
    targets_no_optimized = rerouter_surface.reroute(0, [3, 10], optimize=False)
    targets_optimized = rerouter_surface.reroute(0, [3, 10], optimize=True)

    assert len(targets_no_optimized) > 0  # Should be able to reconstruct without optimization
    assert len(targets_optimized) > 0  # Should be able to reconstruct with optimization
    assert len(targets_optimized) <= len(targets_no_optimized)  # Optimization should not increase the number of targets
