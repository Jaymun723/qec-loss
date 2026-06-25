import stim

from qec_loss import get_stabilizers
from qec_loss.observable import DetsRerouter, PauliRerouter


REPETITION_CIRCUIT = stim.Circuit("""
    R 0 1 2
    CX 0 1
    CX 2 1
    M 1
    DETECTOR rec[-1]
    M 0 2
    DETECTOR rec[-3] rec[-2] rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-2]
""")


def test_get_stabilizers():
    stabilizers = get_stabilizers(REPETITION_CIRCUIT)
    assert len(stabilizers) == 3
    assert str(stabilizers[0]) == "+ZZZ"
    assert str(stabilizers[1]) == "+Z__"
    assert str(stabilizers[2]) == "+__Z"


def test_pauli_rerouter_properties():
    rerouter = PauliRerouter(REPETITION_CIRCUIT)
    assert len(rerouter.L) == 1
    assert str(rerouter.L[0]) == "+Z__"
    assert rerouter.meas_so_far == [3]
    assert rerouter.final_measurements == [0, 1, 2]


def test_pauli_rerouter_reroute():
    rerouter = PauliRerouter(REPETITION_CIRCUIT)
    assert rerouter.reroute(0, [], optimize=False) == [stim.target_rec(-2)]
    # Pauli rerouting cannot reconstruct when the observable qubit is lost.
    assert rerouter.reroute(0, [0], optimize=False) == []


def test_pauli_rerouter_get_S_and_L_matrices():
    rerouter = PauliRerouter(REPETITION_CIRCUIT)
    S, L = rerouter.get_S_and_L_matrices(0, lost_qubits=[0], optimize=False)
    assert S.rows == 6
    assert S.cols > 0
    assert L.rows == 6
    assert L.cols == 1


def test_dets_rerouter_auto_data_qubits_and_measurement_lookup():
    rerouter = DetsRerouter(REPETITION_CIRCUIT)
    assert rerouter.data_qubits == [0, 2]
    assert rerouter.get_qubit_for_measurement(1) == 0
