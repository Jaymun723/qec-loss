import stim

from qec_loss import LossInstruction, LossyCircuit


def test_lossy_circuit_metadata_and_nominal_circuit():
    lossy = LossyCircuit("""
        R 0 1
        LOSS(0.1) 0
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)
    assert lossy.num_qubits == 2
    assert lossy.num_measurements == 2
    assert lossy.num_detectors == 1
    assert lossy.num_observables == 1
    assert "LOSS" not in str(lossy.nominal_circuit)
    assert lossy.nominal_circuit.num_measurements == 2


def test_loss_instruction_str():
    instr = LossInstruction([0, 1, 2], 0.25, "tag")
    assert str(instr) == "LOSS[tag](0.25) 0 1 2"


def test_lossy_circuit_rerouter_is_pauli_rerouter():
    from qec_loss.observable import PauliRerouter

    lossy = LossyCircuit("""
        R 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    assert isinstance(lossy.rerouter, PauliRerouter)
    assert len(lossy.rerouter.L) == 1
