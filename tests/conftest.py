import stim
import pytest

from qec_loss import add_loss_noise, LossyCircuit


@pytest.fixture
def example_circuit():
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=5, distance=5, after_clifford_depolarization=0.01
    )
    return add_loss_noise(
        stim_circuit,
        loss_after_2_qubit_gate=0.01,
    )
