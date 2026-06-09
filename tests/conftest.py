import stim
import pytest
from qec_loss.add_loss_noise import add_loss_noise
from qec_loss._fast_qec_loss import LossyCircuit


@pytest.fixture
def example_circuit():
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=5, distance=5, after_clifford_depolarization=0.01
    )
    lossy_circuit = add_loss_noise(
        stim_circuit,
        loss_after_2_qubit_gate=0.01,
    )
    return LossyCircuit(str(lossy_circuit))
