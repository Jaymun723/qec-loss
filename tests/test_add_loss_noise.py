import pytest
import stim

from qec_loss import LossyCircuit, add_loss_noise
from qec_loss.utils import count_measurements


def test_count_measurements():
    circuit = stim.Circuit("""
        M 0 1
        MR 2
        X 0
        M 3
    """)
    assert count_measurements(circuit) == 4


def test_add_loss_noise_inserts_loss_instructions():
    circuit = stim.Circuit("""
        R 0 1
        CX 0 1
        H 0
        M 0 1
    """)
    lossy = add_loss_noise(
        circuit,
        loss_before_2_qubit_gate=0.01,
        loss_after_2_qubit_gate=0.02,
        loss_after_1_qubit_gate=0.03,
        loss_after_reset_gate=0.04,
    )
    text = str(lossy)
    assert "LOSS(0.04) 0 1" in text
    assert "LOSS(0.01) 0 1" in text
    assert "LOSS(0.02) 0 1" in text
    assert "LOSS(0.03) 0" in text
    assert text.count("LOSS") == 4
    assert isinstance(lossy, LossyCircuit)


def test_add_loss_noise_zero_rates_adds_nothing():
    circuit = stim.Circuit("R 0\nCX 0 1\nM 0")
    lossy = add_loss_noise(circuit)
    assert "LOSS" not in str(lossy)
    assert lossy.nominal_circuit == circuit


def test_add_loss_noise_tick_not_supported():
    circuit = stim.Circuit("R 0\nTICK\nM 0")
    with pytest.raises(NotImplementedError, match="TICK"):
        add_loss_noise(circuit, loss_after_tick=0.1)
