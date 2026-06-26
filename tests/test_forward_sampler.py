from qec_loss import ForwardSampler, LossyCircuit, SampleBatch
import numpy as np
import pytest


@pytest.mark.parametrize("seed", [None, 7])
def test_forward_sampler(example_circuit, seed):
    if seed:
        sampler = ForwardSampler(example_circuit, seed)
    else:
        sampler = ForwardSampler(example_circuit)

    batch = sampler.sample(10)
    assert isinstance(batch, SampleBatch)
    measurements = batch.measurements
    loss_patterns = batch.loss_patterns
    assert measurements.shape == (10, 145)
    assert batch.detectors.shape == (10, example_circuit.num_detectors)
    assert batch.observables.shape == (10, example_circuit.num_observables)
    assert len(loss_patterns) == 10
    assert sampler.circuit.num_qubits == example_circuit.num_qubits

    # Test __repr__
    repr_str = repr(batch)
    assert repr_str.startswith("SampleBatch(measurements=array(")
    assert "detectors=array(" in repr_str
    assert "observables=array(" in repr_str
    assert "loss_patterns=" in repr_str


def test_forward_sampler_reproducible_with_seed(example_circuit):
    batch_a = ForwardSampler(example_circuit, seed=123).sample(5)
    batch_b = ForwardSampler(example_circuit, seed=123).sample(5)
    assert np.array_equal(batch_a.measurements, batch_b.measurements)
    assert np.array_equal(batch_a.detectors, batch_b.detectors)


def test_forward_sampler_sample_measurements():
    lossy_circuit = LossyCircuit("""
        R 0 1
        LOSS(1) 0
        M 0 1
    """)
    sampler = ForwardSampler(lossy_circuit, seed=0)
    measurements, loss_patterns = sampler.sample_measurements(4)
    assert measurements.shape == (4, 2)
    assert len(loss_patterns) == 4
    for row, pattern in zip(measurements, loss_patterns):
        if pattern:
            idx, qubits = pattern[0]
            assert idx == 0
            assert (row[qubits] == 2).all()


def test_logic_forward():
    lossy_circuit = LossyCircuit("""
        R 0 1 2 3 4 5
        LOSS(0.5) 0 1 3 5
        M 0 1 2 3 4 5
    """)
    sampler = ForwardSampler(lossy_circuit)
    batch = sampler.sample(10)
    measurements = batch.measurements
    loss_patterns = batch.loss_patterns
    for meas, loss_pattern in zip(measurements, loss_patterns):
        assert len(loss_pattern) <= 1
        if len(loss_pattern) == 1:
            assert loss_pattern[0] != []
            assert (meas[loss_pattern[0]] == 2).all()  # Lost qubits should have measurement value 2
