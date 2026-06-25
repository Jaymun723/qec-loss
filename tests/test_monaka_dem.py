import stim

from qec_loss import (
    ForwardSampler,
    LifeCycleManager,
    LossyCircuit,
    MonakaBuilder,
    combine_circuits_into_dem,
    get_loss_dem,
)


def test_get_loss_dem():
    circuit = LossyCircuit("""
        R 0 1
        LOSS(0.5) 0
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)
    manager = LifeCycleManager(circuit)
    segment = manager.get_life_segment_for_measurement(0)
    dem = get_loss_dem(circuit, lost_qubits=[0], life_segment=segment)
    assert dem[0].type == "error"
    assert dem[0].args_copy()[0] == 0.5


def test_monaka_builder_dem_helpers(example_circuit, tmpdir):
    builder = MonakaBuilder(example_circuit, tmpdir)
    sampler = ForwardSampler(example_circuit, seed=0)
    batch = sampler.sample(3)

    manager = builder.life_cycle_manager
    segment = manager.get_life_segment_for_measurement(0)
    assert segment.qubit >= 0

    shot_dem = builder.get_dem_for_shot([], batch.measurements, shot_i=0)
    assert isinstance(shot_dem, stim.DetectorErrorModel)

    batch_dem = builder.get_dem_from_measurements(batch.measurements[0])
    assert isinstance(batch_dem, stim.DetectorErrorModel)


def test_monaka_get_life_segment_dem_simple(tmpdir):
    circuit = LossyCircuit("""
        R 0 1
        LOSS(0.5) 0
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)
    builder = MonakaBuilder(circuit, tmpdir)
    segment = LifeCycleManager(circuit).get_life_segment_for_measurement(0)
    dem = builder.get_life_segment_dem([0], segment)
    assert dem[0].type == "error"
    assert dem[0].args_copy()[0] == 0.5


def test_monaka_decode_batch_nominal_only(example_circuit, tmpdir):
    builder = MonakaBuilder(example_circuit, tmpdir)
    batch = ForwardSampler(example_circuit, seed=1).sample(10)
    decoded = builder.decode_batch(batch, include_loss_dem=False)
    assert decoded.shape == (10, 1)
    assert decoded.dtype == "uint8"


def test_combine_circuits_into_dem_equal_weights():
    c1 = stim.Circuit("""
        R 0
        X_ERROR(0.2) 0
        M 0
        DETECTOR rec[-1]
    """)
    c2 = stim.Circuit("""
        R 0
        X_ERROR(0.4) 0
        M 0
        DETECTOR rec[-1]
    """)
    combined = combine_circuits_into_dem([c1, c2])
    errors = [op.args_copy()[0] for op in combined if op.type == "error"]
    assert len(errors) == 1
    assert abs(errors[0] - 0.3) < 1e-9
