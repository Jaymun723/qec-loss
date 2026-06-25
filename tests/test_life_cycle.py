from qec_loss import LifeCycleManager, LifeSegment, LossyCircuit


def test_life_segment_str():
    segment = LifeSegment(qubit=3, start=1, end=10)
    text = str(segment)
    assert "3" in text
    assert "1" in text
    assert "10" in text


def test_life_cycle_manager():
    # circuit = fast_qec_loss.LossyCircuit("""
    #     R 1 2
    #     REPEAT 2 {
    #         R 0
    #         CX 1 0
    #         DEPOLARIZE2(0.01) 0 1
    #         LOSS(0.01) 0 1
    #         CX 2 0
    #         DEPOLARIZE2(0.01) 0 2
    #         LOSS(0.01) 0 2
    #         MR 0
    #         DETECTOR(0, 0, 0) rec[-1]
    #         SHIFT_COORDS(0, 0, 1)
    #     }
    #     M 1 2
    #     DETECTOR(0, 0, 0) rec[-1] rec[-2]
    #     OBSERVABLE_INCLUDE(0) rec[-1]
    # """)
    circuit = LossyCircuit("""
        R 1 2
        R 0
        CX 1 0
        DEPOLARIZE2(0.01) 0 1
        LOSS(0.01) 0 1
        CX 2 0
        DEPOLARIZE2(0.01) 0 2
        LOSS(0.01) 0 2
        M 0
        DETECTOR(0, 0, 0) rec[-1]
        R 0
        CX 1 0
        DEPOLARIZE2(0.01) 0 1
        LOSS(0.01) 0 1
        CX 2 0
        DEPOLARIZE2(0.01) 0 2
        LOSS(0.01) 0 2
        M 0
        DETECTOR(0, 0, 1) rec[-1]
        M 1 2
        DETECTOR(0, 0, 2) rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)

    life_manager = LifeCycleManager(circuit)

    # ancilla life cycle (qubit 0)
    ancilla_life_cycle = life_manager.get_life_cycle(0)
    assert len(ancilla_life_cycle) == 2
    assert ancilla_life_cycle[0].start == 0
    assert ancilla_life_cycle[0].end == 10
    assert ancilla_life_cycle[0].loss_locations == [4, 7]
    assert ancilla_life_cycle[1].start == 10
    assert ancilla_life_cycle[1].end == 22
    assert ancilla_life_cycle[1].loss_locations == [13, 16]

    # data qubit 1 life cycle (qubit 1)
    data1_life_cycle = life_manager.get_life_cycle(1)
    assert len(data1_life_cycle) == 1
    assert data1_life_cycle[0].start == 0
    assert data1_life_cycle[0].end == 22
    assert data1_life_cycle[0].loss_locations == [4, 13]

    # data qubit 2 life cycle (qubit 2)
    data2_life_cycle = life_manager.get_life_cycle(2)
    assert len(data2_life_cycle) == 1
    assert data2_life_cycle[0].start == 0
    assert data2_life_cycle[0].end == 22
    assert data2_life_cycle[0].loss_locations == [7, 16]

    # measurement to life segment mapping
    life_segment = life_manager.get_life_segment_for_measurement(0)
    assert life_segment.qubit == 0
    assert life_segment.start == 0
    assert life_segment.end == 10
    assert life_segment.loss_locations == [4, 7]

    life_segment = life_manager.get_life_segment_for_measurement(1)
    assert life_segment.qubit == 0
    assert life_segment.start == 10
    assert life_segment.end == 22
    assert life_segment.loss_locations == [13, 16]

    life_segment = life_manager.get_life_segment_for_measurement(2)
    assert life_segment.qubit == 1
    assert life_segment.start == 0
    assert life_segment.end == 22
    assert life_segment.loss_locations == [4, 13]

    life_segment = life_manager.get_life_segment_for_measurement(3)
    assert life_segment.qubit == 2
    assert life_segment.start == 0
    assert life_segment.end == 22
    assert life_segment.loss_locations == [7, 16]


def test_fresh_start():
    circuit = LossyCircuit("""
        R 0
        LOSS(0.5) 0 1
        R 0 1
        LOSS(0.5) 0 1
        M 0 1
    """)

    manager = LifeCycleManager(circuit)

    # qubit 0 should behave normally
    life_cycle_0 = manager.get_life_cycle(0)
    assert len(life_cycle_0) == 2
    assert life_cycle_0[0].start == 0
    assert life_cycle_0[0].end == 2
    assert life_cycle_0[0].loss_locations == [1]
    assert life_cycle_0[1].start == 2
    assert life_cycle_0[1].end == 5
    assert life_cycle_0[1].loss_locations == [3]

    # qubit 1 also, event though it is never first reset
    life_cycle_1 = manager.get_life_cycle(1)
    assert len(life_cycle_1) == 2
    assert life_cycle_1[0].start == 0
    assert life_cycle_1[0].end == 2
    assert life_cycle_1[0].loss_locations == [1]
    assert life_cycle_1[1].start == 2
    assert life_cycle_1[1].end == 5
    assert life_cycle_1[1].loss_locations == [3]
