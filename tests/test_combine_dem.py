import stim
from qec_loss.monaka import combine_circuits_into_dem


def test_combine_circuits_into_dem():
    # Circuit 1: has error D0 D1 with prob 0.1, and error D1 with prob 0.2
    c1 = stim.Circuit("""
        R 0 1 2
        X_ERROR(0.1) 0
        M 0
        DETECTOR(0) rec[-1]
        X_ERROR(0.2) 1
        M 1
        DETECTOR(1) rec[-1] rec[-2]
        DETECTOR(2)
        X_ERROR(0.5) 2
        M 2
        DETECTOR(3) rec[-1]
    """)

    assert c1.detector_error_model() == stim.DetectorErrorModel("""
        error(0.1) D0 D1
        error(0.2) D1
        error(0.5) D3
        detector(0) D0
        detector(1) D1
        detector(2) D2
        detector(3) D3
    """)

    # Circuit 2: has error D0 D1 with prob 0.2, and error D1 with prob 0.3 and error D2 with prob 0.4
    c2 = stim.Circuit("""
        R 0 1 2
        X_ERROR(0.2) 0
        M 0
        DETECTOR(0) rec[-1]
        X_ERROR(0.3) 1
        M 1
        DETECTOR(1) rec[-1] rec[-2]
        X_ERROR(0.4) 2
        M 2
        DETECTOR(2) rec[-1]
    """)

    assert c2.detector_error_model() == stim.DetectorErrorModel("""
        error(0.2) D0 D1
        error(0.3) D1
        error(0.4) D2
        detector(0) D0
        detector(1) D1
        detector(2) D2
    """)

    w = [0.8, 0.2]  # weights for c1 and c2 respectively

    # The expected combined DEM should contain:
    # - error(0.8 * 0.1 + 0.2 * 0.2) D0 D1 = error(0.12) D0 D1 (since D0 D1 is in both)
    # - error(0.8 * 0.2 + 0.2 * 0.3) D1 = error(0.22) D1 (since D1 is in both)
    # - error(0.2 * 0.4) D2 = error(0.08) D2 (since D2 is only in c2)
    # - error(0.8 * 0.5) D3 = error(0.4) D3 (since D3 is only in c1)
    # Let's parse the instructions of the combined DEM to assert correctness

    instructions = combine_circuits_into_dem([c1, c2], w)

    error_dict = {}
    for op in instructions:
        if op.type == "error":
            targets = tuple(sorted(t.val for t in op.targets_copy()))
            error_dict[targets] = op.args_copy()[0]

    print(error_dict)

    tol = 1e-9

    assert len(error_dict) == 4
    assert abs(error_dict[(0, 1)] - 0.12) < tol
    assert abs(error_dict[(1,)] - 0.22) < tol
    assert abs(error_dict[(2,)] - 0.08) < tol
    assert abs(error_dict[(3,)] - 0.4) < tol
