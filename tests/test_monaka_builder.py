import qec_loss._fast_qec_loss as fast_qec_loss


def test_monaka_builder(example_circuit, tmpdir):
    builder = fast_qec_loss.MonakaBuilder(example_circuit, tmpdir)

    builder.prebuild()

    # assert a new folder has been created inside tmpdir
    assert len(tmpdir.listdir()) == 1
    monaka_folder = tmpdir.listdir()[0]
    assert monaka_folder.isdir()

    # assert the nominal circuit file has been created
    nominal_circuit_file = monaka_folder.join("nominal.dem")
    assert nominal_circuit_file.isfile()

    # assert the nominal dem mathc the original circuit
    with open(nominal_circuit_file) as f:
        nominal_circuit = f.read()
    assert nominal_circuit == (
        str(
            example_circuit.nominal_circuit.detector_error_model(
                decompose_errors=True,
                allow_gauge_detectors=False,
                approximate_disjoint_errors=1.0,
                ignore_decomposition_failures=False,
                block_decomposition_from_introducing_remnant_edges=False,
            )
        )
        + "\n"
    )
