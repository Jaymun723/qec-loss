from qec_loss import ForwardSampler, LossyCircuit, MonakaBuilder, SampleBatch


def test_monaka_builder(example_circuit, tmpdir):
    builder = MonakaBuilder(example_circuit, tmpdir)

    assert builder.save_path.is_dir()

    nominal_dem = builder.get_nominal_dem()
    expected_dem = example_circuit.nominal_circuit.detector_error_model(
        decompose_errors=False,
        allow_gauge_detectors=False,
        approximate_disjoint_errors=0.0,
        ignore_decomposition_failures=True,
        block_decomposition_from_introducing_remnant_edges=False,
    )
    assert str(nominal_dem) == str(expected_dem)


def test_decode_batch(example_circuit, tmpdir):
    builder = MonakaBuilder(example_circuit, tmpdir)
    sampler = ForwardSampler(example_circuit, seed=42)
    batch = sampler.sample(10)

    decoded = builder.decode_batch(batch)
    assert decoded.shape == (10, 1)
    assert decoded.dtype == "uint8"
