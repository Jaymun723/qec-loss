We need:

- CircuitSampler
- MeasSampler(with_pattern: bool)
- DetsSampler(with_pattern: bool)

loss_events[loss_index]: {qubit_losts}

```python
lossy_circuit = LossyCircuit("""
    R 0 1
    LOSS(0.1) 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
""")

# ``method`` can be either: forward, graph or hybrid
meas_sampler = lossy_circuit.compile_sampler(method="forward", seed=7)
measurements = meas_sampler.sample(10_000)
# or
measurments, loss_pattern = meas_sampler.sample(10_000, include_loss_pattern=True)

# not really usefull in itslef
dets_sampler = lossy_circuit.compile_detector_sampler(method="forward", seed=7)
dets, obs = dets_sampler.sample(10_000)

# best
exp_sampler = lossy_circuit.compile_experiment_sample(method="forward", seed=7)
exps = exp_sampler.sample(10_000)
# you have access to: exps.detectors, exps.observables, exps.measurments, exps.loss_patterns
```

```python
monaka = Monaka(lossy_circuit)
# monaka.prebuild()
preds = monaka_builder.decode_batch(exps)
preds_nominal_only = monaka_builder.decode_batch(exps, include_loss_dem=False)

ler = (preds != obs).mean()
```