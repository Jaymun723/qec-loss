# qec-loss API

## Lossy circuits

```python
from qec_loss import LossyCircuit, add_loss_noise
import stim

# Build directly from a stim+LOSS string
lossy_circuit = LossyCircuit("""
    R 0 1
    LOSS(0.1) 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
""")

# Or inject LOSS instructions into an existing stim circuit
stim_circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z", rounds=5, distance=5
)
lossy_circuit = add_loss_noise(stim_circuit, loss_after_2_qubit_gate=0.01)
```

## Forward sampling

```python
from qec_loss import ForwardSampler, LossyCircuit

lossy_circuit = LossyCircuit("...")
sampler = ForwardSampler(lossy_circuit, seed=7)
batch = sampler.sample(10_000, reroute_observable=True)

measurements = batch.measurements
detectors = batch.detectors
observables = batch.observables
loss_patterns = batch.loss_patterns
```

## Monaka decoding

```python
from qec_loss import ForwardSampler, MonakaBuilder, LossyCircuit

lossy_circuit = LossyCircuit("...")
builder = MonakaBuilder(lossy_circuit)
sampler = ForwardSampler(lossy_circuit, seed=7)
batch = sampler.sample(10_000)

predictions = builder.decode_batch(batch)
nominal_only = builder.decode_batch(batch, include_loss_dem=False)
```

## Observable rerouting

```python
from qec_loss.observable import DetsRerouter, PauliRerouter
import stim

circuit = stim.Circuit("...")
rerouter = DetsRerouter(circuit, data_qubits=[0, 2])
targets = rerouter.reroute(observable_index=0, lost_qubits=[0], optimize=True)
```

## Physical frame decoding

`PhysicalFrameDecoder` turns a syndrome into a per-qubit correction you can
XOR onto final data measurements. ``OBSERVABLE_INCLUDE`` is optional: without
it the decoder still tracks every non-reset ``M``/``MX``/``MY``/``MZ`` target
(the final data layer in a typical memory experiment) and returns
``final_correction`` for all of them. Matching uses detectors only;
``predicted_obs`` is empty — fold corrections into your own logical.

```python
import stim
from qec_loss.frame import PhysicalFrameDecoder

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.01,
)

# Drop OBSERVABLE_INCLUDE entirely — not needed for frame / final_correction.
circuit_no_obs = stim.Circuit()
for instr in circuit:
    if instr.name != "OBSERVABLE_INCLUDE":
        circuit_no_obs.append(instr)

decoder = PhysicalFrameDecoder(circuit_no_obs)  # tracks all 9 data qubits
# Or restrict: PhysicalFrameDecoder(circuit_no_obs, data_qubits=[1, 3, 5])

measurements = circuit_no_obs.compile_sampler().sample(shots=100)
dets = circuit_no_obs.compile_m2d_converter().convert(
    measurements=measurements, append_observables=False
)

fault_vector, frame, predicted_obs, final_correction = decoder.decode(dets[0])
# final_correction: {qubit: 0/1} for every tracked data qubit
# predicted_obs is empty (no OBSERVABLE_INCLUDE)

# Apply the frame to every final data measurement:
corrected = {
    q: int(measurements[0, m]) ^ final_correction[q]
    for m, q in zip(decoder._final_meas_indices, decoder.final_qubits)
}
# Then XOR any logical support you care about, e.g. Z-logical on qubits 1,3,5:
logical = corrected[1] ^ corrected[3] ^ corrected[5]
```

If the circuit *does* keep ``OBSERVABLE_INCLUDE``, ``obs_groups`` /
``predicted_obs`` still work, and ``final_correction`` remains a correction
for **all** data qubits (not only the observable support).

`decode_batch` is the same pipeline over many shots via
`pymatching.Matching.decode_batch`. Pass `lazy=True` to defer per-fault
simulation until a fault is first selected.

## Module layout

| Module | Contents |
|--------|----------|
| `qec_loss.circuit` | `LossInstruction`, `LossyCircuit` |
| `qec_loss.sampler` | `ForwardSampler`, `SampleBatch`, `Sampler` |
| `qec_loss.observable` | `DetsRerouter`, `PauliRerouter`, `get_stabilizers` |
| `qec_loss.monaka` | `MonakaBuilder`, `LifeCycleManager`, `get_loss_dem`, ... |
| `qec_loss.f2_tensor` | `F2Tensor`, `PackedF2Matrix` |
| `qec_loss.frame` | `PhysicalFrameDecoder` |

The compiled extension lives at `qec_loss._native` and should not be imported directly.
