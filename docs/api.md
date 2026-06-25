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

## Module layout

| Module | Contents |
|--------|----------|
| `qec_loss.circuit` | `LossInstruction`, `LossyCircuit` |
| `qec_loss.sampler` | `ForwardSampler`, `SampleBatch`, `Sampler` |
| `qec_loss.observable` | `DetsRerouter`, `PauliRerouter`, `get_stabilizers` |
| `qec_loss.monaka` | `MonakaBuilder`, `LifeCycleManager`, `get_loss_dem`, ... |
| `qec_loss.f2_tensor` | `F2Tensor`, `PackedF2Matrix` |

The compiled extension lives at `qec_loss._native` and should not be imported directly.
