import qec_loss._fast_qec_loss as fast_qec_loss
from qec_loss.add_loss_noise import add_loss_noise
import stim
import numpy as np
import pymatching as pm

stim_circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=3, distance=3, after_clifford_depolarization=0.01).flattened()
lossy_circuit = fast_qec_loss.LossyCircuit(add_loss_noise(stim_circuit, loss_before_2_qubit_gate=0.005).__str__())
sampler = fast_qec_loss.ForwardSampler(lossy_circuit, seed=8)

batch_no_reroute = sampler.sample(1000, reroute_observables=False)
batch_reroute = sampler.sample(1000, reroute_observables=True)

monaka_builder = fast_qec_loss.MonakaBuilder(lossy_circuit, model_path="./monaka_model")

preds_monaka_no_reroute = monaka_builder.decode_batch(batch_no_reroute)
preds_monaka_reroute = monaka_builder.decode_batch(batch_reroute)

matcher = pm.Matching.from_stim_circuit(lossy_circuit.nominal_circuit)
preds_pm_no_reroute = matcher.decode_batch(batch_no_reroute.detectors)
preds_pm_reroute = matcher.decode_batch(batch_reroute.detectors)

print("Monaka (no reroute):", (batch_no_reroute.observables != preds_monaka_no_reroute).mean())
print("Monaka (reroute):", (batch_reroute.observables != preds_monaka_reroute).mean())
print("PM (no reroute):", (batch_no_reroute.observables != preds_pm_no_reroute).mean())
print("PM (reroute):", (batch_reroute.observables != preds_pm_reroute).mean())

