import qec_loss._fast_qec_loss as fast_qec_loss
from qec_loss.add_loss_noise import add_loss_noise
import stim
import numpy as np
import pymatching as pm
import time

def get_ler(loss_prob: float, depol_prob: float, num_samples: int = 1000):
    stim_circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=3, distance=3, after_clifford_depolarization=depol_prob).flattened()
    lossy_circuit = fast_qec_loss.LossyCircuit(add_loss_noise(stim_circuit, loss_before_2_qubit_gate=loss_prob).__str__())
    sampler = fast_qec_loss.ForwardSampler(lossy_circuit, seed=8)
    batch = sampler.sample(num_samples, reroute_observables=True) # use rerouted
    
    t0 = time.time()
    monaka_builder = fast_qec_loss.MonakaBuilder(lossy_circuit, model_path="./monaka_model")
    preds_monaka = monaka_builder.decode_batch(batch)
    t1 = time.time()
    
    matcher = pm.Matching.from_stim_circuit(lossy_circuit.nominal_circuit)
    preds_pm = matcher.decode_batch(batch.detectors)
    t2 = time.time()
    
    print(f"loss={loss_prob}, monaka_time={t1-t0:.2f}s, pm_time={t2-t1:.2f}s")
    return (batch.observables != preds_monaka).mean(), (batch.observables != preds_pm).mean()

depol = 0.01
losses = np.geomspace(1e-4, 5e-2, 5)

for loss in losses:
    monaka_ler, pm_ler = get_ler(loss, depol)
    print(f"monaka_ler={monaka_ler}, pm_ler={pm_ler}")

