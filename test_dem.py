import qec_loss._fast_qec_loss as fast_qec_loss
from qec_loss.add_loss_noise import add_loss_noise
import stim

stim_circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=3, distance=3, after_clifford_depolarization=0.01).flattened()
lossy_circuit = fast_qec_loss.LossyCircuit(add_loss_noise(stim_circuit, loss_before_2_qubit_gate=0.005).__str__())
monaka_builder = fast_qec_loss.MonakaBuilder(lossy_circuit, model_path="./monaka_model")

# Get life cycle manager
lcm = monaka_builder.life_cycle_manager
segment = lcm.get_life_segment_for_measurement(0)

dem = monaka_builder.get_life_segment_dem([segment.qubit], segment)
print("DEM:")
print(dem)
