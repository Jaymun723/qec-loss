from qec_loss import LossyCircuit
from qec_loss.add_loss_noise import add_loss_noise
import stim


def test_lossy_circuit(tmpdir):
    # Test parsing of a valid lossy circuit
    d = 5
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=d, distance=d, after_clifford_depolarization=0.01
    )
    lossy_circuit = add_loss_noise(
        stim_circuit,
        loss_after_2_qubit_gate=0.00183,
        loss_after_reset_gate=1.1e-2,
        loss_after_1_qubit_gate=0.09e-6,
    )
    ref_file = tmpdir / "lossy_circuit_ref.stim"
    with open(ref_file, "w") as f:
        f.write(str(lossy_circuit))

    tmp_file = tmpdir / "lossy_circuit.stim"
    loaded_lossy_circuit = LossyCircuit.from_file(ref_file)
    loaded_lossy_circuit.to_file(tmp_file)

    with open(tmp_file, "r") as f:
        loaded_str = f.read()
        with open(ref_file, "r") as f_ref:
            ref_str = f_ref.read()
            assert loaded_str == ref_str


def test_lossy_circuit_example():
    lossy_circuit = LossyCircuit("""
        R 0 1
        LOSS(0.1) 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
    """)

def test_compile_forward_sampler():
    from qec_loss import ForwardSampler
    lossy_circuit = LossyCircuit("""
        R 0 1
        LOSS(0.1) 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
    """)
    sampler = lossy_circuit.compile_forward_sampler()
    assert isinstance(sampler, ForwardSampler)
    
    sampler_with_seed = lossy_circuit.compile_forward_sampler(seed=42)
    assert isinstance(sampler_with_seed, ForwardSampler)

def test_compile_monaka_builder():
    from qec_loss import MonakaBuilder
    lossy_circuit = LossyCircuit("""
        R 0 1
        LOSS(0.1) 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
    """)
    builder = lossy_circuit.compile_monaka_builder()
    assert isinstance(builder, MonakaBuilder)
    
    builder_with_opt = lossy_circuit.compile_monaka_builder(optimize_rerouting=True)
    assert isinstance(builder_with_opt, MonakaBuilder)
