import stim
import numpy as np
from .rewrite_instruction import rewrite_instruction
from .utils import count_measurements
from .lossy_circuit import LossyCircuit
from .loss_instruction import LossInstruction
from collections import defaultdict


class ForwardSampler:
    def __init__(self, circuit: LossyCircuit, shots: int, seed: int | None = None):
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        meas_count = count_measurements(circuit)
        self.measurements = np.zeros((shots, meas_count), dtype=int)

    def sample_one_shot(self, seed: int | None = None) -> tuple[stim.Circuit, set[int], dict[int, set[int]]]:
        """Sample one shot of the circuit.

        Returns:
            - The final rewritten circuit.
            - A set of measurement indices to overwrite to 2.
            - The loss pattern: a dictionary mapping loss indices to sets of lost qubit indices.
        """
        lost_qubits = set()
        meas_index = 0

        loss_index = 0
        loss_pattern = defaultdict(set)
        shot_measurement_mask = set()

        final_circuit = stim.Circuit()

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        for instruction in self.circuit.instructions:
            if isinstance(instruction, LossInstruction):
                targets = instruction.targets_copy()
                loss_prob = instruction.gate_args_copy()[0]
                dices = self.rng.uniform(size=len(targets)) < loss_prob
                for t, p in zip(targets, dices):
                    if p:
                        lost_qubits.add(t.value)
                        loss_pattern[loss_index].add(t.value)
                loss_index += 1
            else:
                rewritten_instructions, lost_qubits, measure_mask, meas_index = rewrite_instruction(
                    instruction, lost_qubits, meas_index
                )
                if measure_mask:
                    shot_measurement_mask.update(measure_mask)
                if rewritten_instructions:
                    for instr in rewritten_instructions:
                        final_circuit.append(instr)

        return final_circuit, shot_measurement_mask, loss_pattern
