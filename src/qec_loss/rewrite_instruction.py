import stim
import numpy as np
from .utils import _MEASURE_GATES, _RESET_GATES, _1Q_ERROR_GATES, _2Q_ERROR_GATES, _1Q_GATES, _2Q_GATES


def rewrite_instruction(
    instruction: stim.CircuitInstruction,
    lost_qubits: set[int],
    meas_index: int,
) -> tuple[list[stim.CircuitInstruction], set[int], set[int], int]:
    """Rewrite a stim instruction to ignore lost qubits. Returns the rewritten instruction and the updated set of lost qubits.

    Args:
        instruction: The original stim instruction to rewrite.
        lost_qubits: A set of qubit indices that are currently considered lost.
        meas_index: The current measurement index, used to determine which qubits are being measured.

    Returns:
        - The new instruction with lost qubits removed
        - The updated set of lost qubits
        - A list of measurement indices that should be overwritten to 2.
        - The new measurement index.
    """
    targets = instruction.targets_copy()
    gate_args = instruction.gate_args_copy()

    if instruction.name in _1Q_GATES.union(_1Q_ERROR_GATES):
        new_targets = [t for t in targets if t.value not in lost_qubits]
        if new_targets:
            new_instruction = stim.CircuitInstruction(instruction.name, new_targets, gate_args)
            return [new_instruction], lost_qubits, set(), meas_index
        else:
            return [], lost_qubits, set(), meas_index
    elif instruction.name in _2Q_GATES.union(_2Q_ERROR_GATES):
        if lost_qubits.intersection({t.value for t in targets}):
            new_targets = []
            for i in range(0, len(targets), 2):
                control = targets[i]
                target = targets[i + 1]

                if control.value in lost_qubits or target.value in lost_qubits:
                    continue

                new_targets.append(control)
                new_targets.append(target)
            if new_targets:
                new_instruction = stim.CircuitInstruction(instruction.name, new_targets, gate_args)
                return [new_instruction], lost_qubits, set(), meas_index
            else:
                return [], lost_qubits, set(), meas_index
        else:
            return [instruction], lost_qubits, set(), meas_index
    elif instruction.name in _MEASURE_GATES.union(_RESET_GATES):
        rewritten_instructions = []
        lost_qubits = lost_qubits.copy()
        measure_mask = set()

        if instruction.name in _MEASURE_GATES:
            qubits_to_erase = []

            for t in targets:
                if t.value in lost_qubits:
                    measure_mask.add(meas_index)
                    qubits_to_erase.append(t)
                meas_index += 1

            if qubits_to_erase:
                rewritten_instructions.append(stim.CircuitInstruction("DEPOLARIZE1", qubits_to_erase, [0.75]))

        if instruction.name in _RESET_GATES:
            for t in targets:
                if t.value in lost_qubits:
                    lost_qubits.remove(t.value)

        rewritten_instructions.append(instruction)

        return rewritten_instructions, lost_qubits, measure_mask, meas_index
    # elif instruction.name == "OBSERVABLE_INCLUDE":
    #     raise NotImplementedError("OBSERVABLE_INCLUDE is not supported in rewrite_instruction.")
    else:
        return [instruction], lost_qubits, set(), meas_index
