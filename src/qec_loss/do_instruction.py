import stim
import numpy as np
from .utils import _MEASURE_GATES, _RESET_GATES, _1Q_ERROR_GATES, _2Q_ERROR_GATES, _1Q_GATES, _2Q_GATES


def do_instruction(
    instruction: stim.CircuitInstruction,
    tableau: stim.TableauSimulator,
    measurements: np.ndarray,
    lost_qubits: set[int],
) -> set[int]:
    """Apply a stim instruction to the tableau and update measurements, ignoring lost qubits."""
    targets = instruction.targets_copy()
    gate_args = instruction.gate_args_copy()
    if instruction.name in _1Q_GATES.union(_1Q_ERROR_GATES):
        new_targets = [t for t in targets if t.value not in lost_qubits]
        if new_targets:
            new_instruction = stim.CircuitInstruction(instruction.name, new_targets, gate_args)
            tableau.do(new_instruction)
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
                tableau.do(new_instruction)
        else:
            tableau.do(instruction)
    elif instruction.name in _MEASURE_GATES.union(_RESET_GATES):
        tableau.do(instruction)

        if instruction.name in _MEASURE_GATES:
            meas = tableau.current_measurement_record()[-len(targets) :]
            for t, m in zip(targets, meas):
                if t.value not in lost_qubits:
                    measurements[:, t.value] = int(m)
                else:
                    measurements[:, t.value] = 2

        return lost_qubits - {t.value for t in targets}
    else:
        tableau.do(instruction)
    return lost_qubits
