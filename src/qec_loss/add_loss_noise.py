import stim

from qec_loss.circuit import LossInstruction, LossyCircuit
from qec_loss.utils import _1Q_GATES, _2Q_GATES, _RESET_GATES


def _qubit_targets(instruction: stim.CircuitInstruction) -> list[int]:
    return [t.value for t in instruction.targets_copy() if t.is_qubit_target]


def add_loss_noise(
    circuit: stim.Circuit,
    loss_before_2_qubit_gate: float = 0.0,
    loss_after_2_qubit_gate: float = 0.0,
    loss_after_1_qubit_gate: float = 0.0,
    loss_after_reset_gate: float = 0.0,
    loss_after_tick: float = 0.0,
) -> LossyCircuit:
    """Adds loss noise to a circuit.

    Args:
        circuit: The circuit to add loss noise to.
        loss_before_2_qubit_gate: The probability of loss noise before 2-qubit gates.
        loss_after_2_qubit_gate: The probability of loss noise after 2-qubit gates.
        loss_after_1_qubit_gate: The probability of loss noise after 1-qubit gates.
        loss_after_reset_gate: The probability of loss noise after reset gates.
        loss_after_tick: The probability of loss noise after each tick.

    Returns:
        A new circuit with loss noise added.
    """
    lines: list[str] = []
    for instruction in circuit.flattened():
        if instruction.name in _2Q_GATES and loss_before_2_qubit_gate > 0.0:
            lines.append(str(LossInstruction(_qubit_targets(instruction), loss_before_2_qubit_gate)))
        lines.append(str(instruction))
        if instruction.name in _1Q_GATES and loss_after_1_qubit_gate > 0.0:
            lines.append(str(LossInstruction(_qubit_targets(instruction), loss_after_1_qubit_gate)))
        elif instruction.name in _2Q_GATES and loss_after_2_qubit_gate > 0.0:
            lines.append(str(LossInstruction(_qubit_targets(instruction), loss_after_2_qubit_gate)))
        elif instruction.name in _RESET_GATES and loss_after_reset_gate > 0.0:
            lines.append(str(LossInstruction(_qubit_targets(instruction), loss_after_reset_gate)))
        if instruction.name == "TICK" and loss_after_tick > 0.0:
            raise NotImplementedError("Adding loss after TICK instructions is not currently supported.")
    return LossyCircuit("\n".join(lines))
