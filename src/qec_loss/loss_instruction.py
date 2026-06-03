import stim
import re


class LossInstruction:
    """Represents a custom non-Stim instruction for physical qubit loss events.

    This instruction tracks which physical qubits are subject to loss at a specific
    location in the circuit, and what the loss probability is.
    """

    def __init__(
        self,
        targets: list[int | stim.GateTarget],
        gate_args: list[float],
        *,
        tag: str | None = None,
    ):
        self._targets = []
        for t in targets:
            if isinstance(t, stim.GateTarget):
                self._targets.append(t)
            else:
                self._targets.append(stim.GateTarget(t))

        gate_args_list = list(gate_args)
        if len(gate_args_list) != 1:
            raise ValueError("LOSS instruction must have exactly one gate argument representing the loss probability.")
        self._gate_args = gate_args_list
        self._tag = tag

    # from str constructor:
    @classmethod
    def from_str(cls, instruction_str: str) -> "LossInstruction":
        """
        Create a LossInstruction from a string representation.

        Usage example:
        ```python
        instruction = LossInstruction.from_str("LOSS[loss_event](0.1) 0 1 2")
        print(instruction)  # Output: LOSS[loss_event](0.1) 0 1 2
        ```
        """
        # Regex to match the instruction format: LOSS[tag](gate_arg) target1 target2 ...
        pattern = r"^LOSS(?:\[(?P<tag>[^\]]+)\])?\((?P<gate_arg>[^)]+)\)\s+(?P<targets>.+)$"
        match = re.match(pattern, instruction_str.strip())
        if not match:
            raise ValueError(f"Invalid instruction format: {instruction_str}")

        tag = match.group("tag")
        gate_arg = float(match.group("gate_arg"))
        targets_str = match.group("targets")

        targets = []
        for t in targets_str.split():
            try:
                targets.append(int(t))
            except ValueError:
                raise ValueError(f"Invalid target '{t}' in instruction string. Targets must be integers.")

        return cls(targets=targets, gate_args=[gate_arg], tag=tag)

    def __str__(self) -> str:
        target_str = " ".join(str(t.value) for t in self._targets).strip()
        return f"LOSS{f'[{self._tag}]' if self._tag else ''}({self._gate_args[0]}) {target_str}"

    def __repr__(self) -> str:
        return self.__str__()

    def targets_copy(self) -> list[stim.GateTarget]:
        return self._targets.copy()

    def gate_args_copy(self) -> list[float]:
        return self._gate_args.copy()
