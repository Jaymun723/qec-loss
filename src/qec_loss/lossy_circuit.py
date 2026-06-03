from pathlib import Path
import stim
from .loss_instruction import LossInstruction
from .stim_parse import parse_stim_line


class LossyCircuit:
    def __init__(self):
        self.instructions: list[LossInstruction | stim.CircuitInstruction] = []

    @classmethod
    def from_file(cls, file_path: str | Path) -> "LossyCircuit":
        """
        Load a lossy circuit from a file.

        The file should contain lines of the form:
        ```
        LOSS[optional_tag](loss_probability) target1 target2 ...
        ```
        or any valid stim circuit instruction.

        Example usage:
        ```python
        lossy_circuit = LossyCircuit.from_file("path/to/lossy_circuit.txt")
        for instr in lossy_circuit.instructions:
            print(instr)
        ```

        Args:
            file_path: Path to the file containing the lossy circuit instructions.

        Returns:
            An instance of LossyCircuit with the loaded instructions.
        """
        lossy_circuit = cls()
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip empty lines and comments
                if line.startswith("LOSS"):
                    instruction = LossInstruction.from_str(line)
                    lossy_circuit.instructions.append(instruction)
                else:
                    # Assume it's a valid stim instruction
                    lossy_circuit.instructions.append(parse_stim_line(line))
        return lossy_circuit

    def __str__(self) -> str:
        return "\n".join(str(instr) for instr in self.instructions)

    def save_to(self, file_path: str | Path) -> None:
        """
        Save the lossy circuit to a file.

        Args:
            file_path: Path to the file where the lossy circuit should be saved.
        """
        with open(file_path, "w") as f:
            for instr in self.instructions:
                f.write(str(instr) + "\n")
