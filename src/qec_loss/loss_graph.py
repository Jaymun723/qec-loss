from .lossy_circuit import LossyCircuit
from .loss_instruction import LossInstruction
from .do_instruction import do_instruction
from .utils import count_measurements
import numpy as np
import stim


class LossGraph:
    def __init__(self, circuit: LossyCircuit, shots: int, seed: int | None = None):
        meas_count = count_measurements(circuit)
        self.measurements = np.zeros((shots, meas_count), dtype=int)
        tableau = stim.TableauSimulator(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.root = LossGraphNode(
            circuit,
            shots=shots,
            lost_qubits=set(),
            start_idx=0,
            rng=self.rng,
            tableau=tableau,
            measurements=self.measurements,
        )

    def print_tree(self) -> None:
        self.root.print_tree()


class LossGraphNode:
    def __init__(
        self,
        circuit: LossyCircuit,
        shots: int,
        lost_qubits: set[int],
        start_idx: int,
        rng: np.random.Generator,
        tableau: stim.TableauSimulator,
        measurements: np.ndarray,
    ):
        self.shots = shots
        self.children: list[LossGraphNode] = []

        for i in range(start_idx, len(circuit.instructions)):
            instruction = circuit.instructions[i]
            if isinstance(instruction, LossInstruction):
                targets = instruction.targets_copy()
                loss_prob = instruction.gate_args_copy()[0]
                dices = rng.uniform(size=(shots, len(targets))) < loss_prob
                uniq_patterns, counts = np.unique(dices, axis=0, return_counts=True)

                previous_count = 0

                for pattern, count in zip(uniq_patterns, counts):
                    new_lost_qubits = lost_qubits.union({t.value for t, p in zip(targets, pattern) if p})

                    child_node = LossGraphNode(
                        circuit,
                        shots=count,
                        lost_qubits=new_lost_qubits,
                        start_idx=i + 1,
                        rng=rng,
                        tableau=tableau.copy(),
                        measurements=measurements[previous_count : previous_count + count],
                    )

                    self.children.append(child_node)
                    previous_count += count

                break
            else:
                lost_qubits = do_instruction(instruction, tableau, measurements, lost_qubits)

    def draw(self) -> str:
        """Return an ASCII tree for debugging the branching structure.

        Each node prints its shot count so it is easy to see how probability mass
        is split across the tree.
        """

        lines: list[str] = []

        def render(node: LossGraphNode, prefix: str, is_last: bool, is_root: bool = False) -> None:
            details = f"shots={node.shots}"
            if is_root:
                lines.append(details)
            else:
                branch = "+-- " if is_last else "|-- "
                lines.append(f"{prefix}{branch}{details}")

            child_prefix = f"{prefix}{'    ' if is_last else '|   '}"
            for index, child in enumerate(node.children):
                render(child, child_prefix, index == len(node.children) - 1)

        render(self, "", True, True)
        return "\n".join(lines)

    def print_tree(self) -> None:
        print(self.draw())
