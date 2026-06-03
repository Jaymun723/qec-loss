import re
import stim
from typing import Optional, Any

# Group 1: Gate name | Group 2: Optional args in parens | Group 3: Targets
STIM_INSTRUCTION_REGEX = re.compile(r"^([A-Za-z0-9_]+)\s*(?:\(([^)]+)\))?\s*(.*)$")


def parse_stim_target(target_str: str) -> Any:
    """Converts a string target into a native Stim target object."""
    if target_str == "*":
        return stim.target_combiner()

    if target_str.startswith("rec["):
        idx = int(target_str[4:-1])
        return stim.target_rec(idx)

    if target_str.startswith("sweep["):
        idx = int(target_str[6:-1])
        return stim.target_sweep_bit(idx)

    if target_str.startswith("!"):
        idx = int(target_str[1:])
        return stim.target_inv(idx)

    if target_str.startswith("X") and target_str[1:].isdigit():
        idx = int(target_str[1:])
        return stim.target_x(idx)

    if target_str.startswith("Y") and target_str[1:].isdigit():
        idx = int(target_str[1:])
        return stim.target_y(idx)

    if target_str.startswith("Z") and target_str[1:].isdigit():
        idx = int(target_str[1:])
        return stim.target_z(idx)

    # If it's none of the special cases, it must be a standard integer qubit index
    return int(target_str)


def parse_stim_line(line: str) -> Optional[stim.CircuitInstruction]:
    """
    Parses a single line of Stim text format into a native stim.CircuitInstruction.
    """
    clean_line = line.split("#")[0].strip()
    if not clean_line:
        return None

    match = STIM_INSTRUCTION_REGEX.match(clean_line)
    if not match:
        raise ValueError(f"Invalid Stim instruction format: '{clean_line}'")

    gate_name = match.group(1).upper()
    raw_args = match.group(2)
    raw_targets = match.group(3)

    # Parse arguments
    args = []
    if raw_args:
        args = [float(arg.strip()) for arg in raw_args.split(",")]

    # Parse targets into native Stim objects
    targets = []
    if raw_targets:
        targets = [parse_stim_target(t) for t in raw_targets.split()]

    # Return the native Stim object
    return stim.CircuitInstruction(gate_name, targets, args)
