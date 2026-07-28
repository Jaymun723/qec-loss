import stim


def flattened_instructions(circuit: stim.Circuit) -> list[stim.CircuitInstruction]:
    """Flatten a circuit into plain instructions.

    `Circuit.flattened()` expands REPEAT blocks, so no `CircuitRepeatBlock` can
    survive it -- but iterating a `Circuit` is still typed as yielding either.
    """
    return [instr for instr in circuit.flattened() if isinstance(instr, stim.CircuitInstruction)]


def count_measurements(circuit: stim.Circuit) -> int:
    """Count the total number of measurement targets in a stim circuit."""
    return sum(
        len(instr.targets_copy())
        for instr in flattened_instructions(circuit)
        if instr.name in {"M", "MR", "MRX", "MRY", "MRZ"}
    )


_1Q_GATES = {
    "I",
    "X",
    "Y",
    "Z",
    "C_XYZ",
    "C_ZYX",
    "H",
    "H_XY",
    "H_XZ",
    "H_YZ",
    "S",
    "SQRT_X",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "SQRT_Z",
    "SQRT_Z_DAG",
    "S_DAG",
}
_1Q_ERROR_GATES = {
    "HERALDED_ERASE",
    "HERALDED_PAULI_CHANNEL_1",
    "PAULI_CHANNEL_1",
    "DEPOLARIZE1",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
}

_2Q_GATES = {
    "CNOT",
    "CX",
    "CXSWAP",
    "CY",
    "CZ",
    "CZSWAP",
    "ISWAP",
    "ISWAP_DAG",
    "SQRT_XX",
    "SQRT_XX_DAG",
    "SQRT_YY",
    "SQRT_YY_DAG",
    "SQRT_ZZ",
    "SQRT_ZZ_DAG",
    "SWAP",
    "SWAPCX",
    "SWAPCZ",
    "XCX",
    "XCY",
    "XCZ",
    "YCX",
    "YCY",
    "YCZ",
    "ZCX",
    "ZCY",
    "ZCZ",
}
_2Q_ERROR_GATES = {
    "DEPOLARIZE2",
    "PAULI_CHANNEL_2",
}

_RESET_GATES = {
    "MR",
    "MRX",
    "MRY",
    "MRZ",
    "R",
    "RX",
    "RY",
    "RZ",
}
_MEASURE_GATES = {
    "M",
    "MR",
    "MRX",
    "MRY",
    "MRZ",
    "MX",
    "MY",
    "MZ",
}
