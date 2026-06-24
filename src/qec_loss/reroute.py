import stim
import numpy as np

# d = 3
# stim_circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=d, distance=d)
# flat_circuit = stim_circuit.flattened()
# # flat_circuit.to_file("./playground/surface_code_1_round_flat.circuit")


def solve_gf2(A, b):
    A = A.copy().astype(np.uint8)
    b = b.copy().astype(np.uint8)

    m, n = A.shape

    aug = np.concatenate([A, b[:, None]], axis=1)

    row = 0
    pivots = []

    for col in range(n):
        pivot = None

        for r in range(row, m):
            if aug[r, col]:
                pivot = r
                break

        if pivot is None:
            continue

        aug[[row, pivot]] = aug[[pivot, row]]

        for r in range(m):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]

        pivots.append(col)
        row += 1

    for r in range(m):
        if not aug[r, :-1].any() and aug[r, -1]:
            return None

    x = np.zeros(n, dtype=np.uint8)

    for r in reversed(range(len(pivots))):
        c = pivots[r]

        rhs = aug[r, -1]

        for j in range(c + 1, n):
            rhs ^= aug[r, j] & x[j]

        x[c] = rhs

    return x


def reroute(stim_circuit: stim.Circuit, lost: set[int]):
    flat_circuit = stim_circuit.flattened()
    meas_count = flat_circuit.num_measurements
    det_count = flat_circuit.num_detectors
    obs_count = flat_circuit.num_observables
    det_matrix = np.zeros((det_count, meas_count), dtype=int)
    obs_matrix = np.zeros((obs_count, meas_count), dtype=int)
    measurement_record: list[int] = []
    meas_index = 0
    det_index = 0
    obs_index = 0
    support: dict[int, dict[int, int]] = {}
    last_reset: dict[int, int] = {}

    for iop, op in enumerate(flat_circuit):
        if op.name in {"M", "MR", "MY"}:
            for t in op.targets_copy():
                qubit = t.value
                sliced_circuit = flat_circuit[last_reset.get(qubit, 0) : iop]

                inv_tableau = stim.Tableau.from_circuit(
                    sliced_circuit, ignore_reset=True, ignore_measurement=True, ignore_noise=True
                ).inverse()
                measured_pauli = stim.PauliString(sliced_circuit.num_qubits)
                measured_pauli[qubit] = "Z" if op.name in {"M", "MR"} else "Y"

                stabilizer = inv_tableau(measured_pauli)
                stabilizer_indices = stabilizer.pauli_indices()

                def int_to_pauli(i: int) -> str:
                    match i:
                        case 1:
                            return "X"
                        case 2:
                            return "Y"
                        case 3:
                            return "Z"
                        case _:
                            raise ValueError(f"Invalid integer {i} for Pauli conversion.")

                support[meas_index] = {idx: (int_to_pauli(stabilizer[idx])) for idx in stabilizer_indices}

                # del support[meas_index][qubit]

                measurement_record.append(qubit)
                meas_index += 1

        if op.name == "DETECTOR":
            for rec in op.targets_copy():
                det_matrix[det_index, meas_index + rec.value] = 1
            det_index += 1

        if op.name == "OBSERVABLE_INCLUDE":
            for rec in op.targets_copy():
                obs_matrix[obs_index, meas_index + rec.value] = 1
            obs_index += 1

        if op.name in {"R", "MR"}:
            for t in op.targets_copy():
                qubit = t.value
                last_reset[qubit] = iop + 1

    # print("flat_circuit.num_measurements", flat_circuit.num_measurements)
    # print("meas_index", meas_index)
    # print("len(measurement_record)", len(measurement_record))

    print("support")
    print(support)
    print("det_matrix")
    print(det_matrix)
    print("obs_matrix")
    print(obs_matrix)

    bad_measurement = list(
        {meas_index for meas_index, qubits in support.items() if len(lost.intersection(qubits.keys()))}
    )

    print("bad_measurement", bad_measurement)

    A = det_matrix[:, bad_measurement].T
    idx_to_rec = -(np.arange(meas_count)[::-1] + 1)
    obs_rec = []
    for obs_idx in range(obs_count):
        b = obs_matrix[obs_idx, bad_measurement]

        x = solve_gf2(A, b)
        # print("solution x")
        # print(x)

        if x is None:
            print("No solution found for observable", obs_idx)
            continue

        for det_idx in np.where(x)[0]:
            obs_matrix[obs_idx] ^= det_matrix[det_idx]

        # print([measurement_record[i] for i, y in enumerate(obs_matrix[obs_idx]) if y])
        # print(np.where(obs_matrix[obs_idx])[0])
        obs_rec.append([stim.target_rec(idx_to_rec[i]) for i, y in enumerate(obs_matrix[obs_idx]) if y])

    final_circuit = stim.Circuit()
    obs_idx = 0
    for instr in flat_circuit:
        if instr.name == "OBSERVABLE_INCLUDE":
            final_circuit.append("OBSERVABLE_INCLUDE", obs_rec[obs_idx], instr.gate_args_copy())
            obs_idx += 1
        else:
            final_circuit.append(instr)

    return final_circuit
