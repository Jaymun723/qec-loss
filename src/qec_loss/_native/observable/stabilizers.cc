#include "stabilizers.h"
#include <algorithm>
#include <ranges>

namespace qec_loss {
std::vector<stim::PauliString<stim::MAX_BITWORD_WIDTH>>
get_stabilizers(const stim::Circuit &circuit) {
    size_t n_meas = circuit.count_measurements();
    size_t n_qubits = circuit.count_qubits();
    size_t n_ops = circuit.operations.size();

    std::vector<int> first_op_idx(n_qubits, -1);

    std::vector<stim::PauliString<stim::MAX_BITWORD_WIDTH>> active_paulis;
    active_paulis.reserve(n_meas); // reserve space for all measurements

    int meas_idx = n_meas - 1; // int because I'm scared of int overflow if I
                               // use size_t as I am decrementing it

    for (size_t i = 0; i < n_ops; i++) {
        const auto &op = circuit.operations[i];
        if (op.gate_type != stim::GateType::TICK &&
            op.gate_type != stim::GateType::SHIFT_COORDS &&
            op.gate_type != stim::GateType::QUBIT_COORDS &&
            op.gate_type != stim::GateType::DETECTOR &&
            op.gate_type != stim::GateType::OBSERVABLE_INCLUDE) {
            for (const auto &target : op.targets) {
                if (target.is_qubit_target()) {
                    uint32_t qubit = target.qubit_value();
                    if (first_op_idx[qubit] == -1) {
                        first_op_idx[qubit] = i;
                    }
                }
            }
        }
    }

    // enumrate operations in reverse
    for (int idx = n_ops; idx-- > 0;) { // same: int to prevent underflow
        const auto &op = circuit.operations[idx];
        size_t forward_idx = idx;

        if (op.gate_type == stim::GateType::M ||
            op.gate_type == stim::GateType::MX ||
            op.gate_type == stim::GateType::MY) {
            for (const auto &target : op.targets | std::views::reverse) {
                if (target.is_qubit_target()) {
                    uint32_t qubit = target.qubit_value();

                    stim::PauliString<stim::MAX_BITWORD_WIDTH> pauli(n_qubits);

                    if (op.gate_type == stim::GateType::M) {
                        pauli.zs[qubit] = 1;
                    } else if (op.gate_type == stim::GateType::MX) {
                        pauli.xs[qubit] = 1;
                    } else if (op.gate_type == stim::GateType::MY) {
                        pauli.xs[qubit] = 1;
                        pauli.zs[qubit] = 1;
                    }

                    active_paulis.push_back(std::move(pauli));
                    meas_idx--;
                }
            }
        } else if (op.gate_type == stim::GateType::MR ||
                   op.gate_type == stim::GateType::MRX ||
                   op.gate_type == stim::GateType::MRY) {
            for (auto &active_pauli : active_paulis) {
                for (const auto &target : op.targets) {
                    if (target.is_qubit_target()) {
                        uint32_t qubit = target.qubit_value();
                        if (static_cast<int>(forward_idx) >
                            first_op_idx[qubit]) {
                            active_pauli.xs[qubit] = 0;
                            active_pauli.zs[qubit] = 0;
                        }
                    }
                }
            }
            for (const auto &target : op.targets | std::views::reverse) {
                if (target.is_qubit_target()) {
                    uint32_t qubit = target.qubit_value();
                    stim::PauliString<stim::MAX_BITWORD_WIDTH> pauli(n_qubits);
                    if (op.gate_type == stim::GateType::MR) {
                        pauli.zs[qubit] = 1;
                    } else if (op.gate_type == stim::GateType::MRX) {
                        pauli.xs[qubit] = 1;
                    } else if (op.gate_type == stim::GateType::MRY) {
                        pauli.xs[qubit] = 1;
                        pauli.zs[qubit] = 1;
                    }

                    active_paulis.push_back(std::move(pauli));
                    meas_idx--;
                }
            }
        } else if (op.gate_type == stim::GateType::R ||
                   op.gate_type == stim::GateType::RX ||
                   op.gate_type == stim::GateType::RY) {
            for (auto &active_pauli : active_paulis) {
                for (const auto &target : op.targets) {
                    if (target.is_qubit_target()) {
                        uint32_t qubit = target.qubit_value();
                        if (static_cast<int>(forward_idx) >
                            first_op_idx[qubit]) {
                            active_pauli.xs[qubit] = 0;
                            active_pauli.zs[qubit] = 0;
                        }
                    }
                }
            }
        } else if (op.gate_type != stim::GateType::TICK &&
                   op.gate_type != stim::GateType::SHIFT_COORDS &&
                   op.gate_type != stim::GateType::QUBIT_COORDS &&
                   op.gate_type != stim::GateType::DETECTOR &&
                   op.gate_type != stim::GateType::OBSERVABLE_INCLUDE) {
            for (auto &active_pauli : active_paulis) {
                active_pauli.ref().undo_instruction(op);
            }
        }
    }

    std::reverse(active_paulis.begin(), active_paulis.end());

    return active_paulis;
}
} // namespace qec_loss