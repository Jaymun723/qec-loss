#include "do_instruction.h"

#include "utils.h"

namespace qec_loss {
void do_instruction(const stim::CircuitInstruction &instr,
                    stim::Circuit &circuit,
                    std::vector<uint32_t> &lost_measurements,
                    size_t &current_measurement_index,
                    std::vector<uint8_t> &lost_qubits) {
    InstructionCategory cat = categorize_instruction(instr);

    switch (cat) {
    case InstructionCategory::ONE_QUBIT: {
        std::vector<stim::GateTarget> new_targets;
        for (const auto &t : instr.targets) {
            if (t.is_qubit_target()) {
                uint32_t q = t.qubit_value();
                if (!lost_qubits[q]) {
                    new_targets.push_back(t);
                }
            } else {
                new_targets.push_back(t);
            }
        }
        if (!new_targets.empty()) {
            circuit.safe_append(stim::CircuitInstruction(
                instr.gate_type, instr.args, new_targets, instr.tag));
        }
        break;
    }
    case InstructionCategory::TWO_QUBIT: {
        std::vector<stim::GateTarget> new_targets;
        for (size_t i = 0; i + 1 < instr.targets.size(); i += 2) {
            uint32_t q1 = instr.targets[i].qubit_value();
            uint32_t q2 = instr.targets[i + 1].qubit_value();
            if (!lost_qubits[q1] && !lost_qubits[q2]) {
                new_targets.push_back(instr.targets[i]);
                new_targets.push_back(instr.targets[i + 1]);
            }
        }
        if (!new_targets.empty()) {
            circuit.safe_append(stim::CircuitInstruction(
                instr.gate_type, instr.args, new_targets, instr.tag));
        }
        break;
    }
    case InstructionCategory::MEASURE:
    case InstructionCategory::MEASUREMENT_AND_RESET: {
        std::vector<stim::GateTarget> targets_to_depolarize;
        for (const auto &t : instr.targets) {
            if (t.is_qubit_target()) {
                uint32_t q = t.qubit_value();
                if (lost_qubits[q]) {
                    targets_to_depolarize.push_back(t);
                    lost_measurements.push_back(current_measurement_index);

                    if (cat == InstructionCategory::MEASUREMENT_AND_RESET) {
                        lost_qubits[q] = 0;
                    }
                }
            }
            current_measurement_index++;
        }
        if (!targets_to_depolarize.empty()) {
            circuit.safe_append(stim::CircuitInstruction(
                stim::GateType::DEPOLARIZE1, std::vector<double>{0.75},
                targets_to_depolarize, std::string_view{}));
        }
        circuit.safe_append(instr);
        break;
    }
    case InstructionCategory::RESET: {
        for (const auto &t : instr.targets) {
            if (t.is_qubit_target()) {
                uint32_t q = t.qubit_value();
                if (lost_qubits[q]) {
                    lost_qubits[q] = 0;
                }
            }
        }
        circuit.safe_append(instr);
        break;
    }
    default:
        circuit.safe_append(instr);
        break;
    }
}
} // namespace qec_loss