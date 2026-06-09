#include "life_cycle.h"
#include "../utils.h"

namespace qec_loss {
LifeCycleManager::LifeCycleManager(const LossyCircuit &circuit)
    : circuit(circuit),
      life_cycles(circuit.num_qubits, std::vector<LifeSegment>{LifeSegment(
                                          0, circuit.num_instructions)}),
      measurement_to_life_cycle(circuit.num_measurements) {

    size_t current_measurement_index = 0;
    std::vector<uint8_t> fresh_start(circuit.num_qubits, 1);

    for (size_t circuit_index = 0; circuit_index < circuit.instructions.size();
         circuit_index++) {

        const auto &instruction = circuit.instructions[circuit_index];

        // Loss
        if (std::holds_alternative<LossInstruction>(instruction)) {
            const auto &loss_instruction =
                std::get<LossInstruction>(instruction);
            for (const auto &target : loss_instruction.targets) {
                auto &current_life_segment = life_cycles[target].back();
                current_life_segment.loss_locations.push_back(circuit_index);
                fresh_start[target] = 0;
            }
        } else {
            const auto &nominal_index = std::get<size_t>(instruction);
            const auto &instr =
                circuit.nominal_circuit.operations[nominal_index];
            InstructionCategory cat = categorize_instruction(instr);

            if (cat == InstructionCategory::MEASURE ||
                cat == InstructionCategory::MEASUREMENT_AND_RESET ||
                cat == InstructionCategory::RESET) {
                for (const auto &target : instr.targets) {
                    uint32_t q = target.qubit_value();
                    // Record that this measurement is associated with the
                    // current life segment of the target qubit
                    if (cat == InstructionCategory::MEASURE ||
                        cat == InstructionCategory::MEASUREMENT_AND_RESET) {
                        measurement_to_life_cycle[current_measurement_index] =
                            std::make_tuple(q, life_cycles[q].size() - 1);
                        current_measurement_index++;
                    }

                    // Start a new life segment if this is a reset or
                    // measurement+reset
                    if (cat == InstructionCategory::RESET ||
                        cat == InstructionCategory::MEASUREMENT_AND_RESET) {
                        if (fresh_start[q]) {
                            fresh_start[q] = 0;
                        } else {
                            life_cycles[q].back().end = circuit_index;
                            life_cycles[q].emplace_back(
                                circuit_index, circuit.num_instructions);
                        }
                    }
                }
            }
        }
    }
}

const LifeCycle &LifeCycleManager::get_life_cycle(uint32_t qubit_index) const {
    return life_cycles[qubit_index];
}

const std::tuple<uint32_t, const LifeSegment>
LifeCycleManager::get_life_segment_for_measurement(
    size_t measurement_index) const {
    const auto &[qubit_index, life_cycle_index] =
        measurement_to_life_cycle[measurement_index];

    return std::make_tuple(qubit_index,
                           life_cycles[qubit_index][life_cycle_index]);
}

const std::string LifeSegment::str() const {
    std::string result = "LifeSegment(start=" + std::to_string(start) +
                         ", end=" + std::to_string(end) + ", loss_locations=[";
    for (size_t i = 0; i < loss_locations.size(); i++) {
        result += std::to_string(loss_locations[i]);
        if (i < loss_locations.size() - 1) {
            result += ", ";
        }
    }
    result += "])";
    return result;
}

} // namespace qec_loss