#pragma once

#include "loss_instruction.h"
#include "stim/circuit/circuit.h"
#include "stim/circuit/circuit_instruction.h"
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace qec_loss {
/// Lightweight owning wrapper for a stim instruction
using Instruction = std::variant<LossInstruction, stim::CircuitInstruction>;

class LossyCircuit {
  private:
    size_t total_measurements_upper_bound = 0;

  public:
    stim::Circuit nominal_circuit;
    size_t num_qubits = 0;
    std::vector<Instruction> instructions;

    LossyCircuit() = delete;

    LossyCircuit(const std::string_view circuit_str);
    static LossyCircuit from_file(const std::filesystem::path &circuit_path);

    void to_file(const std::filesystem::path &circuit_path) const;

    std::string str() const;
};
}; // namespace qec_loss