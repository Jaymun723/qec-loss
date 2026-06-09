#pragma once

#include "loss_instruction.h"
#include "stim/circuit/circuit.h"
#include "stim/circuit/circuit_instruction.h"
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace qec_loss {
/// Lightweight owning wrapper for a stim instruction or reference to
/// nominal_circuit operations
using Instruction = std::variant<LossInstruction, size_t>;

class LossyCircuit {
  public:
    stim::Circuit nominal_circuit;
    std::vector<Instruction> instructions;

    size_t num_qubits;
    size_t num_measurements;
    size_t num_detectors;
    size_t num_observables;
    size_t num_instructions;

    LossyCircuit() = delete;

    LossyCircuit(const std::string_view circuit_str);
    static LossyCircuit from_file(const std::filesystem::path &circuit_path);

    void to_file(const std::filesystem::path &circuit_path) const;

    std::string str() const;
};
}; // namespace qec_loss