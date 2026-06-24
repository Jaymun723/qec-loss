#pragma once

#include "../observable/reroute.h"
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
  private:
    struct ParseResult {
        stim::Circuit nominal_circuit;
        std::vector<Instruction> instructions;
    };

    static ParseResult parse_circuit(const std::string_view circuit_str);
    LossyCircuit(ParseResult parse_result);

  public:
    const stim::Circuit nominal_circuit;
    const std::vector<Instruction> instructions;

    const size_t num_qubits;
    const size_t num_measurements;
    const size_t num_detectors;
    const size_t num_observables;
    const size_t num_instructions;

    const PauliRerouter rerouter;

    LossyCircuit(const std::string_view circuit_str);
    static LossyCircuit from_file(const std::filesystem::path &circuit_path);

    void to_file(const std::filesystem::path &circuit_path) const;

    std::string str() const;

    stim::Circuit get_circuit_without_observables() const {
        stim::Circuit circuit_without_observables;
        for (const auto &op : nominal_circuit.operations) {
            if (op.gate_type != stim::GateType::OBSERVABLE_INCLUDE) {
                circuit_without_observables.safe_append(op,
                                                        /* block_fusion=*/true);
            }
        }
        return circuit_without_observables;
    }
};
}; // namespace qec_loss