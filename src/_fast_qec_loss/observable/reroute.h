#pragma once

#include "../f2_tensor/packed_f2_matrix.h"
#include "stim/circuit/circuit.h"

namespace qec_loss {
class Rerouter {
  private:
    stim::Circuit circuit;
    PackedF2Matrix S;
    PackedF2Matrix L;
    std::vector<uint32_t> data_qubits_;

    size_t measurement_index = 0;
    size_t detector_index = 0;
    size_t observable_index = 0;
    std::vector<uint32_t> measurement_to_qubit;

    void parse(const stim::CircuitInstruction &instruction);

  public:
    Rerouter(const stim::Circuit &circuit);
    Rerouter(const stim::Circuit &circuit, std::vector<uint32_t> data_qubits);

    std::vector<stim::GateTarget>
    reroute(const size_t observable_index,
            const std::vector<size_t> &lost_qubits, bool optimize) const;

    const std::vector<uint32_t> &get_data_qubits() const {
        return data_qubits_;
    };
};

} // namespace qec_loss