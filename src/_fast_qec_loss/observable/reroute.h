#pragma once

#include "../f2_tensor/packed_f2_matrix.h"
#include "stim/circuit/circuit.h"
#include "stim/stabilizers/pauli_string.h"

namespace qec_loss {
class Rerouter {
  protected:
    const stim::Circuit &circuit;

    size_t measurement_index = 0;
    size_t observable_index = 0;
    std::vector<uint32_t> measurement_to_qubit;

    virtual void parse(const stim::CircuitInstruction &instruction) = 0;

  public:
    Rerouter(const stim::Circuit &circuit)
        : circuit(circuit),
          measurement_to_qubit(circuit.count_measurements(), 0) {}

    virtual std::vector<stim::GateTarget>
    reroute(const size_t observable_index,
            const std::vector<uint32_t> &lost_qubits, bool optimize) const = 0;

    uint32_t get_qubit_for_measurement(size_t measurement_index) const {
        return measurement_to_qubit[measurement_index];
    }
};

class DetsRerouter : public Rerouter {
  private:
    PackedF2Matrix S;
    PackedF2Matrix L;
    size_t detector_index = 0;
    std::vector<uint32_t> data_qubits_;

  protected:
    void parse(const stim::CircuitInstruction &instruction) override;

  public:
    DetsRerouter(const stim::Circuit &circuit);
    DetsRerouter(const stim::Circuit &circuit,
                 const std::vector<uint32_t> &data_qubits);

    const std::vector<uint32_t> &get_data_qubits() const {
        return data_qubits_;
    };

    std::vector<stim::GateTarget>
    reroute(const size_t observable_index,
            const std::vector<uint32_t> &lost_qubits,
            bool optimize) const override;
};

class PauliRerouter : public Rerouter {
  private:
    std::vector<size_t> final_measurements_;
    std::vector<stim::PauliString<stim::MAX_BITWORD_WIDTH>> L_;
    std::vector<size_t> meas_so_far_;

    const std::vector<stim::PauliString<stim::MAX_BITWORD_WIDTH>>
        measurement_to_pauli;

    const size_t qubit_count;

  protected:
    void parse(const stim::CircuitInstruction &instruction) override;

  public:
    PauliRerouter(const stim::Circuit &circuit);

    const std::vector<stim::PauliString<stim::MAX_BITWORD_WIDTH>> &
    get_L() const {
        return L_;
    }
    const std::vector<size_t> &get_final_measurements() const {
        return final_measurements_;
    }
    const std::vector<size_t> &get_meas_so_far() const { return meas_so_far_; }

    std::pair<PackedF2Matrix, PackedF2Matrix>
    get_S_and_L_matrices(const size_t observable_index,
                         const std::vector<uint32_t> &lost_qubits,
                         bool optimize) const;

    std::vector<int>
    riroute(const PackedF2Matrix &S, const PackedF2Matrix &L,
            const std::vector<size_t> &available_measurements) const;

    std::vector<stim::GateTarget>
    reroute(const size_t observable_index,
            const std::vector<uint32_t> &lost_qubits,
            bool optimize) const override;
};

} // namespace qec_loss