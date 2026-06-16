#include "reroute.h"
#include "../utils.h"

#include <algorithm>

namespace qec_loss {
Rerouter::Rerouter(const stim::Circuit &circuit)
    : circuit(circuit),
      S(circuit.count_detectors(), circuit.count_measurements()),
      L(circuit.count_observables(), circuit.count_measurements()),
      measurement_to_qubit(circuit.count_measurements(), 0) {
    for (const auto &instruction : circuit.operations) {
        parse(instruction);
        if (measurement_index == circuit.count_measurements() &&
            instruction.count_measurement_results() > 0) {
            // last measurement processed, heuristically assume that the qubits
            // measured are the data qubits
            for (const auto &t : instruction.targets) {
                if (t.is_qubit_target()) {
                    uint32_t qubit = t.qubit_value();
                    data_qubits_.push_back(qubit);
                }
            }
        }
    }
}

Rerouter::Rerouter(const stim::Circuit &circuit,
                   std::vector<uint32_t> data_qubits)
    : circuit(circuit), data_qubits_(data_qubits),
      S(circuit.count_detectors(), circuit.count_measurements()),
      L(circuit.count_observables(), circuit.count_measurements()),
      measurement_to_qubit(circuit.count_measurements(), 0) {
    for (const auto &instruction : circuit.operations) {
        parse(instruction);
    }
}

std::vector<stim::GateTarget>
Rerouter::reroute(const size_t observable_index,
                  const std::vector<size_t> &lost_qubits, bool optimize) const {
    size_t M = circuit.count_measurements();
    size_t D = circuit.count_detectors();

    std::vector<size_t> bad_measurements;
    for (size_t i = 0; i < measurement_to_qubit.size(); i++) {
        uint32_t qubit = measurement_to_qubit[i];
        bool is_lost = std::find(lost_qubits.begin(), lost_qubits.end(),
                                 qubit) != lost_qubits.end();
        bool is_data = std::find(data_qubits_.begin(), data_qubits_.end(),
                                 qubit) != data_qubits_.end();
        if (is_lost || !is_data) {
            bad_measurements.push_back(i);
        }
    }

    if (bad_measurements.empty()) {
        std::vector<stim::GateTarget> targets;
        for (size_t i = 0; i < M; i++) {
            if (L(observable_index, i)) {
                targets.push_back(stim::GateTarget::rec(-(int32_t)(M - i)));
            }
        }
        return targets;
    }

    size_t B_count = bad_measurements.size();
    PackedF2Matrix A(B_count, D);
    PackedF2Matrix b(B_count, 1);

    for (size_t k = 0; k < B_count; k++) {
        size_t bad_idx = bad_measurements[k];
        b(k, 0) = L(observable_index, bad_idx);
        for (size_t d = 0; d < D; d++) {
            A(k, d) = S(d, bad_idx);
        }
    }

    PackedF2Matrix w(D, 1);
    try {
        w = A.solve(b);
    } catch (const std::runtime_error &) {
        return {};
    }

    if (optimize) {
        PackedF2Matrix K_mat = A.kernel();
        size_t K_k = K_mat.rows();
        if (K_k > 0 && K_k <= 15) {
            PackedF2Matrix KS = K_mat * S;
            PackedF2Matrix V = KS.T();

            PackedF2Matrix ST = S.T();
            PackedF2Matrix ST_w = ST * w;

            std::vector<uint8_t> L_curr(M);
            for (size_t i = 0; i < M; i++) {
                L_curr[i] = L(observable_index, i) ^ ST_w(i, 0);
            }

            size_t num_masks = 1ULL << K_k;
            size_t best_weight = M + 1;
            size_t best_mask = 0;
            std::vector<uint8_t> candidate(M);

            for (size_t mask = 0; mask < num_masks; mask++) {
                std::copy(L_curr.begin(), L_curr.end(), candidate.begin());
                for (size_t col = 0; col < K_k; col++) {
                    if ((mask >> col) & 1) {
                        for (size_t r = 0; r < M; r++) {
                            candidate[r] ^= V(r, col);
                        }
                    }
                }
                size_t weight = 0;
                for (size_t r = 0; r < M; r++) {
                    weight += candidate[r];
                }
                if (weight < best_weight) {
                    best_weight = weight;
                    best_mask = mask;
                }
            }

            for (size_t col = 0; col < K_k; col++) {
                if ((best_mask >> col) & 1) {
                    for (size_t d = 0; d < D; d++) {
                        w(d, 0) = w(d, 0) ^ K_mat(col, d);
                    }
                }
            }
        } else {
            // warn that optimization was skipped due to large kernel size
            std::cerr
                << "Warning: Optimization skipped due to large kernel size: "
                << K_k << std::endl;
        }
    }

    std::vector<stim::GateTarget> targets;
    PackedF2Matrix ST = S.T();
    PackedF2Matrix ST_w = ST * w;

    for (size_t i = 0; i < M; i++) {
        uint8_t final_val = L(observable_index, i) ^ ST_w(i, 0);
        if (final_val) {
            targets.push_back(stim::GateTarget::rec(-(int32_t)(M - i)));
        }
    }
    return targets;
}

void Rerouter::parse(const stim::CircuitInstruction &instruction) {
    InstructionCategory category = categorize_instruction(instruction);
    if (category == InstructionCategory::MEASURE ||
        category == InstructionCategory::MEASUREMENT_AND_RESET) {
        for (const auto &t : instruction.targets) {
            if (t.is_qubit_target()) {
                measurement_to_qubit[measurement_index] = t.qubit_value();
            }
            measurement_index++;
        }
    } else if (instruction.gate_type == stim::GateType::DETECTOR) {
        for (const auto &t : instruction.targets) {
            if (t.is_measurement_record_target()) {
                int32_t m = (int32_t)measurement_index + t.value();
                if (m < 0) {
                    throw std::invalid_argument(
                        "Invalid DETECTOR instruction: measurement record "
                        "target with negative lookback.");
                }
                S(detector_index, m) = 1;
            }
        }
        detector_index++;
    } else if (instruction.gate_type == stim::GateType::OBSERVABLE_INCLUDE) {
        for (const auto &t : instruction.targets) {
            if (t.is_measurement_record_target()) {
                int32_t m = (int32_t)measurement_index + t.value();
                if (m < 0) {
                    throw std::invalid_argument(
                        "Invalid OBSERVABLE_INCLUDE instruction: measurement "
                        "record target with negative lookback.");
                }
                L(observable_index, m) = 1;
            }
        }
        observable_index++;
    }
}
} // namespace qec_loss