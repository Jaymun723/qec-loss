#include "reroute.h"
#include "../utils.h"
#include "./stabilizers.h"

#include <algorithm>
#include <unordered_set>

namespace qec_loss {

void DetsRerouter::parse_instruction(
    const stim::CircuitInstruction &instruction, PackedF2Matrix &S,
    PackedF2Matrix &L, std::vector<uint32_t> &measurement_to_qubit,
    size_t &measurement_index, size_t &detector_index,
    size_t &observable_index) {
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

DetsRerouter::DetsRerouter(stim::Circuit circuit)
    : Rerouter(circuit.count_measurements()),
      S(circuit.count_detectors(), circuit.count_measurements()),
      L(circuit.count_observables(), circuit.count_measurements()),
      num_measurements_(circuit.count_measurements()),
      num_detectors_(circuit.count_detectors()) {
    size_t measurement_index = 0;
    size_t detector_index = 0;
    size_t observable_index = 0;
    for (const auto &op : circuit.operations) {
        parse_instruction(op, S, L, measurement_to_qubit, measurement_index,
                          detector_index, observable_index);
        size_t delta_measurements = op.count_measurement_results();
        if (delta_measurements > 0 && measurement_index == num_measurements_) {
            // last measurement processed, heuristically assume that the
            // qubits measured are the data qubits
            for (const auto &t : op.targets) {
                if (t.is_qubit_target()) {
                    data_qubits_.push_back(t.qubit_value());
                }
            }
        }
    }
}

DetsRerouter::DetsRerouter(stim::Circuit circuit,
                           const std::vector<uint32_t> &data_qubits)
    : Rerouter(circuit.count_measurements()),
      S(circuit.count_detectors(), circuit.count_measurements()),
      L(circuit.count_observables(), circuit.count_measurements()),
      data_qubits_(data_qubits),
      num_measurements_(circuit.count_measurements()),
      num_detectors_(circuit.count_detectors()) {
    size_t measurement_index = 0;
    size_t detector_index = 0;
    size_t observable_index = 0;
    for (const auto &instruction : circuit.operations) {
        parse_instruction(instruction, S, L, measurement_to_qubit,
                          measurement_index, detector_index, observable_index);
    }
}

std::vector<stim::GateTarget>
DetsRerouter::reroute(const size_t observable_index,
                      const std::vector<uint32_t> &lost_qubits,
                      bool optimize) const {
    const size_t M = num_measurements_;
    const size_t D = num_detectors_;

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

        if (!is_data && is_lost) {
            for (size_t d = 0; d < D; d++) {
                bool detectord_affected = false;
                for (size_t m = 0; m < M; m++) {
                    if (S(d, m)) {
                        uint32_t qubit_other = measurement_to_qubit[m];

                        if (qubit_other == qubit) {
                            detectord_affected = true;
                            break;
                        }
                    }
                }
                if (detectord_affected) {
                    for (size_t m = 0; m < M; m++) {
                        if (S(d, m)) {
                            uint32_t qubit_other = measurement_to_qubit[m];

                            bool is_other_data =
                                std::find(data_qubits_.begin(),
                                          data_qubits_.end(),
                                          qubit_other) != data_qubits_.end();

                            if (is_other_data) {
                                bad_measurements.push_back(m);
                            }
                        }
                    }
                }
            }
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

void PauliRerouter::parse(const stim::CircuitInstruction &instr,
                          size_t &measurement_index, size_t &observable_index) {
    bool does_measure = instr.gate_type == stim::GateType::M ||
                        instr.gate_type == stim::GateType::MX ||
                        instr.gate_type == stim::GateType::MY ||
                        instr.gate_type == stim::GateType::MR ||
                        instr.gate_type == stim::GateType::MRX ||
                        instr.gate_type == stim::GateType::MRY;
    bool does_reset = instr.gate_type == stim::GateType::MR ||
                      instr.gate_type == stim::GateType::MRX ||
                      instr.gate_type == stim::GateType::MRY ||
                      instr.gate_type == stim::GateType::R ||
                      instr.gate_type == stim::GateType::RX ||
                      instr.gate_type == stim::GateType::RY;
    bool is_annotation =
        instr.gate_type == stim::GateType::DETECTOR ||
        instr.gate_type == stim::GateType::OBSERVABLE_INCLUDE ||
        instr.gate_type == stim::GateType::TICK ||
        instr.gate_type == stim::GateType::QUBIT_COORDS ||
        instr.gate_type == stim::GateType::SHIFT_COORDS;

    if (does_measure) {
        for (const auto &t : instr.targets) {
            if (t.is_qubit_target()) {
                measurement_to_qubit[measurement_index] = t.qubit_value();
                if (!does_reset) {
                    final_measurements_.push_back(measurement_index);
                }
            }
            measurement_index++;
        }

    } else if (!is_annotation || does_reset) {
        final_measurements_.clear();
    }

    if (instr.gate_type == stim::GateType::OBSERVABLE_INCLUDE) {
        if (instr.args.size() == 1) {
            size_t idx = (size_t)instr.args[0];
            if (idx != observable_index) {
                std::cerr
                    << "Warning: OBSERVABLE_INCLUDE instruction has index "
                    << idx << " but expected index is " << observable_index
                    << std::endl;
            }
        }

        stim::PauliString<stim::MAX_BITWORD_WIDTH> obs(qubit_count);

        for (const auto &t : instr.targets) {
            if (t.is_measurement_record_target()) {
                int32_t m = static_cast<int32_t>(measurement_index) + t.value();
                if (m < 0) {
                    throw std::invalid_argument(
                        "Invalid OBSERVABLE_INCLUDE instruction: measurement "
                        "record target with negative lookback.");
                }

                obs.ref().inplace_right_mul_returning_log_i_scalar(
                    measurement_to_pauli[m].ref());
            }
        }

        L_.push_back(obs);
        meas_so_far_[observable_index] = measurement_index;
        observable_index++;
    }
}

PauliRerouter::PauliRerouter(stim::Circuit circuit)
    : Rerouter(circuit.count_measurements()),
      measurement_to_pauli(get_stabilizers(circuit)),
      meas_so_far_(circuit.count_observables(), 0),
      qubit_count(circuit.count_qubits()) {
    size_t measurement_index = 0;
    size_t observable_index = 0;
    for (const auto &instruction : circuit.operations) {
        parse(instruction, measurement_index, observable_index);
    }
}

std::pair<PackedF2Matrix, PackedF2Matrix>
PauliRerouter::get_S_and_L_matrices(const size_t observable_index,
                                    const std::vector<uint32_t> &lost_qubits,
                                    bool optimize) const {
    if (observable_index >= L_.size()) {
        throw std::invalid_argument(
            "Invalid observable index: " + std::to_string(observable_index) +
            " >= " + std::to_string(L_.size()));
    }

    size_t M = meas_so_far_[observable_index];

    std::unordered_set<size_t> bad_measurements;
    for (size_t meas_idx = 0; meas_idx < M; meas_idx++) {

        const auto &pauli = measurement_to_pauli[meas_idx];

        for (const uint32_t lost_qubit : lost_qubits) {
            if (pauli.xs[lost_qubit] != 0 || pauli.zs[lost_qubit] != 0) {
                bad_measurements.insert(meas_idx);
                break;
            }
        }
    }

    std::vector<size_t> available_measurements;
    for (size_t meas_idx = 0; meas_idx < meas_so_far_[observable_index];
         meas_idx++) {
        bool is_bad = bad_measurements.find(meas_idx) != bad_measurements.end();
        bool is_final =
            std::find(final_measurements_.begin(), final_measurements_.end(),
                      meas_idx) != final_measurements_.end();
        if (!is_bad || !is_final) {
            available_measurements.push_back(meas_idx);
        }
    }

    size_t k = available_measurements.size();

    PackedF2Matrix S(2 * qubit_count, k);
    for (size_t col = 0; col < k; col++) {
        size_t meas_idx = available_measurements[col];
        const auto &pauli = measurement_to_pauli[meas_idx];
        for (size_t q = 0; q < qubit_count; q++) {
            S(q, col) = pauli.xs[q] ? 1 : 0;
            S(qubit_count + q, col) = pauli.zs[q] ? 1 : 0;
        }
    }

    PackedF2Matrix L(2 * qubit_count, 1);
    const auto &obs = L_[observable_index];
    for (size_t q = 0; q < qubit_count; q++) {
        L(q, 0) = obs.xs[q] ? 1 : 0;
        L(qubit_count + q, 0) = obs.zs[q] ? 1 : 0;
    }

    return std::make_pair(S, L);
}

std::vector<int> PauliRerouter::riroute(
    const PackedF2Matrix &S, const PackedF2Matrix &L,
    const std::vector<size_t> &available_measurements) const {
    size_t k = available_measurements.size();

    PackedF2Matrix w(k, 1);
    try {
        w = S.solve(L);
    } catch (const std::runtime_error &) {
        return {};
    }

    for (size_t i = 0; i < k; i++) {
        if (w(i, 0)) {
        }
    }

    std::vector<int> targets;
    return targets;
}

std::vector<stim::GateTarget>
PauliRerouter::reroute(const size_t observable_index,
                       const std::vector<uint32_t> &lost_qubits,
                       bool optimize) const {
    if (observable_index >= L_.size()) {
        throw std::invalid_argument(
            "Invalid observable index: " + std::to_string(observable_index) +
            " >= " + std::to_string(L_.size()));
    }

    size_t M = meas_so_far_[observable_index];

    std::unordered_set<size_t> bad_measurements;
    for (size_t meas_idx = 0; meas_idx < M; meas_idx++) {

        const auto &pauli = measurement_to_pauli[meas_idx];

        for (const uint32_t lost_qubit : lost_qubits) {
            if (pauli.xs[lost_qubit] != 0 || pauli.zs[lost_qubit] != 0) {
                bad_measurements.insert(meas_idx);
                break;
            }
        }
    }

    std::vector<size_t> available_measurements;
    for (size_t meas_idx = 0; meas_idx < meas_so_far_[observable_index];
         meas_idx++) {
        bool is_bad = bad_measurements.find(meas_idx) != bad_measurements.end();
        bool is_final =
            std::find(final_measurements_.begin(), final_measurements_.end(),
                      meas_idx) != final_measurements_.end();
        if (!is_bad || !is_final) {
            available_measurements.push_back(meas_idx);
        }
    }

    size_t k = available_measurements.size();

    PackedF2Matrix S(2 * qubit_count, k);
    for (size_t col = 0; col < k; col++) {
        size_t meas_idx = available_measurements[col];
        const auto &pauli = measurement_to_pauli[meas_idx];
        for (size_t q = 0; q < qubit_count; q++) {
            S(q, col) = pauli.xs[q] ? 1 : 0;
            S(qubit_count + q, col) = pauli.zs[q] ? 1 : 0;
        }
    }

    PackedF2Matrix L(2 * qubit_count, 1);
    const auto &obs = L_[observable_index];
    for (size_t q = 0; q < qubit_count; q++) {
        L(q, 0) = obs.xs[q] ? 1 : 0;
        L(qubit_count + q, 0) = obs.zs[q] ? 1 : 0;
    }

    PackedF2Matrix w(k, 1);
    try {
        w = S.solve(L);
    } catch (const std::runtime_error &) {
        return {};
    }

    std::vector<stim::GateTarget> targets;

    for (size_t i = 0; i < k; i++) {
        if (w(i, 0)) {
            size_t meas_idx = available_measurements[i];
            bool is_final = std::find(final_measurements_.begin(),
                                      final_measurements_.end(),
                                      meas_idx) != final_measurements_.end();
            if (is_final) {
                targets.push_back(stim::GateTarget::rec(-(
                    static_cast<int32_t>(M) - static_cast<int32_t>(meas_idx))));
            }
        }
    }

    return targets;
}

} // namespace qec_loss
