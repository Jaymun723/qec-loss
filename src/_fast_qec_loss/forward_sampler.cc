#include "forward_sampler.h"
#include "do_instruction.h"
#include "stim/mem/simd_word.h"
#include "stim/simulators/tableau_simulator.h"
#include "utils.h"

#include <bitset>

namespace qec_loss {

ForwardSampler::ForwardSampler(const LossyCircuit &circuit,
                               std::optional<uint64_t> seed)
    : Sampler(circuit, seed.value_or(std::random_device{}())) {
    dist = std::uniform_real_distribution<double>(0.0, 1.0);
}

void ForwardSampler::populate_shot_circuit(
    stim::Circuit &shot_circuit, std::vector<uint32_t> &lost_measurements,
    LossPattern &loss_patterns) {
    std::vector<uint8_t> lost_qubits(circuit.nominal_circuit.count_qubits(), 0);
    size_t measurement_index = 0;
    bool has_qubit_lost = false;
    size_t loss_index = 0;

    for (const auto &instr : circuit.instructions) {
        if (std::holds_alternative<LossInstruction>(instr)) {
            const auto &loss_instr = std::get<LossInstruction>(instr);

            std::vector<uint32_t> lost_qubits_in_instr;

            for (const auto target : loss_instr.targets) {
                if (dist(rng) < loss_instr.p && lost_qubits[target] == 0) {
                    lost_qubits[target] = 1;
                    has_qubit_lost = true;

                    lost_qubits_in_instr.push_back(target);
                }
            }

            if (!lost_qubits_in_instr.empty()) {
                loss_patterns.emplace_back(loss_index, lost_qubits_in_instr);
            }

            loss_index++;
        } else {
            const auto &circuit_instr =
                circuit.nominal_circuit.operations[std::get<size_t>(instr)];
            // do_instruction(circuit_instr, shot_circuit, lost_measurements,
            //                measurement_index, lost_qubits);
            if (has_qubit_lost) {
                do_instruction(circuit_instr, shot_circuit, lost_measurements,
                               measurement_index, lost_qubits);
                InstructionCategory cat = categorize_instruction(circuit_instr);
                if (cat == InstructionCategory::RESET ||
                    cat == InstructionCategory::MEASUREMENT_AND_RESET) {
                    has_qubit_lost = false;
                    for (const auto &p : lost_qubits) {
                        if (p) {
                            has_qubit_lost = true;
                            break;
                        }
                    }
                }
            } else {
                shot_circuit.safe_append(circuit_instr);
                measurement_index += circuit_instr.count_measurement_results();
            }
        }
    }
}

SampleBatch ForwardSampler::sample(size_t num_samples) {
    SampleBatch batch(num_samples, circuit.num_measurements,
                      circuit.num_detectors, circuit.num_observables);

    auto measurements_access = batch.measurements.mutable_unchecked<2>();
    auto detectors_access = batch.detectors.mutable_unchecked<2>();
    auto observables_access = batch.observables.mutable_unchecked<2>();

    for (size_t shot_i = 0; shot_i < num_samples; shot_i++) {
        stim::Circuit shot_circuit;
        std::vector<uint32_t> lost_measurements;
        LossPattern loss_pattern;

        populate_shot_circuit(shot_circuit, lost_measurements, loss_pattern);

        batch.loss_patterns.push_back(std::move(loss_pattern));

        // Run the shot circuit and record measurements.
        stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> sim(
            std::mt19937_64(rng), circuit.num_qubits);
        sim.safe_do_circuit(shot_circuit);

        // Write measurements to the output array, marking lost measurements
        // with 2.
        const auto &storage = sim.measurement_record.storage;
        size_t n_meas = storage.size();
        for (size_t i = 0; i < n_meas; ++i) {
            measurements_access(shot_i, i) =
                storage[i] ? (uint8_t)1 : (uint8_t)0;
        }
        for (size_t pos : lost_measurements) {
            measurements_access(shot_i, pos) = (uint8_t)2;
        }

        size_t current_meas_index = 0;
        size_t current_det_index = 0;
        size_t current_obs_index = 0;
        for (const auto &op : shot_circuit.operations) {
            if (op.gate_type == stim::GateType::M) {
                current_meas_index += op.targets.size();
            } else if (op.gate_type == stim::GateType::DETECTOR) {
                uint8_t det_val = 0;
                for (const auto &target : op.targets) {
                    if (target.is_measurement_record_target()) {
                        int idx = current_meas_index + target.value();
                        if (idx >= 0 && idx < (int)n_meas) {
                            det_val ^= storage[idx] ? (uint8_t)1 : (uint8_t)0;
                        }
                    }
                }
                detectors_access(shot_i, current_det_index) = det_val;
                current_det_index++;
            } else if (op.gate_type == stim::GateType::OBSERVABLE_INCLUDE) {
                uint8_t obs_val = 0;
                for (const auto &target : op.targets) {
                    if (target.is_measurement_record_target()) {
                        int idx = current_meas_index + target.value();
                        if (idx >= 0 && idx < (int)n_meas) {
                            obs_val ^= storage[idx] ? (uint8_t)1 : (uint8_t)0;
                        }
                    }
                }
                observables_access(shot_i, current_obs_index) = obs_val;
                current_obs_index++;
            }
        }
    }
    return batch;
}
} // namespace qec_loss