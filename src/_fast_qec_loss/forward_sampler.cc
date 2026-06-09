#include "forward_sampler.h"
#include "do_instruction.h"
#include "stim/mem/simd_word.h"
#include "stim/simulators/tableau_simulator.h"
#include "utils.h"

#include <bitset>
#include <chrono>
#include <cstdio>

namespace qec_loss {

constexpr bool ENABLE_PROFILING = true;

ForwardSampler::ForwardSampler(const LossyCircuit &circuit,
                               std::optional<uint64_t> seed)
    : Sampler(circuit, seed.value_or(std::random_device{}())) {
    dist = std::uniform_real_distribution<double>(0.0, 1.0);

    // Precompute instruction categories and detector/observable dependencies.
    size_t current_meas_index = 0;
    nominal_instruction_categories.reserve(
        circuit.nominal_circuit.operations.size());
    for (const auto &op : circuit.nominal_circuit.operations) {
        auto cat = categorize_instruction(op);
        nominal_instruction_categories.push_back(cat);

        if (cat == InstructionCategory::MEASURE ||
            cat == InstructionCategory::MEASUREMENT_AND_RESET) {
            current_meas_index += op.count_measurement_results();
        } else if (op.gate_type == stim::GateType::DETECTOR) {
            std::vector<size_t> deps;
            deps.reserve(op.targets.size());
            for (const auto &target : op.targets) {
                if (target.is_measurement_record_target()) {
                    int idx = current_meas_index + target.value();
                    deps.push_back(idx);
                }
            }
            detector_dependencies.push_back(std::move(deps));
        } else if (op.gate_type == stim::GateType::OBSERVABLE_INCLUDE) {
            std::vector<size_t> deps;
            deps.reserve(op.targets.size());
            for (const auto &target : op.targets) {
                if (target.is_measurement_record_target()) {
                    int idx = current_meas_index + target.value();
                    deps.push_back(idx);
                }
            }
            observable_dependencies.push_back(std::move(deps));
        }
    }
}

void ForwardSampler::populate_shot_circuit(
    stim::Circuit &shot_circuit, std::vector<uint32_t> &lost_measurements,
    LossPattern &loss_patterns) {
    std::vector<uint8_t> lost_qubits(circuit.nominal_circuit.count_qubits(), 0);
    size_t measurement_index = 0;
    size_t num_lost_qubits = 0;
    size_t loss_index = 0;

    for (const auto &instr : circuit.instructions) {
        if (std::holds_alternative<LossInstruction>(instr)) {
            const auto &loss_instr = std::get<LossInstruction>(instr);

            std::vector<uint32_t> lost_qubits_in_instr;

            for (const auto target : loss_instr.targets) {
                if (lost_qubits[target] == 0 && rng() < loss_instr.threshold) {
                    lost_qubits[target] = 1;
                    num_lost_qubits++;

                    lost_qubits_in_instr.push_back(target);
                }
            }

            if (!lost_qubits_in_instr.empty()) {
                loss_patterns.emplace_back(loss_index, lost_qubits_in_instr);
            }

            loss_index++;
        } else {
            size_t nominal_idx = std::get<size_t>(instr);
            const auto &circuit_instr =
                circuit.nominal_circuit.operations[nominal_idx];
            if (num_lost_qubits > 0) {
                do_instruction(circuit_instr,
                               nominal_instruction_categories[nominal_idx],
                               shot_circuit, lost_measurements,
                               measurement_index, lost_qubits, num_lost_qubits);
            } else {
                shot_circuit.safe_append(circuit_instr);
                measurement_index += circuit_instr.count_measurement_results();
            }
        }
    }
}

SampleBatch ForwardSampler::sample(size_t num_samples) {
    double t_batch_alloc = 0.0;
    double t_populate_circuit = 0.0;
    double t_stim_sim_init = 0.0;
    double t_stim_sim_run = 0.0;
    double t_write_measurements = 0.0;
    double t_detectors_observables = 0.0;

    auto t_start = ENABLE_PROFILING
                       ? std::chrono::high_resolution_clock::now()
                       : std::chrono::high_resolution_clock::time_point{};
    SampleBatch batch(num_samples, circuit.num_measurements,
                      circuit.num_detectors, circuit.num_observables);

    auto measurements_access = batch.measurements.mutable_unchecked<2>();
    auto detectors_access = batch.detectors.mutable_unchecked<2>();
    auto observables_access = batch.observables.mutable_unchecked<2>();

    if (ENABLE_PROFILING) {
        auto t_after_alloc = std::chrono::high_resolution_clock::now();
        t_batch_alloc +=
            std::chrono::duration<double>(t_after_alloc - t_start).count();
    }

    for (size_t shot_i = 0; shot_i < num_samples; shot_i++) {
        auto t0 = ENABLE_PROFILING
                      ? std::chrono::high_resolution_clock::now()
                      : std::chrono::high_resolution_clock::time_point{};
        stim::Circuit shot_circuit;
        std::vector<uint32_t> lost_measurements;
        LossPattern loss_pattern;

        populate_shot_circuit(shot_circuit, lost_measurements, loss_pattern);

        batch.loss_patterns.push_back(std::move(loss_pattern));
        auto t1 = ENABLE_PROFILING
                      ? std::chrono::high_resolution_clock::now()
                      : std::chrono::high_resolution_clock::time_point{};
        if (ENABLE_PROFILING) {
            t_populate_circuit +=
                std::chrono::duration<double>(t1 - t0).count();
        }

        // Run the shot circuit and record measurements.
        stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> sim(
            std::mt19937_64(rng), circuit.num_qubits);
        auto t2 = ENABLE_PROFILING
                      ? std::chrono::high_resolution_clock::now()
                      : std::chrono::high_resolution_clock::time_point{};
        if (ENABLE_PROFILING) {
            t_stim_sim_init += std::chrono::duration<double>(t2 - t1).count();
        }

        sim.safe_do_circuit(shot_circuit);
        auto t3 = ENABLE_PROFILING
                      ? std::chrono::high_resolution_clock::now()
                      : std::chrono::high_resolution_clock::time_point{};
        if (ENABLE_PROFILING) {
            t_stim_sim_run += std::chrono::duration<double>(t3 - t2).count();
        }

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
        auto t4 = ENABLE_PROFILING
                      ? std::chrono::high_resolution_clock::now()
                      : std::chrono::high_resolution_clock::time_point{};
        if (ENABLE_PROFILING) {
            t_write_measurements +=
                std::chrono::duration<double>(t4 - t3).count();
        }

        for (size_t det_i = 0; det_i < circuit.num_detectors; ++det_i) {
            uint8_t det_val = 0;
            for (size_t idx : detector_dependencies[det_i]) {
                det_val ^= storage[idx] ? (uint8_t)1 : (uint8_t)0;
            }
            detectors_access(shot_i, det_i) = det_val;
        }

        for (size_t obs_i = 0; obs_i < circuit.num_observables; ++obs_i) {
            uint8_t obs_val = 0;
            for (size_t idx : observable_dependencies[obs_i]) {
                obs_val ^= storage[idx] ? (uint8_t)1 : (uint8_t)0;
            }
            observables_access(shot_i, obs_i) = obs_val;
        }
        auto t5 = ENABLE_PROFILING
                      ? std::chrono::high_resolution_clock::now()
                      : std::chrono::high_resolution_clock::time_point{};
        if (ENABLE_PROFILING) {
            t_detectors_observables +=
                std::chrono::duration<double>(t5 - t4).count();
        }
    }

    if (ENABLE_PROFILING) {
        std::printf("\n--- PROFILING RESULTS (num_samples = %zu) ---\n",
                    num_samples);
        std::printf("Batch alloc:           %9.6f s\n", t_batch_alloc);
        std::printf("Populate shot circuit: %9.6f s\n", t_populate_circuit);
        std::printf("Stim simulator init:   %9.6f s\n", t_stim_sim_init);
        std::printf("Stim simulator run:    %9.6f s\n", t_stim_sim_run);
        std::printf("Write measurements:    %9.6f s\n", t_write_measurements);
        std::printf("Detectors/observables: %9.6f s\n",
                    t_detectors_observables);
        std::printf("-------------------------------------------\n\n");
    }

    return batch;
}
} // namespace qec_loss