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
    std::vector<uint8_t> lost_qubits(circuit.num_qubits, 0);
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
                std::get<stim::CircuitInstruction>(instr);
            do_instruction(circuit_instr, shot_circuit, lost_measurements,
                           measurement_index, lost_qubits);
            // if (has_qubit_lost) {
            //     do_instruction(circuit_instr, shot_circuit,
            //     lost_measurements,
            //                    measurement_index, lost_qubits);
            //     InstructionCategory cat =
            //     categorize_instruction(circuit_instr); if (cat ==
            //     InstructionCategory::RESET ||
            //         cat == InstructionCategory::MEASUREMENT_AND_RESET) {
            //         has_qubit_lost = false;
            //         for (const auto &p : lost_qubits) {
            //             if (p) {
            //                 has_qubit_lost = true;
            //                 break;
            //             }
            //         }
            //     }
            // } else {
            //     shot_circuit.safe_append(circuit_instr);
            //     measurement_index +=
            //     circuit_instr.count_measurement_results();
            // }
        }
    }
}

std::tuple<py::array_t<uint8_t>, std::vector<LossPattern>>
ForwardSampler::sample_measurements(size_t num_samples) {
    py::array_t<uint8_t> measurements(
        {num_samples, circuit.total_measurements_upper_bound});
    auto measurements_access = measurements.mutable_unchecked<2>();
    std::vector<LossPattern> loss_patterns;
    loss_patterns.reserve(num_samples);

    for (size_t shot_i = 0; shot_i < num_samples; shot_i++) {
        stim::Circuit shot_circuit;
        std::vector<uint32_t> lost_measurements;
        LossPattern loss_pattern;

        populate_shot_circuit(shot_circuit, lost_measurements, loss_pattern);

        loss_patterns.push_back(std::move(loss_pattern));

        // Run the shot circuit and record measurements.
        stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> sim(
            std::mt19937_64(rng), circuit.num_qubits);
        sim.safe_do_circuit(shot_circuit);

        // Write measurements to the output array, marking lost measurements
        // with 2.
        const auto &storage = sim.measurement_record.storage;
        size_t n_meas = storage.size();
        for (size_t i = 0; i < n_meas; ++i) {
            measurements_access(shot_i, i) = storage[i] ? 1 : 0;
        }
        for (size_t pos : lost_measurements) {
            measurements_access(shot_i, pos) = 2;
        }
    }
    return std::make_tuple(measurements, loss_patterns);
}

std::tuple<py::array_t<uint8_t>, py::array_t<uint8_t>, std::vector<LossPattern>>
ForwardSampler::sample_detectors(size_t num_samples) {
    return std::tuple<py::array_t<uint8_t>, py::array_t<uint8_t>,
                      std::vector<LossPattern>>();
}
std::vector<Experiment> ForwardSampler::sample_experiments(size_t num_samples) {
    return std::vector<Experiment>();
}
} // namespace qec_loss