#include "get_loss_dem.h"
#include "../observable/reroute.h"
#include "../utils.h"
#include "stim/simulators/error_analyzer.h"
#include <map>
#include <string>
#include <vector>

namespace qec_loss {
void rewrite_instruction_for_loss(const stim::CircuitInstruction &instr,
                                  uint32_t qubit, stim::Circuit &out) {
    //   << "out before: " << out.str() << std::endl;

    // this is kind of duplicate of "do_instruction" but yabai
    InstructionCategory cat = categorize_instruction(instr);
    if (cat == InstructionCategory::ONE_QUBIT) {
        std::vector<stim::GateTarget> new_targets;
        new_targets.reserve(instr.targets.size());

        for (size_t i = 0; i < instr.targets.size(); i++) {
            if (!instr.targets[i].is_qubit_target() ||
                instr.targets[i].qubit_value() != qubit) {
                new_targets.push_back(instr.targets[i]);
            }
        }
        if (!new_targets.empty()) {
            out.safe_append(stim::CircuitInstruction(
                instr.gate_type, instr.args, new_targets, instr.tag));
        }
    } else if (cat == InstructionCategory::TWO_QUBIT) {
        std::vector<stim::GateTarget> new_targets;
        new_targets.reserve(instr.targets.size());
        for (size_t i = 0; i + 1 < instr.targets.size(); i += 2) {
            uint32_t q1 = instr.targets[i].qubit_value();
            uint32_t q2 = instr.targets[i + 1].qubit_value();
            if (q1 != qubit && q2 != qubit) {
                new_targets.push_back(instr.targets[i]);
                new_targets.push_back(instr.targets[i + 1]);
            }
        }
        if (!new_targets.empty()) {
            out.safe_append(stim::CircuitInstruction(
                instr.gate_type, instr.args, new_targets, instr.tag));
        }
    } else if (cat == InstructionCategory::MEASURE ||
               cat == InstructionCategory::MEASUREMENT_AND_RESET) {
        for (const auto &t : instr.targets) {
            if (t.is_qubit_target()) {
                uint32_t q = t.qubit_value();
                if (q == qubit) {
                    out.safe_append(stim::CircuitInstruction(
                        stim::GateType::DEPOLARIZE1, std::vector<double>{0.75},
                        std::vector<stim::GateTarget>{stim::GateTarget(q)},
                        std::string_view{}));
                }
            }
        }
        out.safe_append(instr);
    } else {
        // other gates are unaffected by loss
        out.safe_append(instr);
    }
}

stim::DetectorErrorModel
combine_circuits_into_dem(const std::vector<stim::Circuit> &circuits,
                          const std::vector<double> weights) {
    if (circuits.empty()) {
        return stim::DetectorErrorModel();
    }

    std::map<std::vector<stim::DemTarget>, std::vector<double>> global_errors;
    std::map<std::vector<stim::DemTarget>, std::string> global_tags;
    std::map<stim::DemTarget, std::pair<std::vector<double>, std::string>>
        global_detectors;
    std::map<stim::DemTarget, std::string> global_observables;

    for (size_t i = 0; i < circuits.size(); i++) {
        const auto &circuit = circuits[i];
        double weight = weights[i];

        stim::DetectorErrorModel flat_dem =
            stim::ErrorAnalyzer::circuit_to_detector_error_model(
                circuit,
                /*decompose_errors=*/true,
                /*fold_loops=*/true,
                /*allow_gauge_detectors=*/true, // <- this is main thing
                /*approximate_disjoint_errors_threshold=*/0.0,
                /*ignore_decomposition_failures=*/false,
                /*block_decomposition_from_introducing_remnant_edges=*/false)
                .flattened();

        for (const auto &op : flat_dem.instructions) {
            if (op.type == stim::DemInstructionType::DEM_ERROR) {
                std::vector<stim::DemTarget> targets(op.target_data.begin(),
                                                     op.target_data.end());
                if (!op.arg_data.empty()) {
                    global_errors[targets].push_back(op.arg_data[0] * weight);
                }
                if (!op.tag.empty()) {
                    global_tags[targets] = std::string(op.tag);
                }
            } else if (op.type == stim::DemInstructionType::DEM_DETECTOR) {
                if (!op.target_data.empty()) {
                    stim::DemTarget target = op.target_data[0];
                    std::vector<double> coords(op.arg_data.begin(),
                                               op.arg_data.end());
                    global_detectors[target] = {coords, std::string(op.tag)};
                }
            } else if (op.type ==
                       stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE) {
                if (!op.target_data.empty()) {
                    stim::DemTarget target = op.target_data[0];
                    global_observables[target] = std::string(op.tag);
                }
            }
        }
    }

    // Now construct the final DetectorErrorModel
    stim::DetectorErrorModel combined_dem;

    // Add detector declarations
    for (const auto &pair : global_detectors) {
        stim::DemTarget target = pair.first;
        const auto &coords = pair.second.first;
        const auto &tag = pair.second.second;
        combined_dem.append_detector_instruction(
            stim::SpanRef<const double>(coords.data(),
                                        coords.data() + coords.size()),
            target, tag);
    }

    // 2. Add logical observable declarations
    for (const auto &pair : global_observables) {
        stim::DemTarget target = pair.first;
        const auto &tag = pair.second;
        combined_dem.append_logical_observable_instruction(target, tag);
    }

    // 3. Add error instructions
    for (const auto &pair : global_errors) {
        const auto &targets = pair.first;
        const auto &probs = pair.second;

        double sum_prob = 0.0;
        for (double p : probs) {
            sum_prob += p;
        }

        std::string tag = "";
        auto tag_it = global_tags.find(targets);
        if (tag_it != global_tags.end()) {
            tag = tag_it->second;
        }

        combined_dem.append_error_instruction(
            sum_prob,
            stim::SpanRef<const stim::DemTarget>(
                targets.data(), targets.data() + targets.size()),
            tag);
    }

    return combined_dem;
}

stim::DetectorErrorModel get_loss_dem(const LossyCircuit &circuit,
                                      const uint32_t qubit,
                                      const LifeSegment &life_segment) {
    std::vector<stim::Circuit> result(life_segment.loss_locations.size());

    Rerouter rerouter(circuit.nominal_circuit);

    for (size_t i = 0; i < circuit.instructions.size(); i++) {
        const Instruction &lossy_inst = circuit.instructions[i];
        if (std::holds_alternative<size_t>(lossy_inst)) {
            const stim::CircuitInstruction &stim_instr =
                circuit.nominal_circuit
                    .operations[std::get<size_t>(lossy_inst)];

            bool is_pure_error = ((stim::GATE_DATA[stim_instr.gate_type].flags &
                                   stim::GATE_IS_NOISY) != 0) &&
                                 ((stim::GATE_DATA[stim_instr.gate_type].flags &
                                   stim::GATE_PRODUCES_RESULTS) == 0);
            if (is_pure_error) {
                continue;
            }

            // reroutting them through the loss
            if (stim_instr.gate_type == stim::GateType::OBSERVABLE_INCLUDE) {
                size_t obs_index = static_cast<size_t>(stim_instr.args[0]);
                for (auto &r : result) {
                    r.safe_append(stim::CircuitInstruction(
                        stim_instr.gate_type, stim_instr.args,
                        rerouter.reroute(obs_index, {qubit}, /*optimize=*/true),
                        stim_instr.tag));
                }
            } else if (i <= life_segment.start ||
                       i >= life_segment.end) { // easy skip
                size_t target_inst = std::get<size_t>(lossy_inst);
                for (auto &r : result) {
                    r.safe_append(
                        stim_instr); // safe to append directly since these
                                     // instructions are unaffected by loss
                }
            } else {
                for (size_t loss_loc_idx = 0;
                     loss_loc_idx < life_segment.loss_locations.size();
                     loss_loc_idx++) {
                    size_t loss_loc = life_segment.loss_locations[loss_loc_idx];
                    stim::Circuit &out = result[loss_loc_idx];

                    if (i <= loss_loc) {
                        out.safe_append(stim_instr);
                    } else {
                        // need to rewrite the instruction to account for the
                        // loss of the qubit
                        // std::cout << "rewriting instruction for loss at "
                        //           << loss_loc << std::endl;
                        rewrite_instruction_for_loss(stim_instr, qubit, out);
                    }
                }
            }
        }
    }

    std::vector<double> p(life_segment.loss_locations.size(), 0.0);
    // initialize p_i with the probabilities of the loss instructions at the
    // loss locations
    for (size_t loss_loc_idx = 0;
         loss_loc_idx < life_segment.loss_locations.size(); loss_loc_idx++) {
        size_t loss_loc = life_segment.loss_locations[loss_loc_idx];
        if (!std::holds_alternative<LossInstruction>(
                circuit.instructions[loss_loc])) {
            throw std::runtime_error(
                "Expected a loss instruction at loss location:" +
                std::to_string(loss_loc));
        } else {
            p[loss_loc_idx] =
                std::get<LossInstruction>(circuit.instructions[loss_loc]).p;
        }
    }

    std::vector<double> w = p;
    double tot = 0.0;
    for (size_t i = 0; i < life_segment.loss_locations.size(); i++) {
        for (size_t j = 0; j < i; j++) {
            w[i] *= (1.0 - p[j]);
        }
        tot += w[i];
    }
    for (size_t i = 0; i < life_segment.loss_locations.size(); i++) {
        w[i] /= tot;
    }

    // for (size_t i = 0; i < result.size(); i++) {
    //     std::cout << "circuit for loss at " << life_segment.loss_locations[i]
    //               << " with weight " << w[i] << ":" << std::endl;
    //     std::cout << result[i].str() << std::endl;

    //     stim::DetectorErrorModel dem =
    //         stim::ErrorAnalyzer::circuit_to_detector_error_model(
    //             result[i],
    //             /*decompose_errors=*/true,
    //             /*fold_loops=*/true,
    //             /*allow_gauge_detectors=*/true, // <- this is main thing
    //             /*approximate_disjoint_errors_threshold=*/0.0,
    //             /*ignore_decomposition_failures=*/false,
    //             /*block_decomposition_from_introducing_remnant_edges=*/false)
    //             .flattened();
    //     std::cout << "dem for loss at " << life_segment.loss_locations[i]
    //               << " with weight " << w[i] << ":" << std::endl;
    //     std::cout << dem.str() << std::endl;
    // }

    return combine_circuits_into_dem(result, w);
}
} // namespace qec_loss