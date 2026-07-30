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
    // std::map<std::vector<stim::DemTarget>, std::string> global_tags;
    // std::map<stim::DemTarget, std::pair<std::vector<double>, std::string>>
    //     global_detectors;
    // std::map<stim::DemTarget, std::string> global_observables;

    for (size_t i = 0; i < circuits.size(); i++) {
        const auto &circuit = circuits[i];
        double weight = weights[i];

        // std::cout << "Processing circuit " << i << std::endl
        //           << circuit.str() << std::endl;
        stim::DetectorErrorModel flat_dem =
            stim::ErrorAnalyzer::circuit_to_detector_error_model(
                circuit,
                /*decompose_errors=*/false,
                /*fold_loops=*/true,
                /*allow_gauge_detectors=*/true, // <- this is main thing
                /*approximate_disjoint_errors_threshold=*/0.0,
                /*ignore_decomposition_failures=*/true,
                /*block_decomposition_from_introducing_remnant_edges=*/false)
                .flattened();

        for (const auto &op : flat_dem.instructions) {
            if (op.type == stim::DemInstructionType::DEM_ERROR) {
                std::vector<stim::DemTarget> targets(op.target_data.begin(),
                                                     op.target_data.end());
                if (!op.arg_data.empty()) {
                    global_errors[targets].push_back(op.arg_data[0] * weight);
                }
                // if (!op.tag.empty()) {
                //     global_tags[targets] = std::string(op.tag);
                // }
            }
            // } else if (op.type == stim::DemInstructionType::DEM_DETECTOR) {
            //     if (!op.target_data.empty()) {
            //         stim::DemTarget target = op.target_data[0];
            //         std::vector<double> coords(op.arg_data.begin(),
            //                                    op.arg_data.end());
            //         global_detectors[target] = {coords, std::string(op.tag)};
            //     }
            // } else if (op.type ==
            //            stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE) {
            //     if (!op.target_data.empty()) {
            //         stim::DemTarget target = op.target_data[0];
            //         global_observables[target] = std::string(op.tag);
            //     }
            // }
        }
    }

    // Now construct the final DetectorErrorModel
    stim::DetectorErrorModel combined_dem;

    // Add detector declarations
    // for (const auto &pair : global_detectors) {
    //     stim::DemTarget target = pair.first;
    //     const auto &coords = pair.second.first;
    //     const auto &tag = pair.second.second;
    //     combined_dem.append_detector_instruction(
    //         stim::SpanRef<const double>(coords.data(),
    //                                     coords.data() + coords.size()),
    //         target, tag);
    // }

    // 2. Add logical observable declarations
    // for (const auto &pair : global_observables) {
    //     stim::DemTarget target = pair.first;
    //     const auto &tag = pair.second;
    //     combined_dem.append_logical_observable_instruction(target, tag);
    // }

    // 3. Add error instructions
    for (const auto &pair : global_errors) {
        const auto &targets = pair.first;
        const auto &probs = pair.second;

        double sum_prob = 0.0;
        for (double p : probs) {
            sum_prob += p;
        }

        // std::string tag = "";
        // auto tag_it = global_tags.find(targets);
        // if (tag_it != global_tags.end()) {
        //     tag = tag_it->second;
        // }

        combined_dem.append_error_instruction(
            sum_prob,
            stim::SpanRef<const stim::DemTarget>(
                targets.data(), targets.data() + targets.size()),
            "");
    }

    return combined_dem;
}

std::vector<stim::Circuit> get_loss_rewritten_circuits(
    const LossyCircuit &circuit, const std::vector<uint32_t> &lost_qubits,
    const LifeSegment &life_segment, bool optimize_rerouting) {
    const uint32_t qubit = life_segment.qubit;
    std::vector<stim::Circuit> result(life_segment.loss_locations.size());

    size_t obs_index = 0;

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
                std::vector<stim::GateTarget> new_targets(
                    circuit.rerouter.reroute(obs_index, lost_qubits,
                                             optimize_rerouting));
                for (auto &r : result) {
                    r.safe_append(stim::CircuitInstruction(
                        stim_instr.gate_type, stim_instr.args, new_targets,
                        stim_instr.tag));
                }

                obs_index++;
            } else {
                for (size_t loss_loc_idx = 0;
                     loss_loc_idx < life_segment.loss_locations.size();
                     loss_loc_idx++) {
                    size_t loss_loc = life_segment.loss_locations[loss_loc_idx];
                    stim::Circuit &out = result[loss_loc_idx];

                    if (i <= loss_loc || i > life_segment.end) {
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

    return result;
}

std::vector<stim::Circuit>
get_loss_rewritten_circuits(const LossyCircuit &circuit,
                            const LifeSegment &life_segment) {
    const uint32_t qubit = life_segment.qubit;
    std::vector<stim::Circuit> result(life_segment.loss_locations.size());

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

            for (size_t loss_loc_idx = 0;
                 loss_loc_idx < life_segment.loss_locations.size();
                 loss_loc_idx++) {
                size_t loss_loc = life_segment.loss_locations[loss_loc_idx];
                stim::Circuit &out = result[loss_loc_idx];

                if (i <= loss_loc || i > life_segment.end) {
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

    return result;
}

namespace {
/// Posterior weights over candidate loss locations of a life segment:
/// w_i = p_i * prod_{j < i} (1 - p_j), normalized. Matches the Python
/// reference (`weights[i] = q * (1 - q)**i` for uniform loss probability q).
std::vector<double>
loss_location_weights(const LossyCircuit &circuit,
                      const LifeSegment &life_segment) {
    std::vector<double> p(life_segment.loss_locations.size(), 0.0);
    for (size_t loss_loc_idx = 0;
         loss_loc_idx < life_segment.loss_locations.size(); loss_loc_idx++) {
        size_t loss_loc = life_segment.loss_locations[loss_loc_idx];
        if (!std::holds_alternative<LossInstruction>(
                circuit.instructions[loss_loc])) {
            throw std::runtime_error(
                "Expected a loss instruction at loss location:" +
                std::to_string(loss_loc));
        }
        p[loss_loc_idx] =
            std::get<LossInstruction>(circuit.instructions[loss_loc]).p;
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
    return w;
}
} // namespace

stim::DetectorErrorModel get_loss_dem(const LossyCircuit &circuit,
                                      const std::vector<uint32_t> &lost_qubits,
                                      const LifeSegment &life_segment,
                                      bool optimize_rerouting) {
    std::vector<stim::Circuit> result = get_loss_rewritten_circuits(
        circuit, lost_qubits, life_segment, optimize_rerouting);

    std::vector<double> w = loss_location_weights(circuit, life_segment);

    return combine_circuits_into_dem(result, w);
}

stim::DetectorErrorModel
combine_dems_native(const std::vector<stim::DetectorErrorModel> &dems,
                    const std::vector<double> &weights) {
    std::map<std::vector<stim::DemTarget>, double> global_errors;

    for (size_t i = 0; i < dems.size(); i++) {
        double weight = weights[i];
        for (const auto &op : dems[i].instructions) {
            if (op.type == stim::DemInstructionType::DEM_ERROR &&
                !op.arg_data.empty()) {
                std::vector<stim::DemTarget> targets(op.target_data.begin(),
                                                     op.target_data.end());
                global_errors[targets] += op.arg_data[0] * weight;
            }
        }
    }

    stim::DetectorErrorModel combined_dem;
    for (const auto &pair : global_errors) {
        const auto &targets = pair.first;
        combined_dem.append_error_instruction(
            pair.second,
            stim::SpanRef<const stim::DemTarget>(targets.data(),
                                                 targets.data() + targets.size()),
            "");
    }
    return combined_dem;
}

stim::DetectorErrorModel
circuit_to_dem_gauge_observables(const stim::Circuit &circuit) {
    uint64_t num_dets = circuit.count_detectors();
    stim::Circuit flat = circuit.flattened();

    // Map OBSERVABLE_INCLUDE instructions onto temporary detectors so that
    // undeterministic observables are tolerated by allow_gauge_detectors.
    stim::Circuit new_circuit;
    std::vector<int64_t> dets_to_obs;
    std::vector<int64_t> obs_to_dets(circuit.count_observables(), -1);
    uint64_t dets = num_dets;

    for (const auto &op : flat.operations) {
        if (op.gate_type == stim::GateType::OBSERVABLE_INCLUDE) {
            if (op.args.size() != 1) {
                throw std::invalid_argument(
                    "OBSERVABLE_INCLUDE should have exactly one argument.");
            }
            int64_t obs_idx = static_cast<int64_t>(op.args[0]);
            if (obs_to_dets[obs_idx] != -1) {
                throw std::invalid_argument(
                    "OBSERVABLE_INCLUDE should not be called multiple times "
                    "for the same observable.");
            }
            obs_to_dets[obs_idx] = static_cast<int64_t>(dets);
            dets_to_obs.push_back(obs_idx);
            dets++;
            new_circuit.safe_append(stim::CircuitInstruction(
                stim::GateType::DETECTOR, {}, op.targets, op.tag));
        } else {
            new_circuit.safe_append(op);
        }
    }

    stim::DetectorErrorModel dem =
        stim::ErrorAnalyzer::circuit_to_detector_error_model(
            new_circuit,
            /*decompose_errors=*/false,
            /*fold_loops=*/true,
            /*allow_gauge_detectors=*/true,
            /*approximate_disjoint_errors_threshold=*/0.0,
            /*ignore_decomposition_failures=*/true,
            /*block_decomposition_from_introducing_remnant_edges=*/false)
            .flattened();

    // Map the temporary detectors back to logical observables.
    stim::DetectorErrorModel new_dem;
    for (const auto &instruction : dem.instructions) {
        if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
            std::vector<stim::DemTarget> new_targets;
            new_targets.reserve(instruction.target_data.size());
            for (const auto &target : instruction.target_data) {
                stim::DemTarget new_target = target;
                if (target.is_relative_detector_id() &&
                    target.val() >= num_dets) {
                    int64_t obs_idx = dets_to_obs[target.val() - num_dets];
                    if (obs_idx == -1) {
                        throw std::runtime_error(
                            "Target " + std::to_string(target.val()) +
                            " does not correspond to any observable.");
                    }
                    new_target =
                        stim::DemTarget::observable_id((uint64_t)obs_idx);
                }
                new_targets.push_back(new_target);
            }
            new_dem.append_error_instruction(
                instruction.arg_data.empty() ? 0.0 : instruction.arg_data[0],
                stim::SpanRef<const stim::DemTarget>(
                    new_targets.data(), new_targets.data() + new_targets.size()),
                instruction.tag);
        } else if (instruction.type == stim::DemInstructionType::DEM_DETECTOR) {
            if (instruction.target_data.size() != 1) {
                throw std::runtime_error(
                    "Detector instructions should have exactly one target.");
            }
            stim::DemTarget target = instruction.target_data[0];
            if (target.is_relative_detector_id() && target.val() >= num_dets) {
                int64_t obs_idx = dets_to_obs[target.val() - num_dets];
                if (obs_idx == -1) {
                    throw std::runtime_error(
                        "Detector target " + std::to_string(target.val()) +
                        " does not correspond to any observable.");
                }
                new_dem.append_logical_observable_instruction(
                    stim::DemTarget::observable_id((uint64_t)obs_idx),
                    instruction.tag);
            } else {
                new_dem.append_detector_instruction(instruction.arg_data,
                                                    target, instruction.tag);
            }
        } else if (instruction.type ==
                       stim::DemInstructionType::DEM_SHIFT_DETECTORS ||
                   instruction.type ==
                       stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE) {
            new_dem.append_dem_instruction(instruction);
        } else {
            throw std::runtime_error(
                "Unexpected DEM instruction type after flattening: " +
                std::to_string((int)instruction.type));
        }
    }
    return new_dem;
}

stim::DetectorErrorModel
get_loss_segment_dem(const LossyCircuit &circuit,
                     const LifeSegment &life_segment) {
    std::vector<stim::Circuit> rewritten =
        get_loss_rewritten_circuits(circuit, life_segment);

    std::vector<stim::DetectorErrorModel> dems;
    dems.reserve(rewritten.size());
    for (const auto &c : rewritten) {
        dems.push_back(circuit_to_dem_gauge_observables(c));
    }

    std::vector<double> w = loss_location_weights(circuit, life_segment);

    return combine_dems_native(dems, w);
}

std::vector<stim::DetectorErrorModel>
get_loss_segment_dems(const LossyCircuit &circuit,
                      const std::vector<LifeSegment> &life_segments) {
    std::vector<stim::DetectorErrorModel> dems;
    dems.reserve(life_segments.size());
    for (const auto &life_segment : life_segments) {
        dems.push_back(get_loss_segment_dem(circuit, life_segment));
    }
    return dems;
}
} // namespace qec_loss