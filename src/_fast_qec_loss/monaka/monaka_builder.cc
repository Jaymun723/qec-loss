#include "monaka_builder.h"
#include "stim/simulators/error_analyzer.h"
#define XXH_INLINE_ALL
#include "../utils.h"
#include "get_loss_dem.h"
#include "xxhash.h"
#include <fstream>
#include <iostream>
#include <string>

#include "../observable/reroute.h"
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"
#include "pymatching/sparse_blossom/matcher/mwpm.h"
#include <algorithm>

namespace qec_loss {
std::filesystem::path
MonakaBuilder::get_save_path(const std::filesystem::path &model_path) const {
    std::string circuit_str = circuit.nominal_circuit.str();
    // hash the circuit to get a unique name for the save file
    XXH64_hash_t hash =
        XXH64(circuit_str.data(), circuit_str.size(), /* seed= */ 0);
    std::filesystem::path save_path = model_path / to_base36(hash);
    std::filesystem::create_directories(save_path);
    return save_path;
}

// stim::DetectorErrorModel
// MonakaBuilder::retrieve_loss_dem(const LifeSegment &life_segment) {
//     std::filesystem::path dem_path =
//         save_path /
//         (to_base36(std::hash<LifeSegment>{}(life_segment)) + ".dem");
//     if (std::filesystem::exists(dem_path)) {
//         std::ifstream dem_file(dem_path);
//         std::string dem_str((std::istreambuf_iterator<char>(dem_file)),
//                             std::istreambuf_iterator<char>());
//         return stim::DetectorErrorModel(dem_str);
//     } else {
//         stim::DetectorErrorModel dem =
//             get_loss_dem(circuit, life_segment, optimize_rerouting);
//         std::ofstream dem_file(dem_path);
//         dem_file << "# " << life_segment.str() << "\n";
//         dem_file << dem.str() << "\n";
//         dem_file.close();
//         std::cout << "Saved loss DEM for " << life_segment.str() << " to "
//                   << dem_path << std::endl;
//         return dem;
//     }
// }

MonakaBuilder::MonakaBuilder(const LossyCircuit &circuit,
                             const std::filesystem::path &model_path,
                             bool optimize_rerouting)
    : circuit(circuit), save_path(get_save_path(model_path)),
      life_cycle_manager(circuit), optimize_rerouting(optimize_rerouting) {}

stim::DetectorErrorModel MonakaBuilder::get_nominal_dem(
    const std::vector<uint32_t> &lost_data_qubits) const {
    stim::Circuit final_circuit;

    size_t obs_index = 0;

    for (const auto &instr : circuit.nominal_circuit.operations) {
        if (instr.gate_type == stim::GateType::OBSERVABLE_INCLUDE) {
            std::vector<stim::GateTarget> new_targets =
                circuit.rerouter.reroute(obs_index, lost_data_qubits,
                                         optimize_rerouting);
            obs_index++;
            final_circuit.safe_append(
                stim::CircuitInstruction(instr.gate_type, instr.args,
                                         new_targets, instr.tag),
                /* block_fusion=*/true);
        } else {
            final_circuit.safe_append(instr, /* block_fusion=*/true);
        }
    }

    stim::DetectorErrorModel dem =
        stim::ErrorAnalyzer::circuit_to_detector_error_model(
            final_circuit,
            /*decompose_errors=*/true,
            /*fold_loops=*/true,
            /*allow_gauge_detectors=*/false,
            /*approximate_disjoint_errors_threshold=*/0.0,
            /*ignore_decomposition_failures=*/false,
            /*block_decomposition_from_introducing_remnant_edges=*/false);
    return dem;
}

stim::DetectorErrorModel MonakaBuilder::get_life_segment_dem(
    const std::vector<uint32_t> &lost_data_qubits,
    const LifeSegment &life_segment) const {
    return get_loss_dem(circuit, lost_data_qubits, life_segment,
                        optimize_rerouting);
}

stim::DetectorErrorModel
MonakaBuilder::get_dem_for_shot(const std::vector<uint32_t> &lost_data_qubits,
                                py::array_t<uint8_t> measurements,
                                size_t shot_i) {
    stim::DetectorErrorModel final_dem(get_nominal_dem(lost_data_qubits));

    std::cout << "nominal dem:\n" << final_dem.str() << std::endl;

    auto measurements_access = measurements.unchecked<2>();
    size_t num_measurements = measurements.shape(1);
    for (size_t i = 0; i < num_measurements; i++) {
        if (measurements_access(shot_i, i) == 2) {
            LifeSegment life_segment =
                life_cycle_manager.get_life_segment_for_measurement(i);
            std::cout << "Life segment for measurement " << i << ": "
                      << life_segment.str() << std::endl;
            final_dem += get_life_segment_dem(lost_data_qubits, life_segment);
        }
    }
    return final_dem;
}

py::array_t<uint8_t> MonakaBuilder::decode_batch(SampleBatch &batch) {
    size_t num_samples = batch.measurements.shape(0);
    size_t num_measurements = batch.measurements.shape(1);
    size_t num_detectors = batch.detectors.shape(1);
    size_t num_observables = batch.observables.shape(1);

    py::array_t<uint8_t> decoded({static_cast<py::ssize_t>(num_samples),
                                  static_cast<py::ssize_t>(num_observables)});
    auto decoded_access = decoded.mutable_unchecked<2>();

    auto measurements_access = batch.measurements.unchecked<2>();
    auto detectors_access = batch.detectors.unchecked<2>();
    auto observables_access = batch.observables.unchecked<2>();

    for (size_t shot_i = 0; shot_i < num_samples; shot_i++) {
        // 1. Identify lost qubits for this shot
        std::vector<uint32_t> lost_qubits;
        for (size_t i = 0; i < num_measurements; i++) {
            if (measurements_access(shot_i, i) == 2) {
                LifeSegment life_segment =
                    life_cycle_manager.get_life_segment_for_measurement(i);
                lost_qubits.push_back(life_segment.qubit);
                auto &data_qubits = circuit.rerouter.get_data_qubits();
                bool is_data =
                    std::find(data_qubits.begin(), data_qubits.end(),
                              life_segment.qubit) != data_qubits.end();
                bool already_present =
                    std::find(lost_qubits.begin(), lost_qubits.end(),
                              life_segment.qubit) != lost_qubits.end();
                if (is_data && !already_present) {
                    lost_qubits.push_back(life_segment.qubit);
                }
            }
        }

        // Compute prediction for each observable in this shot
        for (size_t obs_i = 0; obs_i < num_observables; obs_i++) {
            // The observable is unroutable
            if (observables_access(shot_i, obs_i) == 2) {
                decoded_access(shot_i, obs_i) = 0; // <- forces a logical error
                continue;
            }

            std::cout << "Decoding shot " << shot_i << ", observable " << obs_i
                      << std::endl;

            // 4. Construct the shot-specific DEM by combining the nominal DEM
            // with the loss DEMs
            stim::DetectorErrorModel shot_dem =
                get_dem_for_shot(lost_qubits, batch.measurements, shot_i);

            // 5. Build the PyMatching decoder for this shot's DEM
            pm::Mwpm mwpm = pm::detector_error_model_to_mwpm(
                shot_dem, pm::NUM_DISTINCT_WEIGHTS);

            // 6. Gather the detection events (triggered detectors)
            std::vector<uint64_t> detection_events;
            for (size_t det_i = 0; det_i < num_detectors; det_i++) {
                if (detectors_access(shot_i, det_i) == 1) {
                    detection_events.push_back(det_i);
                }
            }

            // 7. Run PyMatching to find the correction
            pm::MatchingResult res =
                pm::decode_detection_events_for_up_to_64_observables(
                    mwpm, detection_events, /*edge_correlations=*/false);

            uint8_t correction_val = (res.obs_mask & 1);

            // 8. The predicted logical observable value is the XOR of the
            // rerouted measurement and the correction
            decoded_access(shot_i, obs_i) =
                observables_access(shot_i, obs_i) ^ correction_val;
        }
    }

    return decoded;
}

} // namespace qec_loss