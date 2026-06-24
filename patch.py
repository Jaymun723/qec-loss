import re

with open("/data/nieme-qec-loss/src/_fast_qec_loss/monaka/monaka_builder.cc", "r") as f:
    code = f.read()

# Hoist DEM generation and PyMatching out of the obs_i loop
old_code = """        // Compute prediction for each observable in this shot
        for (size_t obs_i = 0; obs_i < num_observables; obs_i++) {
            // The observable is unroutable
            // if (observables_access(shot_i, obs_i) == 2) {
            //     decoded_access(shot_i, obs_i) = 0; // <- forces a logical
            //     error continue;
            // }

            // std::cout << "Decoding shot " << shot_i << ", observable " <<
            // obs_i
            //           << std::endl;

            // 4. Construct the shot-specific DEM by combining the nominal DEM
            // with the loss DEMs
            stim::DetectorErrorModel shot_dem =
                get_dem_for_shot(lost_qubits, batch.measurements, shot_i);

            // std::cout << "Shot-specific DEM:\n" << shot_dem.str() <<
            // std::endl;

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
            // std::cout << "Running PyMatching for shot " << shot_i <<
            // std::endl;
            pm::MatchingResult res =
                pm::decode_detection_events_for_up_to_64_observables(
                    mwpm, detection_events, /*edge_correlations=*/false);

            uint8_t correction_val = (res.obs_mask & 1);

            // 8. The predicted logical observable value is the XOR of the
            // rerouted measurement and the correction
            decoded_access(shot_i, obs_i) =
                observables_access(shot_i, obs_i) ^ correction_val;
        }"""

new_code = """        // 4. Construct the shot-specific DEM by combining the nominal DEM
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

        // Compute prediction for each observable in this shot
        for (size_t obs_i = 0; obs_i < num_observables; obs_i++) {
            uint8_t correction_val = ((res.obs_mask >> obs_i) & 1);

            // 8. The predicted logical observable value is the XOR of the
            // rerouted measurement and the correction
            decoded_access(shot_i, obs_i) =
                observables_access(shot_i, obs_i) ^ correction_val;
        }"""

if old_code in code:
    code = code.replace(old_code, new_code)
    with open("/data/nieme-qec-loss/src/_fast_qec_loss/monaka/monaka_builder.cc", "w") as f:
        f.write(code)
    print("Patched monaka_builder.cc")
else:
    print("Could not find old_code to patch!")
