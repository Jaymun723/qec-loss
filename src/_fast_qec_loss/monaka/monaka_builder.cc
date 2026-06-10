#include "monaka_builder.h"
#include "stim/simulators/error_analyzer.h"
#define XXH_INLINE_ALL
#include "xxhash.h"
#include <fstream>
#include <string>

namespace qec_loss {
std::filesystem::path
MonakaBuilder::get_save_path(const std::filesystem::path &model_path) const {
    std::string circuit_str = circuit.nominal_circuit.str();
    // hash the circuit to get a unique name for the save file
    XXH64_hash_t hash =
        XXH64(circuit_str.data(), circuit_str.size(), 0 /* seed */);
    std::filesystem::path save_path = model_path / std::to_string(hash);
    std::filesystem::create_directories(save_path);
    return save_path;
}

MonakaBuilder::MonakaBuilder(const LossyCircuit &circuit,
                             const std::filesystem::path &model_path)
    : circuit(circuit), save_path(get_save_path(model_path)) {}

stim::DetectorErrorModel MonakaBuilder::get_nominal_dem() const {
    if (nominal_dem_cache.has_value()) {
        return nominal_dem_cache.value();
    }
    if (std::filesystem::exists(save_path / "nominal.dem")) {
        std::ifstream dem_file(save_path / "nominal.dem");
        std::string dem_str((std::istreambuf_iterator<char>(dem_file)),
                            std::istreambuf_iterator<char>());
        const stim::DetectorErrorModel dem(dem_str);
        nominal_dem_cache.emplace(dem);
        return dem;
    } else {
        stim::DetectorErrorModel dem =
            stim::ErrorAnalyzer::circuit_to_detector_error_model(
                circuit.nominal_circuit,
                /*decompose_errors=*/true,
                /*fold_loops=*/true,
                /*allow_gauge_detectors=*/false,
                /*approximate_disjoint_errors_threshold=*/0.0,
                /*ignore_decomposition_failures=*/false,
                /*block_decomposition_from_introducing_remnant_edges=*/false);
        std::ofstream dem_file(save_path / "nominal.dem");
        dem_file << dem.str() << "\n";
        nominal_dem_cache.emplace(dem);
        return dem;
    }
}

void MonakaBuilder::prebuild() {
    // first build the nominal DEM to populate the cache and save it to disk
    get_nominal_dem();
}

py::array_t<uint8_t> MonakaBuilder::decode_batch(SampleBatch &batch) const {
    return py::array_t<uint8_t>();
}
} // namespace qec_loss