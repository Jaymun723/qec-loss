#pragma once

#include "../circuit/lossy_circuit.h"
#include "../sampler/sampler.h"
#include "life_cycle.h"
#include "stim/dem/detector_error_model.h"
#include <filesystem>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <unordered_map>

namespace py = pybind11;

namespace qec_loss {
class MonakaBuilder {
  private:
    std::filesystem::path
    get_save_path(const std::filesystem::path &model_path) const;

  public:
    const LossyCircuit &circuit;
    const std::filesystem::path save_path;

    const LifeCycleManager life_cycle_manager;
    bool optimize_rerouting;

    std::string get_rerouted_observables_string(const std::vector<uint32_t> &lost_qubits) const;
    mutable std::unordered_map<std::string, stim::DetectorErrorModel> nominal_dem_cache;
    mutable std::unordered_map<std::string, stim::DetectorErrorModel> loss_dem_cache;

    MonakaBuilder() = delete;

    MonakaBuilder(const LossyCircuit &circuit,
                  const std::filesystem::path &model_path,
                  bool optimize_rerouting = false);

    stim::DetectorErrorModel
    get_nominal_dem(const std::vector<uint32_t> &lost_qubits) const;

    stim::DetectorErrorModel
    get_life_segment_dem(const std::vector<uint32_t> &lost_qubits,
                         const LifeSegment &life_segment) const;

    stim::DetectorErrorModel
    get_dem_for_shot(const std::vector<uint32_t> &lost_qubits,
                     py::array_t<uint8_t> measurements, size_t shot_i);

    py::array_t<uint8_t> decode_batch(SampleBatch &batch);
};
} // namespace qec_loss