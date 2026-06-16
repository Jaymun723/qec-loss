#pragma once

#include "../circuit/lossy_circuit.h"
#include "../sampler/sampler.h"
#include "stim/dem/detector_error_model.h"
#include <filesystem>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace qec_loss {
class MonakaBuilder {
  private:
    std::filesystem::path
    get_save_path(const std::filesystem::path &model_path) const;

    mutable std::optional<stim::DetectorErrorModel> nominal_dem_cache;

  public:
    const LossyCircuit &circuit;
    const std::filesystem::path save_path;

    MonakaBuilder() = delete;

    MonakaBuilder(const LossyCircuit &circuit,
                  const std::filesystem::path &model_path);

    stim::DetectorErrorModel get_nominal_dem() const;
    void prebuild();

    py::array_t<uint8_t> decode_batch(SampleBatch &batch) const;
};
} // namespace qec_loss