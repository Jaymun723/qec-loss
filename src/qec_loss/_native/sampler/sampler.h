#pragma once

#include "../circuit/lossy_circuit.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <ranges>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace qec_loss {

using LossPattern = std::vector<std::tuple<size_t, std::vector<uint32_t>>>;

inline std::vector<LossPattern> convert_loss_patterns(
    const std::vector<std::unordered_map<size_t, std::vector<uint32_t>>>
        &loss_patterns) {
    std::vector<LossPattern> converted;
    converted.reserve(loss_patterns.size());
    for (const auto &pattern : loss_patterns) {
        LossPattern converted_pattern;
        converted_pattern.reserve(pattern.size());
        for (const auto &[key, value] : pattern) {
            converted_pattern.emplace_back(key, value);
        }
        converted.push_back(std::move(converted_pattern));
    }
    return converted;
}

inline std::vector<std::unordered_map<size_t, std::vector<uint32_t>>>
convert_loss_patterns_back(const std::vector<LossPattern> &loss_patterns) {
    std::vector<std::unordered_map<size_t, std::vector<uint32_t>>> converted;
    converted.reserve(loss_patterns.size());
    for (const auto &pattern : loss_patterns) {
        std::unordered_map<size_t, std::vector<uint32_t>> converted_pattern;
        for (const auto &[key, value] : pattern) {
            converted_pattern[key] = value;
        }
        converted.push_back(std::move(converted_pattern));
    }
    return converted;
}

struct SampleBatch {
    py::array_t<uint8_t> measurements;
    py::array_t<uint8_t> detectors;
    py::array_t<uint8_t> observables;
    std::vector<LossPattern> loss_patterns;

    SampleBatch(size_t num_samples, size_t num_measurements,
                size_t num_detectors, size_t num_observables)
        : measurements({num_samples, num_measurements}),
          detectors({num_samples, num_detectors}),
          observables({num_samples, num_observables}) {
        loss_patterns.reserve(num_samples);
    };

    SampleBatch(py::array_t<uint8_t> measurements,
                py::array_t<uint8_t> detectors,
                py::array_t<uint8_t> observables,
                std::vector<LossPattern> loss_patterns)
        : measurements(std::move(measurements)),
          detectors(std::move(detectors)), observables(std::move(observables)),
          loss_patterns(std::move(loss_patterns)) {
        if (this->measurements.ndim() != 2 || this->detectors.ndim() != 2 ||
            this->observables.ndim() != 2) {
            throw std::invalid_argument(
                "Measurements, detectors, and observables must be 2D arrays.");
        }
        if (this->measurements.shape(0) != this->detectors.shape(0) ||
            this->measurements.shape(0) != this->observables.shape(0)) {
            throw std::invalid_argument(
                "Measurements, detectors, and observables must have the same "
                "number of samples.");
        }
    }

    SampleBatch(py::array_t<uint8_t> measurements,
                py::array_t<uint8_t> detectors,
                py::array_t<uint8_t> observables,
                std::vector<std::unordered_map<size_t, std::vector<uint32_t>>>
                    loss_patterns)
        : measurements(std::move(measurements)),
          detectors(std::move(detectors)), observables(std::move(observables)),
          loss_patterns(convert_loss_patterns(loss_patterns)) {}
};

class Sampler {
  protected:
    std::mt19937_64 rng;

  public:
    LossyCircuit circuit;
    Sampler(const LossyCircuit &circuit, uint64_t seed);
    virtual ~Sampler() = default;

    virtual SampleBatch sample(size_t num_samples,
                               bool reroute_observables = false,
                               bool optimize_retoute = false) = 0;
    virtual std::tuple<py::array_t<uint8_t>, std::vector<LossPattern>>
    sample_measurements(size_t num_samples) = 0;
};
} // namespace qec_loss
