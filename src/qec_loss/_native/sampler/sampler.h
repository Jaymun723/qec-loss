#pragma once

#include "../circuit/lossy_circuit.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <tuple>

namespace py = pybind11;

namespace qec_loss {

using LossPattern = std::vector<std::tuple<size_t, std::vector<uint32_t>>>;

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
