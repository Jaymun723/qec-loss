#pragma once

#include "lossy_circuit.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <tuple>

namespace py = pybind11;

namespace qec_loss {

using LossPattern = std::vector<std::tuple<uint32_t, std::vector<uint32_t>>>;

class Experiment {
  public:
    py::array_t<uint8_t> measurements;
    py::array_t<uint8_t> detectors;
    py::array_t<uint8_t> observables;
    std::vector<LossPattern> loss_patterns;
};

class Sampler {
  protected:
    std::mt19937_64 rng;
    LossyCircuit circuit;

  public:
    Sampler(const LossyCircuit &circuit, uint64_t seed);
    virtual ~Sampler() = default;

    virtual std::tuple<py::array_t<uint8_t>, std::vector<LossPattern>>
    sample_measurements(size_t num_samples) = 0;

    virtual std::tuple<py::array_t<uint8_t>, py::array_t<uint8_t>,
                       std::vector<LossPattern>>
    sample_detectors(size_t num_samples) = 0;

    virtual std::vector<Experiment> sample_experiments(size_t num_samples) = 0;
};
} // namespace qec_loss
