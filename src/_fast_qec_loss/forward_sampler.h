#pragma once

#include "sampler.h"
#include "stim/circuit/circuit.h"
#include <tuple>

namespace qec_loss {
class ForwardSampler : public Sampler {
  private:
    std::uniform_real_distribution<double> dist;
    void populate_shot_circuit(stim::Circuit &shot_circuit,
                               std::vector<uint32_t> &lost_measurements,
                               LossPattern &loss_patterns);

  public:
    ForwardSampler(const LossyCircuit &circuit,
                   std::optional<uint64_t> seed = std::nullopt);

    std::tuple<py::array_t<uint8_t>, std::vector<LossPattern>>
    sample_measurements(size_t num_samples) override;

    std::tuple<py::array_t<uint8_t>, py::array_t<uint8_t>,
               std::vector<LossPattern>>
    sample_detectors(size_t num_samples) override;

    std::vector<Experiment> sample_experiments(size_t num_samples) override;
};
} // namespace qec_loss