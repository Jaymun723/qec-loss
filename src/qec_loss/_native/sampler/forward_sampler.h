#pragma once

#include "../utils.h"
#include "sampler.h"
#include "stim/circuit/circuit.h"
#include <tuple>

namespace qec_loss {
class ForwardSampler : public Sampler {
  private:
    std::uniform_real_distribution<double> dist;

    std::vector<std::vector<size_t>> detector_dependencies;
    std::vector<std::vector<size_t>> observable_dependencies;
    std::vector<InstructionCategory> nominal_instruction_categories;

    void populate_shot_circuit(stim::Circuit &shot_circuit,
                               std::vector<size_t> &lost_measurements,
                               LossPattern &loss_patterns);

  public:
    ForwardSampler(const LossyCircuit &circuit,
                   std::optional<uint64_t> seed = std::nullopt);

    SampleBatch sample(size_t num_samples, bool reroute_observables = false,
                       bool optimize_retoute = false) override;

    std::tuple<py::array_t<uint8_t>, std::vector<LossPattern>>
    sample_measurements(size_t num_samples) override;
};
} // namespace qec_loss