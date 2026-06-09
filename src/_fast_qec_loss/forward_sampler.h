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

    SampleBatch sample(size_t num_samples) override;
};
} // namespace qec_loss