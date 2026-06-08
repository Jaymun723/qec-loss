#include "sampler.h"

namespace qec_loss {

Sampler::Sampler(const LossyCircuit &circuit, uint64_t seed)
    : circuit(circuit), rng(seed) {}

} // namespace qec_loss