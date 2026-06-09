#pragma once

#include "../circuit/lossy_circuit.h"
#include <cstdint>
#include <tuple>
#include <vector>

namespace qec_loss {

struct LifeSegment {
    size_t start;
    size_t end;
    std::vector<size_t> loss_locations;

    LifeSegment(size_t start, size_t end)
        : start(start), end(end), loss_locations() {}

    const std::string str() const;
};

using LifeCycle = std::vector<LifeSegment>;

class LifeCycleManager {
  private:
    std::vector<LifeCycle> life_cycles;
    std::vector<std::tuple<uint32_t, size_t>> measurement_to_life_cycle;
    const LossyCircuit &circuit;

  public:
    LifeCycleManager(const LossyCircuit &circuit);

    const LifeCycle &get_life_cycle(uint32_t qubit_index) const;

    const std::tuple<uint32_t, const LifeSegment>
    get_life_segment_for_measurement(size_t measurement_index) const;
};

} // namespace qec_loss