#pragma once

#include "../circuit/lossy_circuit.h"
#include <cstdint>
#include <tuple>
#include <vector>

#define XXH_INLINE_ALL
#include "xxhash.h"

namespace qec_loss {

struct LifeSegment {
    uint32_t qubit;

    size_t start;
    size_t end;
    std::vector<size_t> loss_locations;

    LifeSegment(uint32_t qubit, size_t start, size_t end)
        : qubit(qubit), start(start), end(end), loss_locations() {}

    bool operator==(const LifeSegment &other) const;

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

    const LifeSegment
    get_life_segment_for_measurement(size_t measurement_index) const;
};

} // namespace qec_loss

namespace std {
template <> struct hash<qec_loss::LifeSegment> {
    size_t operator()(const qec_loss::LifeSegment &ls) const {
        uint64_t h = XXH3_64bits(&ls.start, sizeof(ls.start));
        h = XXH3_64bits_withSeed(&ls.end, sizeof(ls.end), h);
        if (!ls.loss_locations.empty()) {
            h = XXH3_64bits_withSeed(ls.loss_locations.data(),
                                     ls.loss_locations.size() * sizeof(size_t),
                                     h);
        }
        return static_cast<size_t>(h);
    }
};
} // namespace std