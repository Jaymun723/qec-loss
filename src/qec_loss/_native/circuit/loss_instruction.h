#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace qec_loss {
class LossInstruction {
  public:
    LossInstruction() = delete;

    double p;
    uint64_t threshold;

    std::vector<uint32_t> targets;

    std::string tag;

    LossInstruction(std::vector<uint32_t> targets, double p,
                    std::string_view tag);

    LossInstruction(std::vector<uint32_t> targets, double p)
        : LossInstruction(std::move(targets), p, "") {};

    LossInstruction(std::string_view instruction_str);

    std::string str() const;
};
} // namespace qec_loss