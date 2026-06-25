#pragma once

#include "stim/circuit/circuit_instruction.h"
#include <cstdint>
#include <string>

namespace qec_loss {

enum class InstructionCategory : uint8_t {
    ONE_QUBIT,
    TWO_QUBIT,
    MEASURE,
    RESET,
    MEASUREMENT_AND_RESET,
    OTHER,
};

InstructionCategory
categorize_instruction(const stim::CircuitInstruction &instruction);

std::string to_base36(uint64_t value);

} // namespace qec_loss