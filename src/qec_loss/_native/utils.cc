#include "utils.h"

#include <algorithm>
#include <iterator>
#include <sstream>

namespace qec_loss {

std::string to_base36(uint64_t value) {
    if (value == 0)
        return "0";
    static const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::string result;
    while (value > 0) {
        result += chars[value % 36];
        value /= 36;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

InstructionCategory
categorize_instruction(const stim::CircuitInstruction &instruction) {
    using stim::GateType;
    switch (instruction.gate_type) {
        // 1q Gate
    case GateType::I:
    case GateType::X:
    case GateType::Y:
    case GateType::Z:
    case GateType::C_XYZ:
    case GateType::C_ZYX:
    case GateType::C_NXYZ:
    case GateType::C_XNYZ:
    case GateType::C_XYNZ:
    case GateType::C_NZYX:
    case GateType::C_ZNYX:
    case GateType::C_ZYNX:
    case GateType::H:
    case GateType::H_XY:
    case GateType::H_YZ:
    case GateType::H_NXY:
    case GateType::H_NXZ:
    case GateType::H_NYZ:
    case GateType::SQRT_X:
    case GateType::SQRT_X_DAG:
    case GateType::SQRT_Y:
    case GateType::SQRT_Y_DAG:
    case GateType::S:
    case GateType::S_DAG:
    // 1q error
    case GateType::HERALDED_ERASE:
    case GateType::HERALDED_PAULI_CHANNEL_1:
    case GateType::PAULI_CHANNEL_1:
    case GateType::DEPOLARIZE1:
    case GateType::X_ERROR:
    case GateType::Y_ERROR:
    case GateType::Z_ERROR:
    case GateType::I_ERROR:
        return InstructionCategory::ONE_QUBIT;
    // 2q Gate
    case GateType::CX:
    case GateType::CY:
    case GateType::CZ:
    case GateType::XCX:
    case GateType::XCY:
    case GateType::XCZ:
    case GateType::YCX:
    case GateType::YCY:
    case GateType::YCZ:
    case GateType::SQRT_XX:
    case GateType::SQRT_XX_DAG:
    case GateType::SQRT_YY:
    case GateType::SQRT_YY_DAG:
    case GateType::SQRT_ZZ:
    case GateType::SQRT_ZZ_DAG:
    case GateType::CZSWAP:
    case GateType::ISWAP:
    case GateType::ISWAP_DAG:
    case GateType::CXSWAP:
    case GateType::SWAP:
    case GateType::SWAPCX:
    case GateType::II:
        // 2q error
    case GateType::DEPOLARIZE2:
    case GateType::II_ERROR:
    case GateType::PAULI_CHANNEL_2:
        return InstructionCategory::TWO_QUBIT;
        // meas only
    case GateType::M:
    case GateType::MX:
    case GateType::MY:
        return InstructionCategory::MEASURE;
    // reset only
    case GateType::R:
    case GateType::RX:
    case GateType::RY:
        return InstructionCategory::RESET;
    // meas and reset
    case GateType::MR:
    case GateType::MRX:
    case GateType::MRY:
        return InstructionCategory::MEASUREMENT_AND_RESET;
    default:
        return InstructionCategory::OTHER;
    }
}
} // namespace qec_loss