#pragma once

#include "stim/circuit/circuit.h"
#include "stim/circuit/circuit_instruction.h"
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

namespace qec_loss {

/// @brief Processes a single instruction, updating the circuit and tracking
/// lost measurements and qubits.
/// @param instr The instruction to process.
/// @param circuit The final to populate with the processed instruction.
/// @param lost_measurements A list of indices for measurements that have been
/// lost.
/// @param current_measurement_index A reference to the index of the current
/// measurement.
/// @param lost_qubits A mask indicating which qubits have been lost (1 for
/// lost, 0 for not lost).
void do_instruction(const stim::CircuitInstruction &instr,
                    stim::Circuit &circuit,
                    std::vector<uint32_t> &lost_measurements,
                    size_t &current_measurement_index,
                    std::vector<uint8_t> &lost_qubits);

} // namespace qec_loss