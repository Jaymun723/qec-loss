#pragma once

#include "stim/circuit/circuit.h"
#include "stim/stabilizers/pauli_string.h"

#include <vector>

namespace qec_loss {

std::vector<stim::PauliString<stim::MAX_BITWORD_WIDTH>>
get_stabilizers(const stim::Circuit &circuit);

}