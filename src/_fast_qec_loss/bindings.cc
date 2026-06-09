#include <pybind11/pybind11.h>
#include "circuit/circuit.pybind.h"
#include "sampler/sampler.pybind.h"

#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "xxhash.h"

namespace py = pybind11;

PYBIND11_MODULE(_fast_qec_loss, m) {
    m.doc() = "Cpp quantum error correction loss module.";
    qec_loss::pybind_circuit(m);
    qec_loss::pybind_sampler(m);
}