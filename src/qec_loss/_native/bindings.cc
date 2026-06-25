#include "circuit/circuit.pybind.h"
#include "f2_tensor/f2_tensor.pybind.h"
#include "monaka/monaka.pybind.h"
#include "observable/reroute.pybind.h"
#include "sampler/sampler.pybind.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
    m.doc() = "Loss sampling and Monaka style decoding for quantum error "
              "correction circuits";
    qec_loss::pybind_circuit(m);
    qec_loss::pybind_sampler(m);
    qec_loss::pybind_monaka_builder(m);
    qec_loss::pybind_f2_tensor(m);
    qec_loss::pybind_reroute(m);
}