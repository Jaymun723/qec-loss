#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace qec_loss {
void pybind_monaka_builder(py::module &m);
} // namespace qec_loss
