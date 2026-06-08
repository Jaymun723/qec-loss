#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "loss_instruction.h"
#include "lossy_circuit.h"

namespace py = pybind11;

PYBIND11_MODULE(_fast_qec_loss, m) {
    m.doc() = "Cpp quantum error correction loss module.";

    py::class_<qec_loss::LossInstruction>(m, "LossInstruction")
        .def(py::init<std::string_view>())
        .def(py::init<std::vector<uint32_t>, double, std::string_view>())
        .def(py::init<std::vector<uint32_t>, double>()) // For convenience
        .def_readonly("p", &qec_loss::LossInstruction::p)
        .def_readonly("targets", &qec_loss::LossInstruction::targets)
        .def_readonly("tag", &qec_loss::LossInstruction::tag)
        .def("__str__", &qec_loss::LossInstruction::str);

    py::class_<qec_loss::LossyCircuit>(m, "LossyCircuit")
        .def(py::init<std::string_view>())
        .def_static("from_file", &qec_loss::LossyCircuit::from_file)
        .def("to_file", &qec_loss::LossyCircuit::to_file)
        .def("__str__", &qec_loss::LossyCircuit::str)
        .def("nominal_circuit", [](const qec_loss::LossyCircuit &c) {
            return c.nominal_circuit.str();
        });
}