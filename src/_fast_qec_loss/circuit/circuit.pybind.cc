#include "circuit.pybind.h"
#include "loss_instruction.h"
#include "lossy_circuit.h"
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace qec_loss {

void pybind_circuit(py::module &m) {
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
        .def_property_readonly(
            "nominal_circuit",
            [](const qec_loss::LossyCircuit &c) -> py::object {
                return py::module_::import("stim").attr("Circuit")(
                    c.nominal_circuit.str());
            })
        .def_readonly("num_qubits", &qec_loss::LossyCircuit::num_qubits)
        .def_readonly("num_measurements",
                      &qec_loss::LossyCircuit::num_measurements)
        .def_readonly("num_detectors", &qec_loss::LossyCircuit::num_detectors)
        .def_readonly("num_observables",
                      &qec_loss::LossyCircuit::num_observables);
}

} // namespace qec_loss
