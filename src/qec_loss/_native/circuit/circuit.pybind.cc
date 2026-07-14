#include "circuit.pybind.h"
#include "loss_instruction.h"
#include "lossy_circuit.h"
#include "../sampler/forward_sampler.h"
#include "../monaka/monaka_builder.h"
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace qec_loss {

void pybind_circuit(py::module &m) {
    py::class_<qec_loss::LossInstruction>(
        m, "LossInstruction",
        R"doc(Loss instruction for circuit definition.)doc")
        .def(py::init<std::string_view>(), py::arg("instruction_str"))
        .def(py::init<std::vector<uint32_t>, double, std::string_view>(),
             py::arg("targets"), py::arg("p"), py::arg("tag"))
        .def(py::init<std::vector<uint32_t>, double>(), py::arg("targets"),
             py::arg("p")) // For convenience
        .def_readonly("p", &qec_loss::LossInstruction::p)
        .def_readonly("targets", &qec_loss::LossInstruction::targets)
        .def_readonly("tag", &qec_loss::LossInstruction::tag)
        .def("__str__", &qec_loss::LossInstruction::str);

    py::class_<qec_loss::LossyCircuit>(
        m, "LossyCircuit",
        R"doc(Lossy circuit for quantum error correction.)doc")
        .def(py::init<std::string_view>(), py::arg("circuit_str"),
             R"doc(Initialize a LossyCircuit from a stim-like circuit string.)doc")
        .def_static("from_file", &qec_loss::LossyCircuit::from_file,
                    py::arg("circuit_path"),
                    R"doc(Load a LossyCircuit from a file.)doc")
        .def("to_file", &qec_loss::LossyCircuit::to_file,
             R"doc(Save the LossyCircuit to a file.)doc")
        .def("__str__", &qec_loss::LossyCircuit::str)
        .def_property_readonly(
            "nominal_circuit",
            [](const qec_loss::LossyCircuit &c) -> py::object {
                return py::module_::import("stim").attr("Circuit")(
                    c.nominal_circuit.str());
            })
        .def_readonly("rerouter", &qec_loss::LossyCircuit::rerouter)
        .def_readonly("num_qubits", &qec_loss::LossyCircuit::num_qubits)
        .def_readonly("num_measurements",
                      &qec_loss::LossyCircuit::num_measurements)
        .def_readonly("num_detectors", &qec_loss::LossyCircuit::num_detectors)
        .def_readonly("num_observables",
                      &qec_loss::LossyCircuit::num_observables)
        .def("compile_forward_sampler",
             [](const qec_loss::LossyCircuit &c, std::optional<uint64_t> seed) {
                 return qec_loss::ForwardSampler(c, seed);
             },
             py::arg("seed") = py::none(),
             R"doc(Compile the circuit into a ForwardSampler for simulating outcomes.
             
Args:
    seed: Optional random seed.

Returns:
    A ForwardSampler instance.)doc")
        .def("compile_monaka_builder",
             [](const qec_loss::LossyCircuit &c, bool optimize_rerouting) {
                 return std::make_unique<qec_loss::MonakaBuilder>(c, optimize_rerouting);
             },
             py::arg("optimize_rerouting") = false,
             R"doc(Compile the circuit into a MonakaBuilder for tracking lost qubits.
             
Args:
    optimize_rerouting: Whether to optimize rerouting logic.

Returns:
    A MonakaBuilder instance.)doc");
}

} // namespace qec_loss
