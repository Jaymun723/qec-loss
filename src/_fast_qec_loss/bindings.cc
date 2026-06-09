#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "forward_sampler.h"
#include "loss_instruction.h"
#include "lossy_circuit.h"
#include "sampler.h"

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

    py::class_<qec_loss::Sampler>(m, "Sampler")
        .def("sample", &qec_loss::Sampler::sample)
        .def_readonly("circuit", &qec_loss::Sampler::circuit);

    py::class_<qec_loss::SampleBatch>(m, "Experiment")
        .def_readonly("measurements", &qec_loss::SampleBatch::measurements)
        .def_readonly("detectors", &qec_loss::SampleBatch::detectors)
        .def_readonly("observables", &qec_loss::SampleBatch::observables)
        .def_readonly("loss_patterns", &qec_loss::SampleBatch::loss_patterns);

    py::class_<qec_loss::ForwardSampler, qec_loss::Sampler>(m, "ForwardSampler")
        .def(
            py::init<const qec_loss::LossyCircuit &, std::optional<uint64_t>>(),
            py::arg("circuit"), py::arg("seed") = py::none())
        .def_readonly("circuit", &qec_loss::Sampler::circuit)
        .def("sample", &qec_loss::ForwardSampler::sample);
}