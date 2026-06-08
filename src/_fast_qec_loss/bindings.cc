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

    py::class_<qec_loss::Sampler>(m, "Sampler")
        .def("sample_measurements", &qec_loss::Sampler::sample_measurements)
        .def("sample_detectors", &qec_loss::Sampler::sample_detectors)
        .def("sample_experiments", &qec_loss::Sampler::sample_experiments);

    py::class_<qec_loss::Experiment>(m, "Experiment")
        .def_readonly("measurements", &qec_loss::Experiment::measurements)
        .def_readonly("detectors", &qec_loss::Experiment::detectors)
        .def_readonly("observables", &qec_loss::Experiment::observables)
        .def_readonly("loss_patterns", &qec_loss::Experiment::loss_patterns);

    py::class_<qec_loss::ForwardSampler, qec_loss::Sampler>(m, "ForwardSampler")
        .def(
            py::init<const qec_loss::LossyCircuit &, std::optional<uint64_t>>(),
            py::arg("circuit"), py::arg("seed") = py::none())
        .def("sample_measurements",
             &qec_loss::ForwardSampler::sample_measurements);

    py::class_<qec_loss::LossyCircuit>(m, "LossyCircuit")
        .def(py::init<std::string_view>())
        .def_static("from_file", &qec_loss::LossyCircuit::from_file)
        .def("to_file", &qec_loss::LossyCircuit::to_file)
        .def("__str__", &qec_loss::LossyCircuit::str)
        .def("nominal_circuit", [](const qec_loss::LossyCircuit &c) {
            return c.nominal_circuit.str();
        });
}