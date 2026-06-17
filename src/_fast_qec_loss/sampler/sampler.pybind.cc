#include "sampler.pybind.h"
#include "forward_sampler.h"
#include "sampler.h"
#include <pybind11/stl.h>

namespace qec_loss {

void pybind_sampler(py::module &m) {
    py::class_<qec_loss::Sampler>(m, "Sampler")
        .def("sample", &qec_loss::Sampler::sample, py::arg("num_samples"),
             py::arg("reroute_observables") = false,
             py::arg("optimize_retoute") = false)
        .def_readonly("circuit", &qec_loss::Sampler::circuit);

    py::class_<qec_loss::SampleBatch>(m, "SampleBatch")
        .def_readonly("measurements", &qec_loss::SampleBatch::measurements)
        .def_readonly("detectors", &qec_loss::SampleBatch::detectors)
        .def_readonly("observables", &qec_loss::SampleBatch::observables)
        .def_readonly("loss_patterns", &qec_loss::SampleBatch::loss_patterns)
        .def("__repr__", [](const qec_loss::SampleBatch &self) {
            return "SampleBatch(measurements=" +
                   py::repr(self.measurements).cast<std::string>() +
                   ", detectors=" +
                   py::repr(self.detectors).cast<std::string>() +
                   ", observables=" +
                   py::repr(self.observables).cast<std::string>() +
                   ", loss_patterns=" +
                   py::repr(py::cast(self.loss_patterns)).cast<std::string>() +
                   ")";
        });

    py::class_<qec_loss::ForwardSampler, qec_loss::Sampler>(m, "ForwardSampler")
        .def(
            py::init<const qec_loss::LossyCircuit &, std::optional<uint64_t>>(),
            py::arg("circuit"), py::arg("seed") = py::none())
        .def_readonly("circuit", &qec_loss::Sampler::circuit)
        .def("sample", &qec_loss::ForwardSampler::sample, py::arg("num_samples"),
             py::arg("reroute_observables") = false,
             py::arg("optimize_retoute") = false)
        .def("sample_measurements",
             &qec_loss::ForwardSampler::sample_measurements);
}

} // namespace qec_loss
