#include "sampler.pybind.h"
#include "forward_sampler.h"
#include "sampler.h"
#include <pybind11/stl.h>

namespace qec_loss {

void pybind_sampler(py::module &m) {
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

} // namespace qec_loss
