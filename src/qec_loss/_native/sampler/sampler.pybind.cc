#include "sampler.pybind.h"
#include "forward_sampler.h"
#include "sampler.h"
#include <pybind11/stl.h>

namespace qec_loss {

void pybind_sampler(py::module &m) {
    py::class_<Sampler>(m, "Sampler", R"doc(Base Sampler for quantum circuits.)doc")
        .def("sample", &Sampler::sample, py::arg("num_samples"),
             py::arg("reroute_observables") = false,
             py::arg("optimize_retoute") = false,
             R"doc(Generate samples from the circuit.)doc")
        .def_readonly("circuit", &Sampler::circuit, R"doc(The circuit being sampled.)doc");

    py::class_<SampleBatch>(m, "SampleBatch", R"doc(A batch of circuit samples containing measurements, detectors, observables, and loss patterns.)doc")
        .def(py::init<py::array_t<uint8_t>, py::array_t<uint8_t>,
                      py::array_t<uint8_t>, std::vector<LossPattern>>(),
             py::arg("measurements"), py::arg("detectors"),
             py::arg("observables"), py::arg("loss_patterns"), R"doc(Initialize SampleBatch with LossPatterns.)doc")
        .def(py::init<py::array_t<uint8_t>, py::array_t<uint8_t>,
                      py::array_t<uint8_t>,
                      std::vector<
                          std::unordered_map<size_t, std::vector<uint32_t>>>>(),
             py::arg("measurements"), py::arg("detectors"),
             py::arg("observables"), py::arg("loss_patterns"), R"doc(Initialize SampleBatch with mapped loss patterns.)doc")
        .def_readonly("measurements", &SampleBatch::measurements)
        .def_readonly("detectors", &SampleBatch::detectors)
        .def_readonly("observables", &SampleBatch::observables)
        .def_property_readonly("loss_patterns",
                               [](const SampleBatch &self) {
                                   return convert_loss_patterns_back(
                                       self.loss_patterns);
                               })
        .def("__repr__",
             [](const SampleBatch &self) {
                 return "SampleBatch(measurements=" +
                        py::repr(self.measurements).cast<std::string>() +
                        ", detectors=" +
                        py::repr(self.detectors).cast<std::string>() +
                        ", observables=" +
                        py::repr(self.observables).cast<std::string>() +
                        ", loss_patterns=" +
                        py::repr(py::cast(self.loss_patterns))
                            .cast<std::string>() +
                        ")";
             })
        .def(py::pickle(
            [](const SampleBatch &self) { // __getstate__
                return py::make_tuple(self.measurements, self.detectors,
                                      self.observables, self.loss_patterns);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid SampleBatch state");

                return SampleBatch(t[0].cast<py::array_t<uint8_t>>(),
                                   t[1].cast<py::array_t<uint8_t>>(),
                                   t[2].cast<py::array_t<uint8_t>>(),
                                   t[3].cast<std::vector<LossPattern>>());
            }));

    py::class_<ForwardSampler, Sampler>(m, "ForwardSampler", R"doc(A sampler that tracks erasure errors forward through the circuit.)doc")
        .def(py::init<const LossyCircuit &, std::optional<uint64_t>>(),
             py::arg("circuit"), py::arg("seed") = py::none(), R"doc(Initialize the ForwardSampler.)doc")
        .def_readonly("circuit", &Sampler::circuit, R"doc(The circuit being sampled.)doc")
        .def("sample", &ForwardSampler::sample, py::arg("num_samples"),
             py::arg("reroute_observables") = false,
             py::arg("optimize_retoute") = false,
             R"doc(Generate samples taking loss routing into account.)doc")
        .def("sample_measurements", &ForwardSampler::sample_measurements, R"doc(Sample only measurement outcomes.)doc");
}

} // namespace qec_loss
