#include "sampler.pybind.h"
#include "forward_sampler.h"
#include "sampler.h"
#include <pybind11/stl.h>

namespace qec_loss {

void pybind_sampler(py::module &m) {
    py::class_<Sampler>(m, "Sampler")
        .def("sample", &Sampler::sample, py::arg("num_samples"),
             py::arg("reroute_observables") = false,
             py::arg("optimize_retoute") = false)
        .def_readonly("circuit", &Sampler::circuit);

    py::class_<SampleBatch>(m, "SampleBatch")
        .def(py::init<py::array_t<uint8_t>, py::array_t<uint8_t>,
                      py::array_t<uint8_t>, std::vector<LossPattern>>(),
             py::arg("measurements"), py::arg("detectors"),
             py::arg("observables"), py::arg("loss_patterns"))
        .def(py::init<py::array_t<uint8_t>, py::array_t<uint8_t>,
                      py::array_t<uint8_t>,
                      std::vector<
                          std::unordered_map<size_t, std::vector<uint32_t>>>>(),
             py::arg("measurements"), py::arg("detectors"),
             py::arg("observables"), py::arg("loss_patterns"))
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

    py::class_<ForwardSampler, Sampler>(m, "ForwardSampler")
        .def(py::init<const LossyCircuit &, std::optional<uint64_t>>(),
             py::arg("circuit"), py::arg("seed") = py::none())
        .def_readonly("circuit", &Sampler::circuit)
        .def("sample", &ForwardSampler::sample, py::arg("num_samples"),
             py::arg("reroute_observables") = false,
             py::arg("optimize_retoute") = false)
        .def("sample_measurements", &ForwardSampler::sample_measurements);
}

} // namespace qec_loss
