#include "monaka.pybind.h"
#include "get_loss_dem.h"
#include "life_cycle.h"
#include "monaka_builder.h"
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <optional>

namespace qec_loss {
void pybind_monaka_builder(py::module &m) {
    py::class_<MonakaBuilder>(m, "MonakaBuilder")
        .def(py::init<const LossyCircuit &, const std::filesystem::path &>(),
             py::arg("circuit"), py::arg("model_path"))
        .def_readonly("save_path", &MonakaBuilder::save_path)
        .def_property_readonly("nominal_dem", &MonakaBuilder::get_nominal_dem)
        .def("prebuild", &MonakaBuilder::prebuild)
        .def("decode_batch", &MonakaBuilder::decode_batch);

    py::class_<LifeSegment>(m, "LifeSegment")
        .def(py::init<size_t, size_t>(), py::arg("start"), py::arg("end"))
        .def_readonly("start", &LifeSegment::start)
        .def_readonly("end", &LifeSegment::end)
        .def("__str__", &LifeSegment::str)
        .def("__repr__", &LifeSegment::str)
        .def_readonly("loss_locations", &LifeSegment::loss_locations);

    py::class_<LifeCycleManager>(m, "LifeCycleManager")
        .def(py::init<const LossyCircuit &>(), py::arg("circuit"))
        .def("get_life_cycle", &LifeCycleManager::get_life_cycle)
        .def("get_life_segment_for_measurement",
             &LifeCycleManager::get_life_segment_for_measurement);

    m.def(
        "get_loss_dem",
        [](const LossyCircuit &circuit, const uint32_t qubit,
           const LifeSegment &life_segment) {
            stim::DetectorErrorModel dem =
                get_loss_dem(circuit, qubit, life_segment);
            return py::module_::import("stim").attr("DetectorErrorModel")(
                dem.str());
        },
        py::arg("circuit"), py::arg("qubit"), py::arg("life_segment"));

    m.def(
        "combine_circuits_into_dem",
        [](const std::vector<py::object> &py_circuits,
           const std::optional<std::vector<double>> &weights) -> py::object {
            std::vector<stim::Circuit> circuits;
            for (const auto &py_c : py_circuits) {
                circuits.push_back(
                    stim::Circuit(py::str(py_c).cast<std::string>()));
            }
            std::vector<double> w;
            if (weights.has_value()) {
                w = weights.value();
            } else {
                w = std::vector<double>(circuits.size(), 1.0 / circuits.size());
            }
            stim::DetectorErrorModel dem =
                combine_circuits_into_dem(circuits, w);
            return py::module_::import("stim").attr("DetectorErrorModel")(
                dem.str());
        },
        py::arg("circuits"), py::arg("weights") = py::none());
}

} // namespace qec_loss