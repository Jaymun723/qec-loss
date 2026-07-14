#include "monaka.pybind.h"
#include "get_loss_dem.h"
#include "life_cycle.h"
#include "monaka_builder.h"
#include <optional>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace qec_loss {
void pybind_monaka_builder(py::module &m) {
    py::class_<MonakaBuilder>(m, "MonakaBuilder", R"doc(Builder to analyze and compile detector error models under loss.)doc")
        .def(py::init<const LossyCircuit &, bool>(), py::arg("circuit"),
             py::arg("optimize_rerouting") = false, R"doc(Initialize MonakaBuilder with a LossyCircuit.)doc")
        .def(
            "get_nominal_dem",
            [](MonakaBuilder &self, std::vector<uint32_t> lost_qubits) {
                stim::DetectorErrorModel dem =
                    self.get_nominal_dem(lost_qubits);
                return py::module_::import("stim").attr("DetectorErrorModel")(
                    dem.str());
            },
            py::arg("lost_qubits") = std::vector<uint32_t>(),
            R"doc(Get the nominal DetectorErrorModel given a set of initially lost qubits.)doc")
        .def("get_dems_from_batch",
             [](MonakaBuilder &self, SampleBatch &batch) {
                 std::vector<stim::DetectorErrorModel> dems =
                     self.get_dems_from_batch(batch);
                 std::vector<py::object> py_dems;
                 for (const auto &dem : dems) {
                     py_dems.push_back(py::module_::import("stim").attr(
                         "DetectorErrorModel")(dem.str()));
                 }
                 return py_dems;
             },
             R"doc(Get DEMs for an entire batch of samples.)doc")
        .def("decode_batch", &MonakaBuilder::decode_batch, py::arg("batch"),
             py::arg("include_loss_dem") = true,
             py::arg("post_select_on_usable_shots") = true,
             R"doc(Decode a batch of loss outcomes.)doc")
        .def_readonly("life_cycle_manager", &MonakaBuilder::life_cycle_manager,
             R"doc(Manager for qubit lifecycles.)doc")
        .def("get_dem_from_measurements",
             [](MonakaBuilder &self, py::array_t<uint8_t> measurements) {
                 stim::DetectorErrorModel dem =
                     self.get_dem_from_measurements(measurements);
                 return py::module_::import("stim").attr("DetectorErrorModel")(
                     dem.str());
             },
             R"doc(Get the DEM given a specific set of measurement outcomes.)doc")
        .def("get_life_segment_dem",
             [](MonakaBuilder &self, const std::vector<uint32_t> &lost_qubits,
                const LifeSegment &life_segment) {
                 stim::DetectorErrorModel dem =
                     self.get_life_segment_dem(lost_qubits, life_segment);
                 //  std::cout << "get_life_segment_dem: workded" << std::endl;
                 //  std::cout << "dem.str():\n" << dem.str() << "\n";
                 //  std::cout << "life_segment:\n" << life_segment.str() <<
                 //  "\n";
                 return py::module_::import("stim").attr("DetectorErrorModel")(
                     dem.str());
             },
             R"doc(Get the DEM restricted to a specific life segment.)doc");

    py::class_<LifeSegment>(m, "LifeSegment", R"doc(Represents a continuous segment of a qubit's lifecycle.)doc")
        .def(py::init<uint32_t, size_t, size_t>(), py::arg("qubit"),
             py::arg("start"), py::arg("end"), R"doc(Initialize LifeSegment.)doc")
        .def_readonly("start", &LifeSegment::start)
        .def_readonly("end", &LifeSegment::end)
        .def_readonly("qubit", &LifeSegment::qubit)
        .def("__str__", &LifeSegment::str)
        .def("__repr__", &LifeSegment::str)
        .def_readonly("loss_locations", &LifeSegment::loss_locations);

    py::class_<LifeCycleManager>(m, "LifeCycleManager", R"doc(Manages and resolves lifecycles of qubits.)doc")
        .def(py::init<const LossyCircuit &>(), py::arg("circuit"))
        .def("get_life_cycle", &LifeCycleManager::get_life_cycle, R"doc(Get the full lifecycle of a given qubit.)doc")
        .def("get_life_segment_for_measurement",
             &LifeCycleManager::get_life_segment_for_measurement, R"doc(Get the life segment associated with a measurement.)doc");

    m.def(
        "get_loss_rewritten_circuits",
        [](const LossyCircuit &circuit,
           const std::vector<uint32_t> &lost_qubits,
           const LifeSegment &life_segment, bool optimize_rerouting) {
            std::vector<stim::Circuit> result = get_loss_rewritten_circuits(
                circuit, lost_qubits, life_segment, optimize_rerouting);
            std::vector<py::object> py_circuits;
            for (const auto &c : result) {
                py_circuits.push_back(
                    py::module_::import("stim").attr("Circuit")(c.str()));
            }
            return py_circuits;
        },
        py::arg("circuit"), py::arg("lost_qubits"), py::arg("life_segment"),
        py::arg("optimize_rerouting") = false,
        R"doc(Get rewritten circuits accounting for lost qubits in a segment.)doc");

    m.def(
        "get_loss_dem",
        [](const LossyCircuit &circuit,
           const std::vector<uint32_t> &lost_qubits,
           const LifeSegment &life_segment, bool optimize_rerouting) {
            stim::DetectorErrorModel dem = get_loss_dem(
                circuit, lost_qubits, life_segment, optimize_rerouting);
            return py::module_::import("stim").attr("DetectorErrorModel")(
                dem.str());
        },
        py::arg("circuit"), py::arg("lost_qubits"), py::arg("life_segment"),
        py::arg("optimize_rerouting") = false,
        R"doc(Get the effective DetectorErrorModel for a life segment with lost qubits.)doc");

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
        py::arg("circuits"), py::arg("weights") = py::none(),
        R"doc(Combine multiple circuits into a single weighted DetectorErrorModel.)doc");
}

} // namespace qec_loss