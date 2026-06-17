#include "monaka.pybind.h"
#include "get_loss_dem.h"
#include "life_cycle.h"
#include "monaka_builder.h"
#include <optional>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace qec_loss {
void pybind_monaka_builder(py::module &m) {
    py::class_<MonakaBuilder>(m, "MonakaBuilder")
        .def(py::init<const LossyCircuit &, const std::filesystem::path &,
                      bool>(),
             py::arg("circuit"), py::arg("model_path"),
             py::arg("optimize_rerouting") = false)
        .def_readonly("save_path", &MonakaBuilder::save_path)
        .def_property_readonly("nominal_dem", &MonakaBuilder::get_nominal_dem)
        .def("decode_batch", &MonakaBuilder::decode_batch)
        .def_readonly("life_cycle_manager", &MonakaBuilder::life_cycle_manager)
        .def(
            "get_dem_for_shot",
            [](MonakaBuilder &self,
               const std::vector<uint32_t> &lost_data_qubits,
               py::array_t<uint8_t> measurements, size_t shot_i) {
                stim::DetectorErrorModel dem = self.get_dem_for_shot(
                    lost_data_qubits, measurements, shot_i);
                return py::module_::import("stim").attr("DetectorErrorModel")(
                    dem.str());
            },
            py::arg("lost_data_qubits"), py::arg("measurements"),
            py::arg("shot_i"))
        .def("get_life_segment_dem",
             [](MonakaBuilder &self,
                const std::vector<uint32_t> &lost_data_qubits,
                const LifeSegment &life_segment) {
                 stim::DetectorErrorModel dem =
                     self.get_life_segment_dem(lost_data_qubits, life_segment);
                 //  std::cout << "get_life_segment_dem: workded" << std::endl;
                 //  std::cout << "dem.str():\n" << dem.str() << "\n";
                 //  std::cout << "life_segment:\n" << life_segment.str() <<
                 //  "\n";
                 return py::module_::import("stim").attr("DetectorErrorModel")(
                     dem.str());
             });

    py::class_<LifeSegment>(m, "LifeSegment")
        .def(py::init<uint32_t, size_t, size_t>(), py::arg("qubit"),
             py::arg("start"), py::arg("end"))
        .def_readonly("start", &LifeSegment::start)
        .def_readonly("end", &LifeSegment::end)
        .def_readonly("qubit", &LifeSegment::qubit)
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
        [](const LossyCircuit &circuit,
           const std::vector<uint32_t> &lost_data_qubits,
           const LifeSegment &life_segment, bool optimize_rerouting) {
            stim::DetectorErrorModel dem = get_loss_dem(
                circuit, lost_data_qubits, life_segment, optimize_rerouting);
            return py::module_::import("stim").attr("DetectorErrorModel")(
                dem.str());
        },
        py::arg("circuit"), py::arg("lost_data_qubits"),
        py::arg("life_segment"), py::arg("optimize_rerouting") = false);

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