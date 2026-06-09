#include "monaka.pybind.h"
#include "life_cycle.h"
#include "monaka_builder.h"
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

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
}

} // namespace qec_loss