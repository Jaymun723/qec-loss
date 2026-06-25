#include "reroute.pybind.h"
#include "./stabilizers.h"
#include "reroute.h"
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace qec_loss {

void pybind_reroute(py::module &m) {
    py::class_<DetsRerouter>(m, "DetsRerouter")
        .def(py::init([](py::object stim_circuit,
                         std::vector<uint32_t> data_qubits) {
                 std::string circuit_str =
                     py::str(stim_circuit).cast<std::string>();
                 stim::Circuit circuit(circuit_str);
                 return DetsRerouter(circuit, data_qubits);
             }),
             py::arg("circuit"), py::arg("data_qubits"))
        .def(py::init([](py::object stim_circuit) {
                 std::string circuit_str =
                     py::str(stim_circuit).cast<std::string>();
                 stim::Circuit circuit(circuit_str);
                 return DetsRerouter(circuit);
             }),
             py::arg("circuit"))
        .def("get_qubit_for_measurement",
             &DetsRerouter::get_qubit_for_measurement,
             py::arg("measurement_index"))
        .def_property_readonly("data_qubits", &DetsRerouter::get_data_qubits)
        .def(
            "reroute",
            [](const DetsRerouter &self, size_t observable_index,
               const std::vector<uint32_t> &lost_qubits, bool optimize) {
                std::vector<stim::GateTarget> targets =
                    self.reroute(observable_index, lost_qubits, optimize);
                py::list py_targets;
                py::object stim_mod = py::module_::import("stim");
                for (const auto &t : targets) {
                    py_targets.append(
                        stim_mod.attr("target_rec")(t.rec_offset()));
                }
                return py_targets;
            },
            py::arg("observable_index"), py::arg("lost_qubits"),
            py::arg("optimize") = false);

    m.def(
        "get_stabilizers",
        [](py::object stim_circuit) {
            std::string circuit_str = py::str(stim_circuit).cast<std::string>();
            stim::Circuit circuit(circuit_str);

            std::vector<stim::PauliString<stim::MAX_BITWORD_WIDTH>>
                stabilizers = get_stabilizers(circuit);

            py::list py_stabilizers;
            py::object stim_mod = py::module_::import("stim");
            for (const auto &s : stabilizers) {
                py_stabilizers.append(stim_mod.attr("PauliString")(s.str()));
            }
            return py_stabilizers;
        },
        py::arg("circuit"));

    py::class_<PauliRerouter>(m, "PauliRerouter")
        .def(py::init([](py::object stim_circuit) {
                 std::string circuit_str =
                     py::str(stim_circuit).cast<std::string>();
                 stim::Circuit circuit(circuit_str);
                 return PauliRerouter(circuit);
             }),
             py::arg("circuit"))
        .def_property_readonly(
            "L",
            [](const PauliRerouter &self) {
                py::list py_L;
                py::object stim_mod = py::module_::import("stim");
                for (const auto &s : self.get_L()) {
                    py_L.append(stim_mod.attr("PauliString")(s.str()));
                }
                return py_L;
            })
        .def_property_readonly("final_measurements",
                               &PauliRerouter::get_final_measurements)
        .def_property_readonly("meas_so_far", &PauliRerouter::get_meas_so_far)
        .def(
            "reroute",
            [](const PauliRerouter &self, size_t observable_index,
               const std::vector<uint32_t> &lost_qubits, bool optimize) {
                const auto &targets =
                    self.reroute(observable_index, lost_qubits, optimize);
                py::list py_targets;
                py::object stim_mod = py::module_::import("stim");
                for (const auto &t : targets) {
                    py_targets.append(
                        stim_mod.attr("target_rec")(t.rec_offset()));
                }
                return py_targets;
            },
            py::arg("observable_index"), py::arg("lost_qubits"),
            py::arg("optimize") = false)
        .def("riroute", &PauliRerouter::riroute, py::arg("S"), py::arg("L"),
             py::arg("available_measurements"))
        .def("get_S_and_L_matrices", &PauliRerouter::get_S_and_L_matrices,
             py::arg("observable_index"), py::arg("lost_qubits"),
             py::arg("optimize") = false);
}
} // namespace qec_loss
