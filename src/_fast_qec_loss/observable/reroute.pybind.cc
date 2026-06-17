#include "reroute.pybind.h"
#include "reroute.h"
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace qec_loss {

void pybind_reroute(py::module &m) {
    py::class_<Rerouter>(m, "Rerouter")
        .def(py::init([](py::object stim_circuit,
                         std::vector<uint32_t> data_qubits) {
                 std::string circuit_str =
                     py::str(stim_circuit).cast<std::string>();
                 stim::Circuit circuit(circuit_str);
                 return Rerouter(circuit, data_qubits);
             }),
             py::arg("circuit"), py::arg("data_qubits"))
        .def(py::init([](py::object stim_circuit) {
                 std::string circuit_str =
                     py::str(stim_circuit).cast<std::string>();
                 stim::Circuit circuit(circuit_str);
                 return Rerouter(circuit);
             }),
             py::arg("circuit"))
        .def_property_readonly("data_qubits", &Rerouter::get_data_qubits)
        .def(
            "reroute",
            [](const Rerouter &self, size_t observable_index,
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
}

} // namespace qec_loss
