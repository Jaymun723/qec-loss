#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loss_instruction.h"

namespace py = pybind11;

PYBIND11_MODULE(_fast_qec_loss, m) {
    m.doc() = "Cpp quantum error correction loss module.";

    py::class_<qec_loss::LossInstruction>(m, "LossInstruction")
        .def(py::init<std::string_view>())
        .def(py::init<std::vector<uint32_t>, double, std::string_view>())
        .def(py::init<std::vector<uint32_t>, double>()) // For convenience
        .def_readonly("p", &qec_loss::LossInstruction::p)
        .def_readonly("targets", &qec_loss::LossInstruction::targets)
        .def_readonly("tag", &qec_loss::LossInstruction::tag)
        .def("__str__", [](const qec_loss::LossInstruction &self) {
            std::string result = "LOSS";

            if (!self.tag.empty()) {
                result += "[" + self.tag + "]";
            }

            result += "(" + std::to_string(self.p) + ")";

            for (uint32_t t : self.targets) {
                result += " " + std::to_string(t);
            }

            return result;
        });
}