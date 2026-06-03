#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {
class _FastQecLoss {
    public:
        std::string name() const {
            return "FastQecLoss";
        }
        _FastQecLoss() = default;
};
}

PYBIND11_MODULE(_fast_qec_loss, m) {
    m.doc() = "Cpp quantum error correction loss module.";

    py::class_<_FastQecLoss>(m, "FastQecLoss")
        .def(py::init<>())
        .def("name", &_FastQecLoss::name);
}