#include "f2_tensor.pybind.h"
#include "f2_tensor.h"
#include "packed_f2_tensor.h"
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

namespace qec_loss {

struct SliceRange {
    size_t start;
    size_t end;
    size_t step;
};

void pybind_f2_tensor(py::module &m) {
    auto make_subview =
        [](const F2Tensor &t,
           const std::vector<SliceRange> &ranges) -> F2Tensor {
        if (ranges.size() != t.shape().size()) {
            throw std::invalid_argument(
                "Subview ranges must match the tensor's dimensionality.");
        }

        std::vector<size_t> new_shape(t.shape().size());
        std::vector<size_t> new_strides(t.strides().size());
        size_t new_offset = t.offset();

        for (size_t i = 0; i < ranges.size(); ++i) {
            size_t start = ranges[i].start;
            size_t end = ranges[i].end;
            size_t step = ranges[i].step;

            if (start > end || end > t.shape()[i]) {
                throw std::out_of_range("Subview range is out of bounds.");
            }
            if (step == 0) {
                throw std::invalid_argument("Step size cannot be 0.");
            }

            size_t length = 0;
            if (end > start) {
                length = (end - start + step - 1) / step;
            }

            new_shape[i] = length;
            new_strides[i] = t.strides()[i] * step;
            new_offset += start * t.strides()[i];
        }

        return F2Tensor(t.data_, new_shape, new_strides, new_offset);
    };

    py::class_<F2Tensor>(m, "F2Tensor", py::buffer_protocol())
        .def(py::init<const std::vector<size_t> &>(), py::arg("shape"))
        .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<uint8_t>::value &&
                info.format != "b" && info.format != "?") {
                throw std::invalid_argument("Buffer must be uint8 or bool type.");
            }

            std::vector<size_t> shape;
            std::vector<size_t> strides;
            for (auto s : info.shape) {
                shape.push_back(static_cast<size_t>(s));
            }
            for (auto str : info.strides) {
                strides.push_back(static_cast<size_t>(str));
            }

            py::object parent = b;
            std::shared_ptr<uint8_t> data(static_cast<uint8_t*>(info.ptr), [parent](uint8_t*) {
                py::gil_scoped_acquire gil;
            });

            return F2Tensor(data, shape, strides, 0);
        }), py::arg("buffer"))
        .def_property_readonly("shape", &F2Tensor::shape)
        .def_property_readonly("strides", &F2Tensor::strides)
        .def_property_readonly("offset", &F2Tensor::offset)
        .def_property_readonly("T", &F2Tensor::T)
        .def("subview", [make_subview](const F2Tensor &t, const py::list &ranges_list) {
            std::vector<SliceRange> ranges;
            for (auto item : ranges_list) {
                if (py::isinstance<py::tuple>(item)) {
                    py::tuple tup = py::cast<py::tuple>(item);
                    if (tup.size() == 2) {
                        ranges.push_back({py::cast<size_t>(tup[0]), py::cast<size_t>(tup[1]), 1});
                    } else if (tup.size() == 3) {
                        ranges.push_back({py::cast<size_t>(tup[0]), py::cast<size_t>(tup[1]), py::cast<size_t>(tup[2])});
                    } else {
                        throw std::invalid_argument("Slice range tuple must be size 2 or 3");
                    }
                } else {
                    throw std::invalid_argument("Slice range must be a tuple");
                }
            }
            return make_subview(t, ranges);
        }, py::arg("ranges"))
        .def("rref", &F2Tensor::rref)
        .def("rank", &F2Tensor::rank)
        .def("kernel", &F2Tensor::kernel)
        .def("solve", &F2Tensor::solve, py::arg("b"))
        .def("__matmul__", &F2Tensor::operator*)
        .def("__str__",
             [](const F2Tensor &t) {
                 py::object np_arr =
                     py::module_::import("numpy").attr("asarray")(t);
                 return py::str(np_arr).cast<std::string>();
             })
        .def("__repr__",
             [](const F2Tensor &t) {
                 py::object np_arr =
                     py::module_::import("numpy").attr("asarray")(t);
                 return "F2Tensor(" + py::repr(np_arr).cast<std::string>() +
                        ")";
             })
        .def("to_numpy",
             [](py::object self) {
                 return py::module_::import("numpy").attr("asarray")(self);
             })
        .def("__getitem__",
             [make_subview](const F2Tensor &t, const py::object &key) -> py::object {
                 py::tuple key_tuple;
                 if (py::isinstance<py::tuple>(key)) {
                     key_tuple = py::cast<py::tuple>(key);
                 } else {
                     key_tuple = py::make_tuple(key);
                 }

                 if (key_tuple.size() != t.shape().size()) {
                     throw std::invalid_argument(
                         "Dimensionality mismatch in index/slice");
                 }

                 bool is_slice_access = false;
                 for (const auto &item : key_tuple) {
                     if (py::isinstance<py::slice>(item)) {
                         is_slice_access = true;
                         break;
                     }
                 }

                 if (is_slice_access) {
                     std::vector<SliceRange> ranges;
                     ranges.reserve(t.shape().size());
                     for (size_t i = 0; i < t.shape().size(); ++i) {
                         auto item = key_tuple[i];
                         if (py::isinstance<py::slice>(item)) {
                             py::slice s = py::cast<py::slice>(item);
                             size_t start, stop, step, slice_length;
                             if (!s.compute(t.shape()[i], &start, &stop, &step,
                                            &slice_length)) {
                                 throw py::error_already_set();
                             }
                             if (step <= 0) {
                                 throw std::invalid_argument(
                                     "Only positive step sizes are supported");
                             }
                             ranges.push_back({start, stop, static_cast<size_t>(step)});
                         } else {
                             int64_t val = py::cast<int64_t>(item);
                             if (val < 0) {
                                 val += t.shape()[i];
                             }
                             if (val < 0 ||
                                 static_cast<size_t>(val) >= t.shape()[i]) {
                                 throw std::out_of_range("Index out of bounds");
                             }
                             ranges.push_back({static_cast<size_t>(val), static_cast<size_t>(val + 1), 1});
                         }
                     }
                     return py::cast(make_subview(t, ranges));
                 } else {
                     // Element access
                     size_t flat_index = t.offset();
                     for (size_t i = 0; i < t.shape().size(); ++i) {
                         int64_t val = py::cast<int64_t>(key_tuple[i]);
                         if (val < 0) {
                             val += t.shape()[i];
                         }
                         if (val < 0 ||
                             static_cast<size_t>(val) >= t.shape()[i]) {
                             throw std::out_of_range("Index out of bounds");
                         }
                         flat_index += val * t.strides()[i];
                     }
                     return py::cast(t.data_ptr()[flat_index]);
                 }
             })
        .def(
            "__setitem__",
            [make_subview](F2Tensor &t, const py::object &key, const py::object &value) {
                py::tuple key_tuple;
                if (py::isinstance<py::tuple>(key)) {
                    key_tuple = py::cast<py::tuple>(key);
                } else {
                    key_tuple = py::make_tuple(key);
                }

                if (key_tuple.size() != t.shape().size()) {
                    throw std::invalid_argument(
                        "Dimensionality mismatch in index/slice");
                }

                bool is_slice_access = false;
                for (const auto &item : key_tuple) {
                    if (py::isinstance<py::slice>(item)) {
                        is_slice_access = true;
                        break;
                    }
                }

                if (is_slice_access) {
                    // Get the subview
                    std::vector<SliceRange> ranges;
                    ranges.reserve(t.shape().size());
                    for (size_t i = 0; i < t.shape().size(); ++i) {
                        auto item = key_tuple[i];
                        if (py::isinstance<py::slice>(item)) {
                            py::slice s = py::cast<py::slice>(item);
                            size_t start, stop, step, slice_length;
                            if (!s.compute(t.shape()[i], &start, &stop, &step,
                                           &slice_length)) {
                                throw py::error_already_set();
                            }
                            if (step <= 0) {
                                throw std::invalid_argument(
                                    "Only positive step sizes are supported");
                            }
                            ranges.push_back({start, stop, static_cast<size_t>(step)});
                         } else {
                            int64_t val = py::cast<int64_t>(item);
                            if (val < 0) {
                                val += t.shape()[i];
                            }
                            if (val < 0 ||
                                static_cast<size_t>(val) >= t.shape()[i]) {
                                throw std::out_of_range("Index out of bounds");
                            }
                            ranges.push_back({static_cast<size_t>(val), static_cast<size_t>(val + 1), 1});
                        }
                    }
                    F2Tensor sub = make_subview(t, ranges);

                    try {
                        // Try to cast to uint8_t (0 or 1)
                        uint8_t val = py::cast<uint8_t>(value);
                        std::vector<size_t> current_indices(sub.shape().size(),
                                                            0);
                        auto fill_helper = [&](auto &self, size_t dim) -> void {
                            if (dim == sub.shape().size()) {
                                size_t flat_idx = sub.offset();
                                for (size_t d = 0; d < sub.shape().size();
                                     ++d) {
                                    flat_idx +=
                                        current_indices[d] * sub.strides()[d];
                                }
                                sub.data_ptr()[flat_idx] = val;
                                return;
                            }
                            for (size_t idx = 0; idx < sub.shape()[dim];
                                 ++idx) {
                                current_indices[dim] = idx;
                                self(self, dim + 1);
                            }
                        };
                        fill_helper(fill_helper, 0);
                    } catch (const py::cast_error &) {
                        // Try to cast to F2Tensor
                        F2Tensor val_tensor = py::cast<F2Tensor>(value);
                        if (val_tensor.shape() != sub.shape()) {
                            throw std::invalid_argument(
                                "Shape mismatch in slice assignment");
                        }
                        std::vector<size_t> current_indices(sub.shape().size(),
                                                            0);
                        auto copy_helper = [&](auto &self, size_t dim) -> void {
                            if (dim == sub.shape().size()) {
                                size_t dest_flat = sub.offset();
                                size_t src_flat = val_tensor.offset();
                                for (size_t d = 0; d < sub.shape().size();
                                     ++d) {
                                    dest_flat +=
                                        current_indices[d] * sub.strides()[d];
                                    src_flat += current_indices[d] *
                                                val_tensor.strides()[d];
                                }
                                sub.data_ptr()[dest_flat] =
                                    val_tensor.data_ptr()[src_flat];
                                return;
                            }
                            for (size_t idx = 0; idx < sub.shape()[dim];
                                 ++idx) {
                                current_indices[dim] = idx;
                                self(self, dim + 1);
                            }
                        };
                        copy_helper(copy_helper, 0);
                    }
                } else {
                    // Element access
                    size_t flat_index = t.offset();
                    for (size_t i = 0; i < t.shape().size(); ++i) {
                        int64_t val = py::cast<int64_t>(key_tuple[i]);
                        if (val < 0) {
                            val += t.shape()[i];
                        }
                        if (val < 0 ||
                            static_cast<size_t>(val) >= t.shape()[i]) {
                            throw std::out_of_range("Index out of bounds");
                        }
                        flat_index += val * t.strides()[i];
                    }
                    t.data_ptr()[flat_index] = py::cast<uint8_t>(value);
                }
            })
        .def_buffer([](F2Tensor &t) -> py::buffer_info {
            std::vector<ssize_t> byte_strides;
            for (auto stride : t.strides()) {
                byte_strides.push_back(
                    static_cast<ssize_t>(stride * sizeof(uint8_t)));
            }
            std::vector<ssize_t> shape;
            for (auto s : t.shape()) {
                shape.push_back(static_cast<ssize_t>(s));
            }
            return py::buffer_info(t.data_ptr() + t.offset(), sizeof(uint8_t),
                                   py::format_descriptor<uint8_t>::format(),
                                   t.shape().size(), shape, byte_strides);
        });

    py::class_<PackedF2Tensor>(m, "PackedF2Tensor")
        .def(py::init<size_t, size_t>(), py::arg("rows"), py::arg("cols"))
        .def_property_readonly("rows", &PackedF2Tensor::rows)
        .def_property_readonly("cols", &PackedF2Tensor::cols)
        .def("get", &PackedF2Tensor::get, py::arg("r"), py::arg("c"))
        .def("set", &PackedF2Tensor::set, py::arg("r"), py::arg("c"), py::arg("val"))
        .def("rref", &PackedF2Tensor::rref)
        .def("rank", &PackedF2Tensor::rank)
        .def("kernel", &PackedF2Tensor::kernel)
        .def("solve", &PackedF2Tensor::solve, py::arg("b"))
        .def("__matmul__", &PackedF2Tensor::operator*);
}

} // namespace qec_loss
