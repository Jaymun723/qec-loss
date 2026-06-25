#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pybind11 {
class module_;
using module = module_;
} // namespace pybind11

namespace qec_loss {

class F2Tensor {
  public:
    friend void pybind_f2_tensor(pybind11::module &m);

    // Allocates a new contiguous tensor with the given shape.
    // Memory is strictly zero-initialized (np.zeros equivalent).
    explicit F2Tensor(const std::vector<size_t> &shape);

    // Creates a zero-copy view of an existing tensor's memory.
    F2Tensor(std::shared_ptr<uint8_t> data, std::vector<size_t> shape,
             std::vector<size_t> strides, size_t offset);

    // Returns a new F2Tensor that is a view with reversed axes.
    // O(1) operation: Only manipulates shape and strides.
    F2Tensor T() const;

    // Standard matrix multiplication over GF(2) / F_2.
    // Uses bitwise AND (&) for multiplication and XOR (^) for addition.
    // O(N^3) operation: Allocates and returns a brand new contiguous tensor.
    F2Tensor operator*(const F2Tensor &other) const;

    template <typename... Args> F2Tensor subview(Args... ranges) const {
        static_assert(sizeof...(ranges) > 0, "Must provide at least one range");
        assert(sizeof...(ranges) == shape_.size() &&
               "Dimensionality mismatch in subview()");

        std::pair<size_t, size_t> range_array[] = {
            std::pair<size_t, size_t>(ranges)...};

        std::vector<size_t> new_shape(shape_.size());
        size_t new_offset = offset_;

        for (size_t i = 0; i < sizeof...(ranges); ++i) {
            size_t start = range_array[i].first;
            size_t end = range_array[i].second;

            if (start > end || end > shape_[i]) {
                throw std::out_of_range("Subview range is out of bounds.");
            }

            new_shape[i] = end - start;
            new_offset += start * strides_[i];
        }

        return F2Tensor(data_, new_shape, strides_, new_offset);
    }

    // ---------------------------------------------------------
    // High-Performance Element Access
    // ---------------------------------------------------------
    // Implemented fully in the header using variadic templates.
    // This allows -O3 to completely inline the index calculations,
    // unroll the loop, and avoid any heap allocations.

    template <typename... Args> uint8_t &at(Args... indices) {
        static_assert(sizeof...(indices) > 0,
                      "Must provide at least one index");
        assert(sizeof...(indices) == shape_.size() &&
               "Dimensionality mismatch in .at()");

        // Expand indices into a stack-allocated array (zero heap overhead)
        size_t idx_array[] = {static_cast<size_t>(indices)...};

        size_t flat_index = offset_;
        for (size_t i = 0; i < sizeof...(indices); ++i) {
            flat_index += idx_array[i] * strides_[i];
        }
        return data_.get()[flat_index];
    }

    template <typename... Args> const uint8_t &at(Args... indices) const {
        static_assert(sizeof...(indices) > 0,
                      "Must provide at least one index");
        assert(sizeof...(indices) == shape_.size() &&
               "Dimensionality mismatch in .at()");

        size_t idx_array[] = {static_cast<size_t>(indices)...};

        size_t flat_index = offset_;
        for (size_t i = 0; i < sizeof...(indices); ++i) {
            flat_index += idx_array[i] * strides_[i];
        }
        return data_.get()[flat_index];
    }

    // ---------------------------------------------------------
    // Linear Algebra over GF(2)
    // ---------------------------------------------------------
    F2Tensor rref() const;
    size_t rank() const;
    F2Tensor kernel() const;
    F2Tensor solve(const F2Tensor &b) const;

    const std::vector<size_t> &shape() const { return shape_; }
    const std::vector<size_t> &strides() const { return strides_; }
    size_t offset() const { return offset_; }

    // Returns the raw pointer (useful for pybind11 interoperability)
    uint8_t *data_ptr() const { return data_.get(); }

  private:
    // Shared data buffer. Reference counted so memory is freed
    // only when the base tensor and all its views are destroyed.
    std::shared_ptr<uint8_t> data_;

    std::vector<size_t> shape_;   // Dimensions of the tensor
    std::vector<size_t> strides_; // Element step-size to reach the next element
    size_t offset_;               // Starting index in the shared buffer

    // Computes C-contiguous (row-major) strides from a given shape
    static std::vector<size_t>
    compute_contiguous_strides(const std::vector<size_t> &shape);
};

} // namespace qec_loss