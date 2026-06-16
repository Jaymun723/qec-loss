#include "f2_tensor.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace qec_loss {

std::vector<size_t>
F2Tensor::compute_contiguous_strides(const std::vector<size_t> &shape) {
    std::vector<size_t> strides(shape.size());
    if (shape.empty())
        return strides;

    // Row-major (C-contiguous) strides:
    // The last dimension always has a stride of 1.
    // Preceding dimensions are the product of the shapes of the dimensions that
    // follow.
    size_t current_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = current_stride;
        current_stride *= shape[i];
    }
    return strides;
}

F2Tensor::F2Tensor(const std::vector<size_t> &shape)
    : shape_(shape), strides_(compute_contiguous_strides(shape)), offset_(0) {
    // Calculate total number of elements
    size_t total_elements = std::accumulate(shape.begin(), shape.end(), 1ULL,
                                            std::multiplies<size_t>());

    // Allocate zero-initialized memory.
    // Using new uint8_t[...]() guarantees elements are zeroed out We use a
    // custom deleter to ensure the array is properly destructed when the
    // shared_ptr ref-count hits 0.
    data_ = std::shared_ptr<uint8_t>(new uint8_t[total_elements](),
                                     std::default_delete<uint8_t[]>());
}

F2Tensor::F2Tensor(std::shared_ptr<uint8_t> data, std::vector<size_t> shape,
                   std::vector<size_t> strides, size_t offset)
    : data_(std::move(data)), shape_(std::move(shape)),
      strides_(std::move(strides)), offset_(offset) {}

F2Tensor F2Tensor::T() const {
    std::vector<size_t> new_shape = shape_;
    std::vector<size_t> new_strides = strides_;

    // Reversing the axes creates a transposed view in O(1) time
    std::reverse(new_shape.begin(), new_shape.end());
    std::reverse(new_strides.begin(), new_strides.end());

    return F2Tensor(data_, new_shape, new_strides, offset_);
}


F2Tensor F2Tensor::operator*(const F2Tensor &other) const {
    // Standard matrix multiplication is only defined for 2D tensors.
    if (shape_.size() != 2 || other.shape().size() != 2) {
        throw std::invalid_argument(
            "Matrix multiplication requires both tensors to be 2-dimensional.");
    }

    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t K_other = other.shape()[0];
    size_t N = other.shape()[1];

    if (K != K_other) {
        throw std::invalid_argument(
            "Inner dimensions must match for matrix multiplication.");
    }

    // Create the output tensor (automatically zero-initialized)
    F2Tensor result({M, N});

    // ---------------------------------------------------------
    // F2 Matrix Multiplication
    // ---------------------------------------------------------
    // Because we defined .at() using variadic templates in the header,
    // passing two arguments here is perfectly safe and highly optimized.
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            uint8_t dot_product = 0;
            for (size_t k = 0; k < K; ++k) {
                // In F2:
                // Addition is XOR (^)
                // Multiplication is AND (&)
                dot_product ^= (this->at(i, k) & other.at(k, j));
            }
            result.at(i, j) = dot_product;
        }
    }

    return result;
}

F2Tensor F2Tensor::rref() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("RREF is only defined for 2D tensors.");
    }

    size_t M = shape_[0];
    size_t N = shape_[1];

    F2Tensor result({M, N});
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result.at(i, j) = this->at(i, j);
        }
    }

    size_t lead = 0;
    for (size_t r = 0; r < M; ++r) {
        if (lead >= N) {
            break;
        }
        size_t i = r;
        while (result.at(i, lead) == 0) {
            i++;
            if (i == M) {
                i = r;
                lead++;
                if (lead == N) {
                    return result;
                }
            }
        }
        if (i != r) {
            for (size_t col = 0; col < N; ++col) {
                std::swap(result.at(i, col), result.at(r, col));
            }
        }
        for (size_t row = 0; row < M; ++row) {
            if (row != r && result.at(row, lead) == 1) {
                for (size_t col = 0; col < N; ++col) {
                    result.at(row, col) ^= result.at(r, col);
                }
            }
        }
        lead++;
    }
    return result;
}

size_t F2Tensor::rank() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Rank is only defined for 2D tensors.");
    }
    F2Tensor rref_mat = this->rref();
    size_t M = rref_mat.shape()[0];
    size_t N = rref_mat.shape()[1];
    size_t r = 0;
    for (size_t i = 0; i < M; ++i) {
        bool all_zero = true;
        for (size_t j = 0; j < N; ++j) {
            if (rref_mat.at(i, j) != 0) {
                all_zero = false;
                break;
            }
        }
        if (!all_zero) {
            r++;
        }
    }
    return r;
}

F2Tensor F2Tensor::kernel() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Kernel is only defined for 2D tensors.");
    }
    F2Tensor rref_mat = this->rref();
    size_t M = rref_mat.shape()[0];
    size_t N = rref_mat.shape()[1];

    std::vector<int> row_to_pivot(M, -1);
    std::vector<bool> is_pivot(N, false);
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            if (rref_mat.at(r, c) == 1) {
                row_to_pivot[r] = static_cast<int>(c);
                is_pivot[c] = true;
                break;
            }
        }
    }

    std::vector<size_t> free_cols;
    for (size_t c = 0; c < N; ++c) {
        if (!is_pivot[c]) {
            free_cols.push_back(c);
        }
    }

    size_t K = free_cols.size();
    F2Tensor kern({K, N});

    for (size_t k = 0; k < K; ++k) {
        size_t free_col = free_cols[k];
        kern.at(k, free_col) = 1;
        for (size_t r = 0; r < M; ++r) {
            int p = row_to_pivot[r];
            if (p != -1) {
                kern.at(k, static_cast<size_t>(p)) = rref_mat.at(r, free_col);
            }
        }
    }
    return kern;
}

F2Tensor F2Tensor::solve(const F2Tensor &b) const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Solve is only defined for a 2D system matrix.");
    }
    if (b.shape().size() != 1) {
        throw std::invalid_argument("Right-hand side b must be a 1D tensor.");
    }
    size_t M = shape_[0];
    size_t N = shape_[1];
    if (b.shape()[0] != M) {
        throw std::invalid_argument("Dimension mismatch between system matrix A and vector b.");
    }

    F2Tensor aug({M, N + 1});
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            aug.at(i, j) = this->at(i, j);
        }
        aug.at(i, N) = b.at(i);
    }

    F2Tensor aug_rref = aug.rref();

    for (size_t i = 0; i < M; ++i) {
        bool all_zero_A = true;
        for (size_t j = 0; j < N; ++j) {
            if (aug_rref.at(i, j) != 0) {
                all_zero_A = false;
                break;
            }
        }
        if (all_zero_A && aug_rref.at(i, N) == 1) {
            throw std::runtime_error("System is inconsistent (no solution).");
        }
    }

    F2Tensor x({N});
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            if (aug_rref.at(r, c) == 1) {
                x.at(c) = aug_rref.at(r, N);
                break;
            }
        }
    }

    return x;
}

} // namespace qec_loss