#include "packed_f2_matrix.h"
#include <algorithm>

namespace qec_loss {

PackedF2Matrix::PackedF2Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), words_per_row_((cols + 63) / 64) {
    data_.resize(rows_ * words_per_row_, 0);
}

PackedF2Matrix PackedF2Matrix::operator*(const PackedF2Matrix &other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument(
            "Dimension mismatch in matrix multiplication.");
    }
    PackedF2Matrix result(rows_, other.cols_);
    size_t words_per_row = other.words_per_row_;

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if ((*this)(i, j) == 1) {
                for (size_t w = 0; w < words_per_row; ++w) {
                    result.data_[i * words_per_row + w] ^=
                        other.data_[j * words_per_row + w];
                }
            }
        }
    }
    return result;
}

PackedF2Matrix PackedF2Matrix::T() const {
    PackedF2Matrix result(cols_, rows_);
    for (size_t r = 0; r < rows_; ++r) {
        for (size_t c = 0; c < cols_; ++c) {
            if ((*this)(r, c)) {
                result(c, r) = 1;
            }
        }
    }
    return result;
}

PackedF2Matrix PackedF2Matrix::rref() const {
    PackedF2Matrix result = *this;
    size_t lead = 0;
    for (size_t r = 0; r < rows_; ++r) {
        if (lead >= cols_) {
            break;
        }
        size_t i = r;
        while (result(i, lead) == 0) {
            i++;
            if (i == rows_) {
                i = r;
                lead++;
                if (lead == cols_) {
                    return result;
                }
            }
        }
        if (i != r) {
            for (size_t w = 0; w < words_per_row_; ++w) {
                std::swap(result.data_[i * words_per_row_ + w],
                          result.data_[r * words_per_row_ + w]);
            }
        }
        for (size_t row = 0; row < rows_; ++row) {
            if (row != r && result(row, lead) == 1) {
                result.xor_rows(row, r);
            }
        }
        lead++;
    }
    return result;
}

size_t PackedF2Matrix::rank() const {
    PackedF2Matrix rref_mat = this->rref();
    size_t r = 0;
    for (size_t i = 0; i < rows_; ++i) {
        bool all_zero = true;
        for (size_t w = 0; w < words_per_row_; ++w) {
            if (rref_mat.data_[i * words_per_row_ + w] != 0) {
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

PackedF2Matrix PackedF2Matrix::kernel() const {
    PackedF2Matrix rref_mat = this->rref();

    std::vector<int> row_to_pivot(rows_, -1);
    std::vector<bool> is_pivot(cols_, false);
    for (size_t r = 0; r < rows_; ++r) {
        for (size_t c = 0; c < cols_; ++c) {
            if (rref_mat(r, c) == 1) {
                row_to_pivot[r] = static_cast<int>(c);
                is_pivot[c] = true;
                break;
            }
        }
    }

    std::vector<size_t> free_cols;
    for (size_t c = 0; c < cols_; ++c) {
        if (!is_pivot[c]) {
            free_cols.push_back(c);
        }
    }

    size_t K = free_cols.size();
    PackedF2Matrix kern(K, cols_);

    for (size_t k = 0; k < K; ++k) {
        size_t free_col = free_cols[k];
        kern(k, free_col) = 1;
        for (size_t r = 0; r < rows_; ++r) {
            int p = row_to_pivot[r];
            if (p != -1) {
                kern(k, static_cast<size_t>(p)) = rref_mat(r, free_col);
            }
        }
    }
    return kern;
}

PackedF2Matrix PackedF2Matrix::solve(const PackedF2Matrix &b) const {
    if (b.rows_ != rows_ || b.cols_ != 1) {
        throw std::invalid_argument(
            "b must be a column vector of shape rows x 1.");
    }

    PackedF2Matrix aug(rows_, cols_ + 1);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            aug(i, j) = (*this)(i, j);
        }
        aug(i, cols_) = b(i, 0);
    }

    PackedF2Matrix aug_rref = aug.rref();

    for (size_t i = 0; i < rows_; ++i) {
        bool all_zero_A = true;
        for (size_t j = 0; j < cols_; ++j) {
            if (aug_rref(i, j) != 0) {
                all_zero_A = false;
                break;
            }
        }
        if (all_zero_A && aug_rref(i, cols_) == 1) {
            throw std::runtime_error("System is inconsistent (no solution).");
        }
    }

    PackedF2Matrix x(cols_, 1);
    for (size_t r = 0; r < rows_; ++r) {
        for (size_t c = 0; c < cols_; ++c) {
            if (aug_rref(r, c) == 1) {
                x(c, 0) = aug_rref(r, cols_);
                break;
            }
        }
    }

    return x;
}

std::string PackedF2Matrix::str() const {
    std::string result;
    for (size_t r = 0; r < rows(); ++r) {
        for (size_t c = 0; c < cols(); ++c) {
            result += std::to_string((*this)(r, c));
            if (c < cols() - 1) {
                result += " ";
            }
        }
        result += "\n";
    }
    return result;
}

} // namespace qec_loss
