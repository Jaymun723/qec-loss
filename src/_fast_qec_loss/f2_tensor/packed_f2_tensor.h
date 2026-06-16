#pragma once

#include <cassert>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <utility>

namespace qec_loss {

class PackedF2Tensor {
  public:
    PackedF2Tensor(size_t rows, size_t cols);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    uint8_t get(size_t r, size_t c) const {
        assert(r < rows_ && c < cols_);
        size_t word_idx = r * words_per_row_ + (c / 64);
        size_t bit_idx = c % 64;
        return (data_[word_idx] >> bit_idx) & 1ULL;
    }

    void set(size_t r, size_t c, uint8_t val) {
        assert(r < rows_ && c < cols_);
        size_t word_idx = r * words_per_row_ + (c / 64);
        size_t bit_idx = c % 64;
        if (val) {
            data_[word_idx] |= (1ULL << bit_idx);
        } else {
            data_[word_idx] &= ~(1ULL << bit_idx);
        }
    }

    PackedF2Tensor operator*(const PackedF2Tensor &other) const;
    PackedF2Tensor rref() const;
    size_t rank() const;
    PackedF2Tensor kernel() const;
    PackedF2Tensor solve(const PackedF2Tensor &b) const;

  private:
    size_t rows_;
    size_t cols_;
    size_t words_per_row_;
    std::vector<uint64_t> data_;
};

} // namespace qec_loss
