#pragma once

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qec_loss {

class PackedF2Matrix {
  public:
    class ProxyRef {
      public:
        ProxyRef(uint64_t &word, size_t bit)
            : word_(word), mask_(1ULL << bit) {}

        operator uint8_t() const { return (word_ & mask_) ? 1 : 0; }

        ProxyRef &operator=(uint8_t val) {
            if (val) {
                word_ |= mask_;
            } else {
                word_ &= ~mask_;
            }
            return *this;
        }

        ProxyRef &operator=(const ProxyRef &other) {
            return *this = static_cast<uint8_t>(other);
        }

      private:
        uint64_t &word_;
        uint64_t mask_;
    };

    PackedF2Matrix(size_t rows, size_t cols);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    ProxyRef operator()(size_t r, size_t c) {
        assert(r < rows_ && c < cols_);
        size_t word_idx = r * words_per_row_ + (c / 64);
        size_t bit_idx = c % 64;
        return ProxyRef(data_[word_idx], bit_idx);
    }

    uint8_t operator()(size_t r, size_t c) const {
        assert(r < rows_ && c < cols_);
        size_t word_idx = r * words_per_row_ + (c / 64);
        size_t bit_idx = c % 64;
        return (data_[word_idx] >> bit_idx) & 1ULL;
    }

    inline void one(size_t r, size_t c) {
        assert(r < rows_ && c < cols_);
        size_t word_idx = r * words_per_row_ + (c / 64);
        size_t bit_idx = c % 64;
        data_[word_idx] |= (1ULL << bit_idx);
    }

    inline void zero(size_t r, size_t c) {
        assert(r < rows_ && c < cols_);
        size_t word_idx = r * words_per_row_ + (c / 64);
        size_t bit_idx = c % 64;
        data_[word_idx] &= ~(1ULL << bit_idx);
    }

    inline void xor_rows(size_t dst, size_t src) {
        assert(dst < rows_ && src < rows_);
        for (size_t w = 0; w < words_per_row_; ++w) {
            data_[dst * words_per_row_ + w] ^= data_[src * words_per_row_ + w];
        }
    }

    inline void xor_bit(size_t r, size_t c) {
        assert(r < rows_ && c < cols_);
        size_t word_idx = r * words_per_row_ + (c / 64);
        size_t bit_idx = c % 64;
        data_[word_idx] ^= (1ULL << bit_idx);
    }

    PackedF2Matrix operator*(const PackedF2Matrix &other) const;
    PackedF2Matrix T() const;
    PackedF2Matrix rref() const;
    size_t rank() const;
    PackedF2Matrix kernel() const;
    PackedF2Matrix solve(const PackedF2Matrix &b) const;

  private:
    size_t rows_;
    size_t cols_;
    size_t words_per_row_;
    std::vector<uint64_t> data_;
};

} // namespace qec_loss
