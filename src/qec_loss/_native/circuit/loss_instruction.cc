#include "loss_instruction.h"

#include <cctype>
#include <charconv>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace qec_loss {
LossInstruction::LossInstruction(std::vector<uint32_t> targets, double p,
                                 std::string_view tag)
    : p(p),
      // Precompute the integer threshold for fast RNG comparison:
      // Comparing a raw 64-bit random integer to p * (2^64 - 1) is
      // mathematically equivalent to dist(rng) < p, but avoids double-precision
      // float division.
      threshold(static_cast<uint64_t>(
          p * static_cast<double>(std::numeric_limits<uint64_t>::max()))),
      targets(std::move(targets)), tag(tag) {}

LossInstruction::LossInstruction(std::string_view s) {
    constexpr std::string_view prefix = "LOSS";

    if (!s.starts_with(prefix)) {
        throw std::invalid_argument("Invalid instruction format: " +
                                    std::string(s));
    }

    size_t pos = prefix.size();

    // Optional [tag]
    if (pos < s.size() && s[pos] == '[') {
        size_t close = s.find(']', pos);

        if (close == std::string_view::npos) {
            throw std::invalid_argument("Missing closing ']'");
        }

        tag = std::string(s.substr(pos + 1, close - pos - 1));

        pos = close + 1;
    }

    // Parse (p)
    if (pos >= s.size() || s[pos] != '(') {
        throw std::invalid_argument("Expected '('");
    }

    size_t close_paren = s.find(')', pos);

    if (close_paren == std::string_view::npos) {
        throw std::invalid_argument("Missing closing ')'");
    }

    auto p_str = s.substr(pos + 1, close_paren - pos - 1);

    auto [ptr, ec] =
        std::from_chars(p_str.data(), p_str.data() + p_str.size(), p);

    if (ec != std::errc{} || ptr != p_str.data() + p_str.size()) {
        throw std::invalid_argument("Invalid probability: " +
                                    std::string(p_str));
    }

    pos = close_paren + 1;

    // Skip whitespace
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }

    if (pos == s.size()) {
        throw std::invalid_argument("Expected at least one target");
    }

    // Parse targets
    while (pos < s.size()) {
        size_t end = pos;

        while (end < s.size() &&
               !std::isspace(static_cast<unsigned char>(s[end]))) {
            ++end;
        }

        auto token = s.substr(pos, end - pos);

        uint32_t qubit;

        auto [target_ptr, target_ec] =
            std::from_chars(token.data(), token.data() + token.size(), qubit);

        if (target_ec != std::errc{} ||
            target_ptr != token.data() + token.size()) {
            throw std::invalid_argument("Invalid target '" +
                                        std::string(token) + "'");
        }

        targets.push_back(qubit);

        pos = end;

        while (pos < s.size() &&
               std::isspace(static_cast<unsigned char>(s[pos]))) {
            ++pos;
        }
    }
    // Precompute the integer threshold for fast RNG comparison:
    // Comparing a raw 64-bit random integer to p * (2^64 - 1) is mathematically
    // equivalent to dist(rng) < p, but avoids double-precision float division.
    threshold = static_cast<uint64_t>(
        p * static_cast<double>(std::numeric_limits<uint64_t>::max()));
}
std::string LossInstruction::str() const {
    std::stringstream s;

    s << "LOSS";

    if (!tag.empty()) {
        s << "[" << tag << "]";
    }

    s << "(" << p << ")";

    for (uint32_t target : targets) {
        s << " " << target;
    }

    return s.str();
}
} // namespace qec_loss