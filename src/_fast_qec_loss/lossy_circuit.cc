#include "lossy_circuit.h"

#include "stim/circuit/circuit.h"
#include "stim/mem/span_ref.h"
#include <fstream>
#include <sstream>

namespace qec_loss {

LossyCircuit::LossyCircuit(const std::string_view circuit_str) {
    size_t pos = 0;
    while (pos < circuit_str.size()) {
        size_t next_pos = circuit_str.find('\n', pos);
        if (next_pos == std::string_view::npos) {
            next_pos = circuit_str.size();
        }
        std::string_view line = circuit_str.substr(pos, next_pos - pos);
        pos = next_pos + 1;

        // Strip whitespace
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t' ||
                                 line.front() == '\r')) {
            line.remove_prefix(1);
        }
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t' ||
                                 line.back() == '\r')) {
            line.remove_suffix(1);
        }
        if (line.empty()) {
            continue;
        }

        if (line[0] == '#') {
            continue; // Skip comments
        }

        if (line.size() >= 4 && line.substr(0, 4) == "LOSS") {
            instructions.emplace_back(LossInstruction(line));
        } else {
            stim::Circuit tmp_circuit;
            try {
                tmp_circuit.append_from_text(std::string(
                    line)); // to prevent memory issues with string_view
            } catch (const std::exception &e) {
                throw std::runtime_error(std::string("Failed to parse line '") +
                                         std::string(line) + "': " + e.what());
            }
            for (const auto op : tmp_circuit.flattened().operations) {
                nominal_circuit.safe_append(op, true);
                instructions.emplace_back(nominal_circuit.operations.back());

                total_measurements_upper_bound +=
                    op.count_measurement_results();
            }
        }
    }
    num_qubits = nominal_circuit.count_qubits();
}

LossyCircuit
LossyCircuit::from_file(const std::filesystem::path &circuit_path) {
    std::ifstream f(circuit_path);
    if (!f) {
        throw std::runtime_error("Cannot open file: " + circuit_path.string());
    }
    std::stringstream buf;
    buf << f.rdbuf();
    std::string text = buf.str();
    return LossyCircuit(text);
}

void LossyCircuit::to_file(const std::filesystem::path &circuit_path) const {
    std::ofstream f(circuit_path);
    if (!f) {
        throw std::runtime_error("Cannot open file for writing: " +
                                 circuit_path.string());
    }
    f << str();
}

std::string LossyCircuit::str() const { // <- yields garbage
    std::stringstream out;
    for (const auto &instr : instructions) {
        if (std::holds_alternative<LossInstruction>(instr)) {
            out << std::get<LossInstruction>(instr).str() << "\n";
        } else {
            out << std::get<stim::CircuitInstruction>(instr).str() << "\n";
        }
    }
    return out.str();
}

} // namespace qec_loss