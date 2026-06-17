#include "lossy_circuit.h"

#include "stim/circuit/circuit.h"
#include "stim/mem/span_ref.h"
#include <fstream>
#include <sstream>

namespace qec_loss {
LossyCircuit::ParseResult
LossyCircuit::parse_circuit(const std::string_view circuit_str) {
    LossyCircuit::ParseResult result;
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
            result.instructions.emplace_back(LossInstruction(line));
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
                result.nominal_circuit.safe_append(op, /* block_fusion=*/true);
                result.instructions.emplace_back(
                    result.nominal_circuit.operations.size() - 1);
            }
        }
    }
    return result;
}

LossyCircuit::LossyCircuit(const std::string_view circuit_str)
    : LossyCircuit(parse_circuit(circuit_str)) {}

LossyCircuit::LossyCircuit(ParseResult parse_result)
    : nominal_circuit(std::move(parse_result.nominal_circuit)),
      instructions(std::move(parse_result.instructions)),
      num_qubits(nominal_circuit.count_qubits()),
      num_measurements(nominal_circuit.count_measurements()),
      num_detectors(nominal_circuit.count_detectors()),
      num_observables(nominal_circuit.count_observables()),
      num_instructions(instructions.size()), rerouter(nominal_circuit) {}

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

std::string LossyCircuit::str() const {
    std::stringstream out;
    for (const auto &instr : instructions) {
        if (std::holds_alternative<LossInstruction>(instr)) {
            out << std::get<LossInstruction>(instr).str() << "\n";
        } else {
            out << nominal_circuit.operations[std::get<size_t>(instr)].str()
                << "\n";
        }
    }
    return out.str();
}

} // namespace qec_loss