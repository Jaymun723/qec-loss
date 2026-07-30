#pragma once
/// @file Following Baranes et al. (https://arxiv.org/abs/2502.20558v3)
/// (Appendices C and D), the effective DEM associated with an observed loss
/// event is obtained by averaging over all compatible loss locations. Let
/// $\ell_i$ denote a candidate loss location and $L$ the observation that
/// the qubit is missing. If the loss probability at location $i$ is $q_i$,
/// the posterior weight assigned to that location is
///
/// \[w_i = P(\ell_i |
///           L) = \frac{q_i \prod_{j < i}(1 - q_j)} {1 - \prod_j(1 - q_j)}.
/// \]
///
/// This corresponds to the probability that the qubit survives all previous
/// loss opportunities, is lost at location $i$, and is known to have been lost
/// somewhere before measurement.
///
/// For each candidate loss location, a conditional loss circuit is constructed
/// by removing the gates acting on the lost qubit after the loss event and
/// inserting a `DEPOLARIZE1(0.75)` immediately before measurement to generate
/// the corresponding supercheck. A detector error model (DEM) is then extracted
/// from this circuit using `detector_error_model(allow_gauge_detectors=True)`
/// The final loss DEM is obtained by averaging the conditional DEMs according
/// to the posterior weights $w_i$. If a detector signature $S$ appears in the
/// DEM associated with location $\ell_i$ with probability $e_i(S)$, its
/// probability in the merged DEM is
///
/// \[e_{\mathrm{eff}}(S) =
/// \sum_i w_i\, e_i(S).
/// \]

#include "../circuit/lossy_circuit.h"
#include "life_cycle.h"
#include "stim/dem/detector_error_model.h"

namespace qec_loss {

stim::DetectorErrorModel
combine_circuits_into_dem(const std::vector<stim::Circuit> &circuits,
                          const std::vector<double> weights);

/// Weighted merge of already-extracted DEMs: error mechanisms with identical
/// targets get their probabilities summed after scaling by `weights`. Mirrors
/// `qec_loss.combine_dems` (Python) and the inner loop of
/// `combine_circuits_into_dem`.
stim::DetectorErrorModel
combine_dems_native(const std::vector<stim::DetectorErrorModel> &dems,
                    const std::vector<double> &weights);

/// Detector error model allowing undeterministic detectors AND observables.
/// C++ port of `surface_boss.utils.get_detector_error_model` with
/// `decompose_errors=False`: each OBSERVABLE_INCLUDE is converted to a
/// temporary DETECTOR (so undeterministic observables are tolerated via
/// `allow_gauge_detectors`), and the resulting extra detectors are mapped
/// back to logical observables afterwards.
stim::DetectorErrorModel
circuit_to_dem_gauge_observables(const stim::Circuit &circuit);

/// Effective DEM for a life segment without observable rerouting, using the
/// gauge-observable trick of `circuit_to_dem_gauge_observables`. This matches
/// the semantics of the Python reference pipeline
/// (`get_loss_rewritten_circuits` + `surface_boss.utils.get_detector_error_model`
/// + `qec_loss.combine_dems`) used by the delayed erasure decoder.
stim::DetectorErrorModel
get_loss_segment_dem(const LossyCircuit &circuit,
                     const LifeSegment &life_segment);

std::vector<stim::DetectorErrorModel>
get_loss_segment_dems(const LossyCircuit &circuit,
                      const std::vector<LifeSegment> &life_segments);

std::vector<stim::Circuit> get_loss_rewritten_circuits(
    const LossyCircuit &circuit, const std::vector<uint32_t> &lost_qubits,
    const LifeSegment &life_segment, bool optimize_rerouting);

std::vector<stim::Circuit>
get_loss_rewritten_circuits(const LossyCircuit &circuit,
                            const LifeSegment &life_segment);

stim::DetectorErrorModel get_loss_dem(const LossyCircuit &circuit,
                                      const std::vector<uint32_t> &lost_qubits,
                                      const LifeSegment &life_segment,
                                      bool optimize_rerouting);
} // namespace qec_loss