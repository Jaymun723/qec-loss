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

// stim::DetectorErrorModel
// combine_circuits_into_dem(const std::vector<stim::Circuit> &circuits,
//                           const std::vector<double> weights);

stim::DetectorErrorModel get_loss_dem(const LossyCircuit &circuit,
                                      const uint32_t qubit,
                                      const LifeSegment &life_segment);
} // namespace qec_loss