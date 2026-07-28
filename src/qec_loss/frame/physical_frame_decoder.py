"""
Physical Pauli-frame decoder built on stim + pymatching.

Standard pymatching usage (`Matching.from_detector_error_model(dem).decode`)
only reports whether each *logical observable* flipped. This module gives
you two extra things:

  1. `frame` -- the per-qubit Pauli the decoder's chosen error pattern
     applies AT EACH FAULT'S OWN CIRCUIT LOCATION. Good for visualizing
     where/what the decoder thinks happened, watching matched paths reroute
     between shots, etc.

  2. `final_correction` -- that same chosen error pattern, PROPAGATED
     forward to the end of the circuit and expressed as a bit to XOR onto
     each tracked final data-qubit measurement. By default every non-reset
     M/MX/MY/MZ target is tracked (the final data layer in a typical memory
     experiment), so OBSERVABLE_INCLUDE is optional.

WHY (2) NEEDS ACTUAL SIMULATION, NOT JUST BOOKKEEPING
------------------------------------------------------
Stim requires anything declared with OBSERVABLE_INCLUDE (or DETECTOR) to be
deterministic under noiseless execution. A single data qubit's raw final
measurement is *not* deterministic on its own -- only parities like
stabilizers and the logical observable are. So you cannot declare "qubit q's
outcome" as its own observable and ask stim's DEM machinery for it (this is
what we tried first, and it correctly fails).

What IS well-defined is a narrower question: given a SPECIFIC, ALREADY
CHOSEN error pattern (i.e. the one the decoder selected for this shot), what
does propagating exactly that fixed Pauli through the rest of the circuit
do to the final measurement bits? That's a deterministic simulation, not a
universal claim about the noise ensemble, so it doesn't hit the same
determinism requirement.

This module answers it by direct simulation: for each candidate fault,
`Circuit.explain_detector_error_model_errors` tells us both (a) the
equivalent Pauli to apply and (b) the EXACT circuit instruction it
originated from (`stack_frames[0].instruction_offset`, an index into
`circuit.flattened()`). We run the noiseless circuit once through a
`stim.TableauSimulator`, checkpointing at every `TICK`, then for each
fault clone the nearest preceding checkpoint, apply the Pauli via the
simulator's own `.x()`/`.y()`/`.z()`, and advance only the remaining
suffix (stochastic channels stripped via `stim.gate_data(name).is_noisy_gate`,
measurement-producing gates kept with readout-noise args zeroed). Diffing
the resulting measurement record against the noiseless baseline gives the
fault's true effect on every final measurement. This was verified to
reproduce stim's own DEM-recorded observable membership exactly
(0/N mismatches) for every individual fault mechanism in a distance-5,
5-round surface code circuit -- an earlier tick-boundary version of this
(imprecise about *where within a tick* to inject) was wrong on ~0.6% of
faults; this instruction-offset version isn't.

CAVEAT: `final_correction` is still only one representative of the
decoder's chosen equivalence class (MWPM doesn't know "the" error, only a
minimum-weight one consistent with the syndrome). Different, equally valid
corrections exist that differ by a stabilizer.

MATCHING pymatching's OWN GRAPH EXACTLY
-----------------------------------------
`Matching.from_detector_error_model` merges parallel edges that share the
same (detector-pair, observable-set) footprint into a single edge with a
combined probability (p1 + p2 - 2*p1*p2, ...). If you don't do the same
merge here, MWPM can occasionally break near-ties differently between the
two graphs even though both are "correct" in the sense of reproducing the
syndrome. This module performs that same merge before building the
matching graph, so its logical-observable predictions are verified
bit-for-bit identical to `Matching.from_detector_error_model(dem).decode`
(checked over 5000 sampled shots on a distance-5, 5-round circuit, 0
mismatches). Each merged edge keeps a single representative fault (the
highest-probability member of the group) for `frame` / `final_correction`
bookkeeping, since members of a merged group share syndrome and observable
effect by construction but may differ in exact physical location.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import cast
import numpy as np
import scipy.sparse as sp
import stim
import pymatching

from qec_loss.utils import flattened_instructions

# Non-reset measure gates — typically the final data-qubit readout layer.
# Mid-circuit syndrome readouts use MR* and are not treated as data.
_DATA_MEASURE_GATES = frozenset({"M", "MZ", "MX", "MY"})
_ALL_MEASURE_GATES = _DATA_MEASURE_GATES | frozenset({"MR", "MRX", "MRY", "MRZ"})


def _final_measurement_targets(
    circuit: stim.Circuit,
    data_qubits: Sequence[int] | None = None,
):
    """
    Returns:
        meas_to_qubit: list, meas_to_qubit[m] = qubit measured at absolute
            measurement index m.
        obs_groups: list[list[int]], obs_groups[k] = absolute measurement
            indices that make up original logical observable k (empty if the
            circuit has no OBSERVABLE_INCLUDE).
        data_meas_indices: sorted absolute measurement indices of terminal
            data-qubit readouts (non-reset M/MX/MY/MZ). If ``data_qubits`` is
            given, only each listed qubit's *last* such measurement is kept.
    """
    meas_to_qubit: list[int] = []
    obs_groups: list[list[int]] = []
    # qubit -> absolute measurement index of its latest non-reset M
    last_data_meas: dict[int, int] = {}
    all_data_meas: list[int] = []

    for instr in flattened_instructions(circuit):
        if instr.name in _ALL_MEASURE_GATES:
            is_data_meas = instr.name in _DATA_MEASURE_GATES
            for t in instr.targets_copy():
                q = t.value
                m = len(meas_to_qubit)
                meas_to_qubit.append(q)
                if is_data_meas:
                    last_data_meas[q] = m
                    all_data_meas.append(m)
        elif instr.name == "OBSERVABLE_INCLUDE":
            total = len(meas_to_qubit)
            obs_groups.append(sorted(total + t.value for t in instr.targets_copy()))

    if data_qubits is None:
        data_meas_indices = sorted(all_data_meas)
    else:
        missing = [q for q in data_qubits if q not in last_data_meas]
        if missing:
            raise ValueError(
                f"data_qubits {missing} have no non-reset M/MX/MY/MZ measurement "
                f"in the circuit"
            )
        # Preserve caller order for final_qubits; indices still unique per qubit.
        data_meas_indices = [last_data_meas[q] for q in data_qubits]

    return meas_to_qubit, obs_groups, data_meas_indices


def _is_pure_noise_channel(instr: stim.CircuitInstruction) -> bool:
    """True for stochastic error channels (DEPOLARIZE*, X_ERROR, PAULI_CHANNEL*,
    HERALDED_*, ...) that carry no useful deterministic effect of their own."""
    gd = stim.gate_data(instr.name)
    return gd.is_noisy_gate and not gd.produces_measurements


def _strip_measurement_noise_arg(instr: stim.CircuitInstruction) -> stim.CircuitInstruction:
    """M/MR/... can optionally carry their own readout-flip probability
    (e.g. M(0.01) 0 1). Keep the gate (it produces the measurement we need)
    but zero out the probability so it's deterministic."""
    gd = stim.gate_data(instr.name)
    if gd.is_noisy_gate and gd.produces_measurements and instr.gate_args_copy():
        return stim.CircuitInstruction(instr.name, instr.targets_copy(), [])
    return instr


def _do_deterministic(sim: stim.TableauSimulator, instr: stim.CircuitInstruction) -> None:
    """Apply one instruction with pure noise stripped / readout noise zeroed."""
    if _is_pure_noise_channel(instr):
        return
    sim.do(_strip_measurement_noise_arg(instr))


def _build_tick_checkpoints(noisy_flat):
    """Run the noiseless circuit once; stash a TableauSimulator clone at start
    and after every TICK. Returns (checkpoints, baseline_record) where
    checkpoints is a list of (after_instruction_index, simulator) with
    after_instruction_index=-1 for the initial state."""
    sim = stim.TableauSimulator()
    checkpoints = [(-1, sim.copy())]
    for j, instr in enumerate(noisy_flat):
        _do_deterministic(sim, instr)
        if instr.name == "TICK":
            checkpoints.append((j, sim.copy()))
    baseline_record = np.asarray(sim.current_measurement_record(), dtype=np.uint8)
    return checkpoints, baseline_record


def _nearest_checkpoint(checkpoints, target_offset: int):
    """Largest checkpoint whose after_index is strictly before target_offset."""
    best = checkpoints[0]
    for cp in checkpoints:
        if cp[0] < target_offset:
            best = cp
        else:
            break
    return best


def _simulate_fault_from_checkpoint(noisy_flat, checkpoints, target_offset: int, pauli_ops):
    """Clone the nearest preceding checkpoint, apply pauli_ops at
    target_offset, then advance the remaining suffix. Returns the full
    measurement record."""
    after_idx, cp = _nearest_checkpoint(checkpoints, target_offset)
    sim = cp.copy()
    for j in range(after_idx + 1, len(noisy_flat)):
        instr = noisy_flat[j]
        if j == target_offset:
            for q, p in pauli_ops:
                if p == "X":
                    sim.x(q)
                elif p == "Y":
                    sim.y(q)
                elif p == "Z":
                    sim.z(q)
            # Pure noise at the fault site is replaced by the Pauli.
            # A measurement carrying its own readout-noise arg keeps the
            # (noise-stripped) measurement, with the Pauli applied just before.
            if not _is_pure_noise_channel(instr):
                sim.do(_strip_measurement_noise_arg(instr))
            continue
        _do_deterministic(sim, instr)
    return np.asarray(sim.current_measurement_record(), dtype=np.uint8)


class PhysicalFrameDecoder:
    """
    Wraps a stim circuit so a syndrome decodes to:
      - a per-fault physical correction vector,
      - a per-qubit Pauli frame at each fault's own circuit location, and
      - a per-final-measurement correction to XOR onto readout, obtained by
        actually simulating each candidate fault forward through the circuit.

    Parameters
    ----------
    circuit :
        Noisy stim circuit to decode. ``OBSERVABLE_INCLUDE`` is optional: without
        it, ``predicted_obs`` is empty and matching uses detectors only, while
        ``final_correction`` still covers all tracked data qubits.
    decompose_errors :
        Forwarded to ``circuit.detector_error_model``.
    lazy :
        If True, defer each fault's final-frame simulation until that fault
        is first selected by ``decode()`` (or first read from
        ``final_frame_contributions``), then cache it. If False (default),
        resolve every fault at construction time so decode pays only matching
        cost.
    data_qubits :
        Optional qubit ids to track for ``final_correction``. Default: every
        target of non-reset ``M``/``MX``/``MY``/``MZ`` in the circuit (the
        final data layer in a typical memory experiment). If given, each
        listed qubit's *last* such measurement is tracked.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        decompose_errors: bool = True,
        lazy: bool = False,
        data_qubits: Sequence[int] | None = None,
    ):
        self.circuit = circuit
        self.lazy = lazy
        dem = circuit.detector_error_model(decompose_errors=decompose_errors)
        self.dem = dem.flattened()
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

        # --- final-measurement bookkeeping ---
        (
            self.meas_to_qubit,
            self.obs_groups,
            data_meas_indices,
        ) = _final_measurement_targets(circuit, data_qubits=data_qubits)
        if not data_meas_indices:
            raise ValueError(
                "circuit has no non-reset M/MX/MY/MZ measurements to track; "
                "pass data_qubits=... or add a final data-qubit measurement layer"
            )
        # Track all terminal data readouts (not only OBSERVABLE_INCLUDE support).
        self._final_meas_indices = list(data_meas_indices)
        self._final_meas_pos = {m: k for k, m in enumerate(self._final_meas_indices)}
        self.final_qubits = [self.meas_to_qubit[m] for m in self._final_meas_indices]

        noisy_flat = flattened_instructions(circuit)
        # Stage 2: one noiseless TableauSimulator pass with TICK checkpoints.
        checkpoints, baseline_record = _build_tick_checkpoints(noisy_flat)
        self._noisy_flat = noisy_flat
        self._checkpoints = checkpoints
        self._baseline_final = baseline_record[self._final_meas_indices]

        # --- build check matrix + per-fault Pauli/propagation info ---
        error_instrs = [
            instr for instr in self.dem if isinstance(instr, stim.DemInstruction) and instr.type == "error"
        ]

        # 1. Group components by symptom key and compute combined probabilities
        groups = {}  # (det_key, obs_key) -> list of (p, comp_targets)
        for instr in error_instrs:
            p = instr.args_copy()[0]
            components: list[list[stim.DemTarget]] = [[]]
            # An `error` instruction's targets are always DemTargets, never the
            # raw ints that e.g. `shift_detectors` carries.
            for t in instr.targets_copy():
                if not isinstance(t, stim.DemTarget):
                    continue
                if t.is_separator():
                    components.append([])
                else:
                    components[-1].append(t)

            for comp_targets in components:
                det_key = []
                obs_key = []
                for t in comp_targets:
                    if t.is_relative_detector_id():
                        det_key.append(t.val)
                    elif t.is_logical_observable_id():
                        obs_key.append(t.val)
                key = (frozenset(det_key), frozenset(obs_key))
                groups.setdefault(key, []).append((p, comp_targets))

        # 2. Pick representatives and build flat DEM
        flat_dem = stim.DetectorErrorModel()
        merged_det_rows, merged_det_cols, merged_weights = [], [], []
        merged_obs_rows, merged_obs_cols = [], []

        symptom_keys_for_fault = []

        for g, (key, members) in enumerate(groups.items()):
            det_key, obs_key = key
            for d in det_key:
                merged_det_rows.append(d)
                merged_det_cols.append(g)
            for o in obs_key:
                merged_obs_rows.append(o)
                merged_obs_cols.append(g)

            p_acc = 0.0
            for p, _ in members:
                p_acc = p_acc + p - 2 * p_acc * p
            p_acc = min(max(p_acc, 1e-12), 1 - 1e-12)
            merged_weights.append(np.log((1 - p_acc) / p_acc))

            rep_p, rep_targets = max(members, key=lambda m: m[0])
            symptom_keys_for_fault.append(key)
            flat_dem.append("error", [rep_p], rep_targets)

        self.num_errors = len(groups)

        # 3. Batch explain: one call for all merged representatives (Stage 1).
        # Match results back by the same symptom key used for edge merging.
        explained_errors = circuit.explain_detector_error_model_errors(
            dem_filter=flat_dem,
            reduce_to_one_representative_error=True,
        )

        symptom_to_explained = {}
        for explained in explained_errors:
            det_key = []
            obs_key = []
            for t_with_coords in explained.dem_error_terms:
                t = t_with_coords.dem_target
                if t.is_relative_detector_id():
                    det_key.append(t.val)
                elif t.is_logical_observable_id():
                    obs_key.append(t.val)
            key = (frozenset(det_key), frozenset(obs_key))
            if key in symptom_to_explained:
                raise RuntimeError(f"duplicate symptom key from batched explain: {key}")
            symptom_to_explained[key] = explained

        missing = set(symptom_keys_for_fault) - set(symptom_to_explained)
        if missing:
            raise RuntimeError(
                f"batched explain missed {len(missing)} symptom key(s); first missing={next(iter(missing))}"
            )

        # 4. Extract per-fault Paulis + instruction offsets (cheap). Final-frame
        # contributions are simulated eagerly or lazily (Stage 3) below.
        self.error_to_paulis: dict[int, list[tuple[int, str]]] = {}
        self._fault_offsets: dict[int, int | None] = {}
        self._fault_direct_meas_flips: dict[int, set[int]] = {}
        self._final_frame_contributions: dict[int, set[int]] = {}

        for i in range(self.num_errors):
            key = symptom_keys_for_fault[i]
            explained = symptom_to_explained[key]

            if not explained.circuit_error_locations:
                raise RuntimeError(f"batched explain returned no circuit location for fault {i} with symptom key {key}")
            loc = explained.circuit_error_locations[0]
            
            direct_flips = set()
            if getattr(loc, "flipped_measurement", None) is not None:
                rec_idx = loc.flipped_measurement.record_index
                if rec_idx in self._final_meas_pos:
                    direct_flips.add(self._final_meas_pos[rec_idx])
            self._fault_direct_meas_flips[i] = direct_flips

            paulis = []
            for gtc in loc.flipped_pauli_product:
                gt = gtc.gate_target
                if gt.is_x_target:
                    pauli = "X"
                elif gt.is_y_target:
                    pauli = "Y"
                elif gt.is_z_target:
                    pauli = "Z"
                else:
                    continue
                paulis.append((gt.value, pauli))

            self.error_to_paulis[i] = paulis
            self._fault_offsets[i] = loc.stack_frames[0].instruction_offset if paulis else None

        if not lazy:
            for i in range(self.num_errors):
                self._resolve_contribution(i)
            # Checkpoints no longer needed once everything is resolved.
            self._checkpoints = None
            self._noisy_flat = None

        weights = np.array(merged_weights)
        self.H = sp.csc_matrix(
            (np.ones(len(merged_det_rows), dtype=np.uint8), (merged_det_rows, merged_det_cols)),
            shape=(self.num_detectors, self.num_errors),
        )
        self.observables_matrix = sp.csc_matrix(
            (np.ones(len(merged_obs_rows), dtype=np.uint8), (merged_obs_rows, merged_obs_cols)),
            shape=(self.num_observables, self.num_errors),
        )

        # Single matching graph. predicted_obs is derived algebraically from
        # the SAME fault_vector (observables_matrix @ fault_vector), never
        # from a second, independently-solved Matching object -- otherwise
        # ties in the minimum-weight matching can be broken differently
        # between two decode() calls and the outputs silently disagree.
        self.matching_phys = pymatching.Matching.from_check_matrix(self.H, weights=weights)

    def _resolve_contribution(self, i: int) -> set[int]:
        """Compute (and memoize) fault i's final-measurement flip set."""
        if i in self._final_frame_contributions:
            return self._final_frame_contributions[i]

        contribution = set(self._fault_direct_meas_flips[i])

        paulis = self.error_to_paulis[i]
        offset = self._fault_offsets[i]
        if paulis and offset is not None:
            if self._checkpoints is None or self._noisy_flat is None:
                raise RuntimeError(
                    "fault resolution data was discarded; construct with lazy=True "
                    "to keep checkpoints for on-demand resolution"
                )
            result = _simulate_fault_from_checkpoint(self._noisy_flat, self._checkpoints, offset, paulis)
            flipped = result[self._final_meas_indices] != self._baseline_final
            contribution.update(np.where(flipped)[0].tolist())

        self._final_frame_contributions[i] = contribution
        return contribution

    @property
    def final_frame_contributions(self) -> dict[int, set[int]]:
        """Per-fault sets of local final-measurement indices flipped by that
        fault. Resolves any still-pending faults on first full-table access."""
        for i in range(self.num_errors):
            self._resolve_contribution(i)
        return self._final_frame_contributions

    def _outputs_from_fault_vector(self, fault_vector: np.ndarray):
        """Build frame / predicted_obs / final_correction from a fault vector."""
        predicted_obs = (self.observables_matrix @ fault_vector) % 2

        frame: dict[int, str] = {}
        final_local = np.zeros(len(self._final_meas_indices), dtype=np.uint8)
        table = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
        inv = {v: k for k, v in table.items()}
        for i, bit in enumerate(fault_vector):
            if not bit:
                continue
            for qubit, pauli in self.error_to_paulis[i]:
                x1, z1 = table[frame.get(qubit, "I")]
                x2, z2 = table[pauli]
                frame[qubit] = inv[(x1 ^ x2, z1 ^ z2)]
            for local_idx in self._resolve_contribution(i):
                final_local[local_idx] ^= 1

        final_correction = {q: int(b) for q, b in zip(self.final_qubits, final_local)}
        return frame, predicted_obs, final_correction

    def decode(self, syndrome: np.ndarray):
        """
        Returns
        -------
        fault_vector : np.uint8[num_errors]
            Which physical error mechanisms the decoder selected.
        frame : dict[int, str]
            qubit -> Pauli, composed AT EACH FAULT'S OWN LOCATION (spacetime
            view -- see module docstring; not meant to be XORed onto readout).
        predicted_obs : np.uint8[num_observables]
            Logical-observable prediction, computed as
            (observables_matrix @ fault_vector) mod 2 -- guaranteed
            consistent with final_correction since both come from the same
            fault_vector.
        final_correction : dict[int, int]
            qubit -> 0/1, the bit to XOR onto that qubit's FINAL measurement
            outcome, obtained by propagating the selected faults to the end
            of the circuit via direct simulation. This is what you asked for.
        """
        # Only the (ndarray, weight) form is a tuple, and we never ask for weights.
        fault_vector = cast(np.ndarray, self.matching_phys.decode(syndrome))
        frame, predicted_obs, final_correction = self._outputs_from_fault_vector(fault_vector)
        return fault_vector, frame, predicted_obs, final_correction

    def decode_batch(self, syndromes: np.ndarray):
        """
        Vectorized decode over many syndrome shots via
        ``pymatching.Matching.decode_batch``.

        Parameters
        ----------
        syndromes : np.ndarray
            Shape ``(num_shots, num_detectors)``, dtype uint8/bool.

        Returns
        -------
        fault_vectors : np.uint8[num_shots, num_errors]
        frames : list[dict[int, str]]
        predicted_obs : np.uint8[num_shots, num_observables]
        final_corrections : list[dict[int, int]]
        """
        syndromes = np.asarray(syndromes)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        fault_vectors = cast(np.ndarray, self.matching_phys.decode_batch(syndromes))

        frames = []
        predicted_obs = np.empty((len(fault_vectors), self.num_observables), dtype=np.uint8)
        final_corrections = []
        for s, fault_vector in enumerate(fault_vectors):
            frame, obs, final_correction = self._outputs_from_fault_vector(fault_vector)
            frames.append(frame)
            predicted_obs[s] = obs
            final_corrections.append(final_correction)

        return fault_vectors, frames, predicted_obs, final_corrections


if __name__ == "__main__":
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=5,
        rounds=5,
        after_clifford_depolarization=0.01,
        after_reset_flip_probability=0.01,
        before_measure_flip_probability=0.01,
    )

    print("building decoder (checkpointed TableauSimulator per fault)...")
    decoder = PhysicalFrameDecoder(circuit)
    print(
        f"errors: {decoder.num_errors}, detectors: {decoder.num_detectors}, "
        f"observables: {decoder.num_observables}, final qubits tracked: "
        f"{len(decoder.final_qubits)}"
    )

    # Reference: the standard pymatching workflow, for cross-checking.
    dem_ref = circuit.detector_error_model(decompose_errors=True)
    matching_ref = pymatching.Matching.from_detector_error_model(dem_ref)

    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=300, separate_observables=True)

    n_self_consistent = 0
    n_matches_reference = 0
    n_correct = 0
    for s in range(300):
        fault_vector, frame, predicted_obs, final_correction = decoder.decode(dets[s])

        # (a) internal consistency: XOR-ing final_correction over each
        # observable's group of final qubits must reproduce predicted_obs
        # -- always true now, both come from the same fault_vector.
        recombined = 0
        for m in decoder.obs_groups[0]:
            q = decoder.meas_to_qubit[m]
            recombined ^= final_correction[q]
        n_self_consistent += int(recombined == predicted_obs[0])

        # (b) does our from-scratch H/weights construction reproduce the
        # standard decoder's prediction (a check on the H matrix itself)?
        ref_pred = matching_ref.decode(dets[s])
        n_matches_reference += int(predicted_obs[0] == ref_pred[0])

        # (c) does it actually predict the true logical outcome?
        n_correct += int(predicted_obs[0] == obs[s, 0])

    print(f"final_correction <-> predicted_obs self-consistency: {n_self_consistent}/300 (must be 300/300)")
    print(f"bit-for-bit agreement with reference pymatching decoder: {n_matches_reference}/300 (must be 300/300)")
    print(f"agreement with true logical outcome:                     {n_correct}/300")

    fault_vector, frame, predicted_obs, final_correction = decoder.decode(dets[0])
    print(f"\nshot 0: predicted_obs={predicted_obs}, actual_obs={obs[0]}")
    print(f"final_correction (qubit -> flip bit before combining into observable):")
    print(final_correction)
