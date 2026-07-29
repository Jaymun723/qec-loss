"""Decompose a detector error model into graphlike components.

Stim's ``Circuit.detector_error_model(decompose_errors=True)`` decomposes errors
in two stages inside ``ErrorAnalyzer`` (see ``extern/stim/.../error_analyzer.cc``):

1. **Local decomposition** of composite Pauli channels (``DEPOLARIZE1/2``, …):
   express higher-weight cases as XOR of simpler co-channel cases, annotated
   with ``^`` separators.
2. **Global decomposition**: rewrite any remaining non-graphlike errors using
   other known graphlike errors in the model
   (``brute_force_decomposition_into_known_graphlike_errors`` and remnant edges).

This module applies the same ideas to an existing ``DetectorErrorModel``. Local
structure is recovered by finding errors whose symptom sets form XOR triples
(the DEM image of a ``DEPOLARIZE1`` channel); the global pass mirrors Stim's
``do_global_error_decomposition_pass``.
"""

from __future__ import annotations

import stim


def is_graphlike(targets: list[stim.DemTarget]) -> bool:
    """True iff every separator-delimited component has at most two detectors."""
    symptom_count = 0
    for t in targets:
        if t.is_separator():
            symptom_count = 0
        elif t.is_relative_detector_id():
            symptom_count += 1
            if symptom_count > 2:
                return False
    return True


def _n_det(ts: list[stim.DemTarget]) -> int:
    return sum(1 for t in ts if t.is_relative_detector_id())


def _has_obs(ts: list[stim.DemTarget]) -> bool:
    return any(t.is_logical_observable_id() for t in ts)


def _targets_key(ts: list[stim.DemTarget]) -> tuple[str, ...]:
    return tuple(repr(t) for t in ts)


def _as_vec(ts: list[stim.DemTarget]) -> frozenset[str]:
    return frozenset(repr(t) for t in ts if not t.is_separator())


def _det_tuple(ts: list[stim.DemTarget]) -> tuple[int, ...]:
    return tuple(sorted(t.val for t in ts if t.is_relative_detector_id()))


def _combine_prob(p_old: float, p_new: float) -> float:
    return p_old * (1.0 - p_new) + (1.0 - p_old) * p_new


def _xor_targets(a: list[stim.DemTarget], b: list[stim.DemTarget]) -> list[stim.DemTarget]:
    present: dict[str, stim.DemTarget] = {}
    for t in list(a) + list(b):
        if t.is_separator():
            continue
        r = repr(t)
        if r in present:
            del present[r]
        else:
            present[r] = t
    items = list(present.values())
    items.sort(key=lambda t: (not t.is_relative_detector_id(), t.val))
    return items


def _split_components(targets: list[stim.DemTarget]) -> list[list[stim.DemTarget]]:
    comps: list[list[stim.DemTarget]] = []
    cur: list[stim.DemTarget] = []
    for t in targets:
        if t.is_separator():
            comps.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        comps.append(cur)
    return comps


def _detector_symptoms(component: list[stim.DemTarget]) -> list[stim.DemTarget]:
    return [t for t in component if t.is_relative_detector_id()]


def _join_sep(comps: list[list[stim.DemTarget]]) -> list[stim.DemTarget]:
    out: list[stim.DemTarget] = []
    for i, c in enumerate(comps):
        if i:
            out.append(stim.target_separator())
        out.extend(c)
    return out


def _obs_mask_of_targets(targets: list[stim.DemTarget]) -> tuple[int, int]:
    obs_mask = 0
    used_mask = 0
    for k, t in enumerate(targets):
        if t.is_logical_observable_id():
            if t.val >= 64:
                raise ValueError("Not implemented: decomposing errors with observable ids >= 64.")
            obs_mask |= 1 << t.val
            used_mask |= 1 << k
    return obs_mask, used_mask


def _brute_force_decomp(
    problem: list[stim.DemTarget],
    known_symptoms: dict[tuple[str, ...], list[stim.DemTarget]],
) -> list[list[stim.DemTarget]] | None:
    """Port of stim::brute_force_decomposition_into_known_graphlike_errors."""
    if len(problem) >= 64:
        raise ValueError("Not implemented: decomposing errors with more than 64 terms.")

    out: list[list[stim.DemTarget]] = []
    obs_mask, used_mask = _obs_mask_of_targets(problem)

    def helper(start: int, used_term_mask: int, remaining_obs_mask: int) -> bool:
        while True:
            if start >= len(problem):
                return remaining_obs_mask == 0
            if ((used_term_mask >> start) & 1) == 0:
                break
            start += 1
        used_term_mask |= 1 << start
        key_list = [problem[start]]
        for k in range(start + 1, len(problem) + 1):
            if k < len(problem):
                if (used_term_mask >> k) & 1:
                    continue
                key_list.append(problem[k])
                used_term_mask ^= 1 << k
            key = tuple(repr(t) for t in key_list)
            match = known_symptoms.get(key)
            if match is not None:
                obs_change = _obs_mask_of_targets(match)[0]
                if helper(start + 1, used_term_mask, remaining_obs_mask ^ obs_change):
                    out.append(match)
                    return True
            if k < len(problem):
                key_list.pop()
                used_term_mask ^= 1 << k
        return False

    if helper(0, used_mask, obs_mask):
        return list(reversed(out))
    return None


def _remnant_decomp(
    component: list[stim.DemTarget],
    known_symptoms: dict[tuple[str, ...], list[stim.DemTarget]],
) -> list[list[stim.DemTarget]] | None:
    """Port of ErrorAnalyzer::decompose_and_append_component_to_tail."""
    done = [False] * len(component)
    num_dets = 0
    for k, t in enumerate(component):
        if t.is_relative_detector_id():
            num_dets += 1
        else:
            done[k] = True
    if num_dets <= 2:
        return [list(component)]

    sparse = list(component)
    result: list[list[stim.DemTarget]] = []

    for k in range(len(component)):
        if not done[k]:
            for k2 in range(k + 1, len(component)):
                if not done[k2]:
                    key = tuple(repr(t) for t in (component[k], component[k2]))
                    if key in known_symptoms:
                        done[k] = True
                        done[k2] = True
                        match = known_symptoms[key]
                        result.append(list(match))
                        sparse = _xor_targets(sparse, match)
                        break

    missed = 0
    for k in range(len(component)):
        if not done[k]:
            key = (repr(component[k]),)
            if key in known_symptoms:
                done[k] = True
                match = known_symptoms[key]
                result.append(list(match))
                sparse = _xor_targets(sparse, match)
            missed += not done[k]

    if missed <= 2:
        if sparse:
            result.append(sparse)
        return result
    return None


def _score_basis(a: list[stim.DemTarget], b: list[stim.DemTarget]) -> tuple:
    """Prefer Stim-like local bases: pairs without observables, then lex detector ids."""
    if _n_det(a) != _n_det(b):
        pair = a if _n_det(a) > _n_det(b) else b
        light = b if _n_det(a) > _n_det(b) else a
        return (1 if _has_obs(pair) else 0, _det_tuple(pair), _det_tuple(light), _has_obs(light))
    ta, tb = _det_tuple(a), _det_tuple(b)
    if ta > tb:
        ta, tb = tb, ta
    return (0, ta, tb, _has_obs(a) + _has_obs(b))


def _order_components(
    a: list[stim.DemTarget], b: list[stim.DemTarget]
) -> tuple[list[stim.DemTarget], list[stim.DemTarget]]:
    if _n_det(a) != _n_det(b):
        if _n_det(a) < _n_det(b):
            a, b = b, a
        return a, b
    if _n_det(a) == 1 and _n_det(b) == 1 and not _has_obs(a) and not _has_obs(b):
        # Deterministic order (ascending detector id). Stim's X-then-Z order is
        # not recoverable from the DEM alone; tests canonicalize both sides.
        if a[0].val > b[0].val:
            a, b = b, a
        return a, b
    if _targets_key(b) < _targets_key(a):
        a, b = b, a
    return a, b


def _find_best_split(
    hts: list[stim.DemTarget],
    by_vec: dict[frozenset[str], tuple[tuple[str, ...], float, list[stim.DemTarget]]],
) -> tuple[list[stim.DemTarget], list[stim.DemTarget]] | None:
    hv = _as_vec(hts)
    nd = _n_det(hts)
    options: list[tuple[tuple, list[stim.DemTarget], list[stim.DemTarget]]] = []
    keys = list(by_vec.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i].symmetric_difference(keys[j]) != hv:
                continue
            a = by_vec[keys[i]][2]
            b = by_vec[keys[j]][2]
            if _as_vec(a) == hv or _as_vec(b) == hv:
                continue
            if not (_n_det(a) < nd and _n_det(b) < nd):
                continue
            if not is_graphlike(a) or not is_graphlike(b):
                continue
            options.append((_score_basis(a, b), a, b))
    if not options:
        return None
    options.sort()
    _, a, b = options[0]
    return _order_components(a, b)


def _local_decompose(
    errors_map: dict[tuple[str, ...], tuple[float, list[stim.DemTarget]]],
) -> dict[tuple[str, ...], tuple[float, list[stim.DemTarget]]]:
    """Recover DEPOLARIZE1-style local rewrites from XOR structure in the DEM."""
    errors_list = list(errors_map.items())
    by_vec = {_as_vec(ts): (k, p, ts) for k, (p, ts) in errors_list}

    rewrites: dict[tuple[str, ...], list[stim.DemTarget]] = {}
    protected: set[tuple[str, ...]] = set()

    for k, (_p, ts) in errors_list:
        if is_graphlike(ts):
            continue
        split = _find_best_split(ts, by_vec)
        if split is not None:
            a, b = split
            rewrites[k] = _join_sep([a, b])
            protected.add(_targets_key(a))
            protected.add(_targets_key(b))

    for k, (p, ts) in errors_list:
        if k in rewrites or k in protected:
            continue
        if not is_graphlike(ts) or _n_det(ts) != 2:
            continue
        split = _find_best_split(ts, by_vec)
        if split is None:
            continue
        a, b = split
        if _n_det(a) != 1 or _n_det(b) != 1:
            continue
        # Only rewrite when this looks like a co-channel Y → X ^ Z (same p),
        # not when unrelated singlets from other faults happen to cover the pair.
        pa = by_vec[_as_vec(a)][1]
        pb = by_vec[_as_vec(b)][1]
        if abs(pa - p) > 1e-12 and abs(pb - p) > 1e-12:
            continue
        rewrites[k] = _join_sep([a, b])

    new_map: dict[tuple[str, ...], tuple[float, list[stim.DemTarget]]] = {}
    for k, (p, ts) in errors_list:
        nts = rewrites.get(k, ts)
        nk = _targets_key(nts)
        if nk in new_map:
            new_map[nk] = (_combine_prob(new_map[nk][0], p), nts)
        else:
            new_map[nk] = (p, nts)
    return new_map


def _global_decompose(
    errors_map: dict[tuple[str, ...], tuple[float, list[stim.DemTarget]]],
    *,
    ignore_decomposition_failures: bool,
    block_remnant_edges: bool,
) -> dict[tuple[str, ...], tuple[float, list[stim.DemTarget]]]:
    """Port of ErrorAnalyzer::do_global_error_decomposition_pass."""
    known: dict[tuple[str, ...], list[stim.DemTarget]] = {}
    for _k, (p, targets) in errors_map.items():
        if p == 0 or not targets:
            continue
        for comp in _split_components(targets):
            syms = _detector_symptoms(comp)
            if len(syms) in (1, 2):
                known[tuple(repr(t) for t in syms)] = list(comp)

    rewrites: list[tuple[tuple[str, ...], float, list[stim.DemTarget]]] = []
    for k, (p, targets) in list(errors_map.items()):
        if p == 0 or not targets or is_graphlike(targets):
            continue
        new_comps: list[list[stim.DemTarget]] = []
        for comp in _split_components(targets):
            decomp = _brute_force_decomp(comp, known)
            if decomp is not None:
                new_comps.extend(decomp)
            elif not block_remnant_edges:
                rem = _remnant_decomp(comp, known)
                if rem is not None:
                    new_comps.extend(rem)
                elif ignore_decomposition_failures:
                    new_comps.append(comp)
                else:
                    raise ValueError(
                        "Failed to decompose errors into graphlike components with at most two symptoms. "
                        f"The error component that failed to decompose is {[str(t) for t in comp]!r}."
                    )
            elif ignore_decomposition_failures:
                new_comps.append(comp)
            else:
                raise ValueError(
                    "Failed to decompose errors into graphlike components with at most two symptoms. "
                    f"The error component that failed to decompose is {[str(t) for t in comp]!r}."
                )
        rewrites.append((k, p, _join_sep(new_comps)))

    for k, p, nts in rewrites:
        del errors_map[k]
        nk = _targets_key(nts)
        if nk in errors_map:
            errors_map[nk] = (_combine_prob(errors_map[nk][0], p), nts)
        else:
            errors_map[nk] = (p, nts)
    return errors_map


def _append_error(out: stim.DetectorErrorModel, p: float, ts: list[stim.DemTarget]) -> None:
    args = [p]
    # stim.DetectorErrorModel.append accepts (name, args, targets)
    out.append("error", args, ts)


def canonicalize_dem_components(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Sort ``^``-separated components of each error for order-insensitive compares.

    Stim's local decomposition emits singlet components in X-then-Z channel order,
    which cannot be recovered from a DEM alone. Sorting components makes two
    decompositions with the same pieces compare equal under ``approx_equals``.
    """

    def dem_target_data(t: stim.DemTarget) -> int:
        if t.is_separator():
            return (1 << 64) - 1
        if t.is_logical_observable_id():
            return (1 << 63) | t.val
        return t.val

    def comp_key(comp: list[stim.DemTarget]) -> tuple[int, ...]:
        return tuple(dem_target_data(t) for t in comp)

    errors: list[tuple[float, list[stim.DemTarget]]] = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = float(inst.args_copy()[0])
        comps = _split_components(list(inst.targets_copy()))
        comps.sort(key=comp_key)
        errors.append((p, _join_sep(comps)))
    errors.sort(key=lambda x: tuple(dem_target_data(t) for t in x[1]))

    out = stim.DetectorErrorModel()
    for p, ts in errors:
        _append_error(out, p, ts)
    for inst in dem:
        if inst.type != "error":
            out.append(inst)
    return out


def decompose_errors(
    dem: stim.DetectorErrorModel,
    *,
    ignore_decomposition_failures: bool = False,
    block_decomposition_from_introducing_remnant_edges: bool = False,
) -> stim.DetectorErrorModel:
    """Decompose non-graphlike DEM errors into graphlike components.

    Mirrors Stim's error decomposition when converting a circuit to a DEM with
    ``decompose_errors=True``, applied to an already-built model.

    Parameters
    ----------
    dem:
        Input detector error model (typically from
        ``circuit.detector_error_model(decompose_errors=False)``).
    ignore_decomposition_failures:
        If true, leave undecomposable components as-is instead of raising.
    block_decomposition_from_introducing_remnant_edges:
        If true, forbid Stim's remnant-edge fallback (both pieces of a split
        must already appear elsewhere).

    Returns
    -------
    stim.DetectorErrorModel
        A model whose error instructions are graphlike (at most two detectors
        per ``^``-separated component), with detector/shift metadata preserved.
    """
    errors_map: dict[tuple[str, ...], tuple[float, list[stim.DemTarget]]] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ts = list(inst.targets_copy())
        k = _targets_key(ts)
        p = float(inst.args_copy()[0])
        if k in errors_map:
            errors_map[k] = (_combine_prob(errors_map[k][0], p), ts)
        else:
            errors_map[k] = (p, ts)

    errors_map = _local_decompose(errors_map)
    errors_map = _global_decompose(
        errors_map,
        ignore_decomposition_failures=ignore_decomposition_failures,
        block_remnant_edges=block_decomposition_from_introducing_remnant_edges,
    )

    def dem_target_data(t: stim.DemTarget) -> int:
        # Match stim::DemTarget::data so error ordering matches ErrorAnalyzer::flush.
        if t.is_separator():
            return (1 << 64) - 1
        if t.is_logical_observable_id():
            return (1 << 63) | t.val
        return t.val

    items = [(p, ts) for p, ts in errors_map.values() if p and ts]
    items.sort(key=lambda x: tuple(dem_target_data(t) for t in x[1]))

    out = stim.DetectorErrorModel()
    for p, ts in items:
        _append_error(out, p, ts)
    for inst in dem:
        if inst.type != "error":
            out.append(inst)
    return canonicalize_dem_components(out)
