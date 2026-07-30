from collections import defaultdict

import stim

from .utils import flattened_dem_instructions


def combine_dems(dems: list[stim.DetectorErrorModel], weights: list[float] | None = None) -> stim.DetectorErrorModel:
    """
    Combines multiple detector error models into a single detector error model.

    Args:
        dems: A list of detector error models to combine.
        weights: Optional list of weights for each detector error model. If not provided, equal weights are assumed.

    Returns:
        A combined detector error model.
    """
    if weights is None:
        weights = [1.0] * len(dems)

    global_errors = defaultdict(list)

    for dem, weight in zip(dems, weights):
        for instruction in flattened_dem_instructions(dem):
            if instruction.type == "error":
                error_key = tuple(instruction.targets_copy())
                error_weight = instruction.args_copy()[0] * weight
                global_errors[error_key].append(error_weight)

    combined_dem = stim.DetectorErrorModel()

    for error_key, weights in global_errors.items():
        combined_weight = sum(weights)
        combined_dem.append("error", combined_weight, list(error_key))

    return combined_dem
