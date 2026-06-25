# Simulations of QEC codes with loss

Python package for sampling and decoding quantum error correction circuits with photon loss.

## Install

```bash
git submodule update --init --recursive
uv pip install -e . --no-build-isolation --group dev
uv run pre-commit install
```

The last command installs a git pre-commit hook that runs **ruff format** on Python and **clang-format** on C++ under `src/qec_loss/_native/`.

To format everything manually:

```bash
uv run pre-commit run --all-files
```

## Quick start

```python
import stim
from qec_loss import LossyCircuit, ForwardSampler, MonakaBuilder, add_loss_noise

stim_circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z", rounds=5, distance=5
)
circuit = add_loss_noise(stim_circuit, loss_after_2_qubit_gate=0.01)

sampler = ForwardSampler(circuit, seed=7)
batch = sampler.sample(1000)

builder = MonakaBuilder(circuit, model_path="/tmp/monaka")
predictions = builder.decode_batch(batch)
```

See [docs/api.md](docs/api.md) for the full API.

## Test

```bash
uv run pytest tests/
```
