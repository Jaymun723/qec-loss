"""
Loss sampling and Monaka style decoding for quantum error correction circuits
"""

# Generated with:
#   pybind11-stubgen qec_loss._native -o . --numpy-array-wrap-with-annotated
#
# pybind11 emits raw C++ names (e.g. `qec_loss::ForwardSampler`) for types that
# are referenced before their `py::class_` is registered; stubgen turns those
# into `...`. Those placeholders are resolved by hand below, so re-generating
# this file requires re-applying them.
from __future__ import annotations
import collections.abc
import os
import numpy
import numpy.typing
import typing
__all__: list[str] = ['DetsRerouter', 'F2Tensor', 'ForwardSampler', 'LifeCycleManager', 'LifeSegment', 'LossInstruction', 'LossyCircuit', 'MonakaBuilder', 'PackedF2Matrix', 'PauliRerouter', 'SampleBatch', 'Sampler', 'combine_circuits_into_dem', 'get_loss_dem', 'get_loss_rewritten_circuits', 'get_stabilizers']
class DetsRerouter:
    @typing.overload
    def __init__(self, circuit: typing.Any, data_qubits: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    @typing.overload
    def __init__(self, circuit: typing.Any) -> None:
        ...
    def get_qubit_for_measurement(self, measurement_index: typing.SupportsInt | typing.SupportsIndex) -> int:
        ...
    def reroute(self, observable_index: typing.SupportsInt | typing.SupportsIndex, lost_qubits: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], optimize: bool = False) -> list:
        ...
    @property
    def data_qubits(self) -> list[int]:
        ...
class F2Tensor:
    """
    A tensor of elements in GF(2).
    """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __getitem__(self, arg0: typing.Any) -> typing.Any:
        ...
    @typing.overload
    def __init__(self, shape: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    @typing.overload
    def __init__(self, buffer: collections.abc.Buffer) -> None:
        ...
    def __matmul__(self, arg0: F2Tensor) -> F2Tensor:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: typing.Any, arg1: typing.Any) -> None:
        ...
    def __str__(self) -> str:
        ...
    def kernel(self) -> F2Tensor:
        """
        Compute the kernel of the tensor (as a matrix).
        """
    def rank(self) -> int:
        """
        Compute the rank of the tensor (as a matrix).
        """
    def rref(self) -> F2Tensor:
        """
        Compute the reduced row echelon form.
        """
    def solve(self, b: F2Tensor) -> F2Tensor:
        """
        Solve the linear system Ax=b.
        """
    def subview(self, ranges: list) -> F2Tensor:
        """
        Get a subview of the tensor.
        """
    def to_numpy(self) -> typing.Any:
        """
        Convert the tensor to a NumPy array.
        """
    @property
    def T(self) -> F2Tensor:
        """
        The transpose of the tensor (only valid for 2D tensors).
        """
    @property
    def offset(self) -> int:
        """
        The data offset.
        """
    @property
    def shape(self) -> list[int]:
        """
        The shape of the tensor.
        """
    @property
    def strides(self) -> list[int]:
        """
        The strides of the tensor.
        """
class ForwardSampler(Sampler):
    """
    A sampler that tracks erasure errors forward through the circuit.
    """
    def __init__(self, circuit: LossyCircuit, seed: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        Initialize the ForwardSampler.
        """
    def sample(self, num_samples: typing.SupportsInt | typing.SupportsIndex, reroute_observables: bool = False, optimize_retoute: bool = False) -> SampleBatch:
        """
        Generate samples taking loss routing into account.
        """
    def sample_measurements(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.uint8], list[list[tuple[int, list[int]]]]]:
        """
        Sample only measurement outcomes.
        """
    @property
    def circuit(self) -> LossyCircuit:
        """
        The circuit being sampled.
        """
class LifeCycleManager:
    """
    Manages and resolves lifecycles of qubits.
    """
    def __init__(self, circuit: LossyCircuit) -> None:
        ...
    def get_life_cycle(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> list[LifeSegment]:
        """
        Get the full lifecycle of a given qubit.
        """
    def get_life_segment_for_measurement(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> LifeSegment:
        """
        Get the life segment associated with a measurement.
        """
class LifeSegment:
    """
    Represents a continuous segment of a qubit's lifecycle.
    """
    def __init__(self, qubit: typing.SupportsInt | typing.SupportsIndex, start: typing.SupportsInt | typing.SupportsIndex, end: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Initialize LifeSegment.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def end(self) -> int:
        ...
    @property
    def loss_locations(self) -> list[int]:
        ...
    @property
    def qubit(self) -> int:
        ...
    @property
    def start(self) -> int:
        ...
class LossInstruction:
    """
    Loss instruction for circuit definition.
    """
    @typing.overload
    def __init__(self, instruction_str: str) -> None:
        ...
    @typing.overload
    def __init__(self, targets: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], p: typing.SupportsFloat | typing.SupportsIndex, tag: str) -> None:
        ...
    @typing.overload
    def __init__(self, targets: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], p: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def p(self) -> float:
        ...
    @property
    def tag(self) -> str:
        ...
    @property
    def targets(self) -> list[int]:
        ...
class LossyCircuit:
    """
    Lossy circuit for quantum error correction.
    """
    @staticmethod
    def from_file(circuit_path: os.PathLike | str | bytes) -> LossyCircuit:
        """
        Load a LossyCircuit from a file.
        """
    def __init__(self, circuit_str: str) -> None:
        """
        Initialize a LossyCircuit from a stim-like circuit string.
        """
    def __str__(self) -> str:
        ...
    def compile_forward_sampler(self, seed: typing.SupportsInt | typing.SupportsIndex | None = None) -> ForwardSampler:
        """
        Compile the circuit into a ForwardSampler for simulating outcomes.
                     
        Args:
            seed: Optional random seed.
        
        Returns:
            A ForwardSampler instance.
        """
    def compile_monaka_builder(self, optimize_rerouting: bool = False) -> MonakaBuilder:
        """
        Compile the circuit into a MonakaBuilder for tracking lost qubits.
                     
        Args:
            optimize_rerouting: Whether to optimize rerouting logic.
        
        Returns:
            A MonakaBuilder instance.
        """
    def to_file(self, arg0: os.PathLike | str | bytes) -> None:
        """
        Save the LossyCircuit to a file.
        """
    @property
    def nominal_circuit(self) -> typing.Any:
        ...
    @property
    def num_detectors(self) -> int:
        ...
    @property
    def num_measurements(self) -> int:
        ...
    @property
    def num_observables(self) -> int:
        ...
    @property
    def num_qubits(self) -> int:
        ...
    @property
    def rerouter(self) -> PauliRerouter:
        ...
class MonakaBuilder:
    """
    Builder to analyze and compile detector error models under loss.
    """
    def __init__(self, circuit: LossyCircuit, optimize_rerouting: bool = False) -> None:
        """
        Initialize MonakaBuilder with a LossyCircuit.
        """
    def decode_batch(self, batch: SampleBatch, include_loss_dem: bool = True, post_select_on_usable_shots: bool = True) -> numpy.typing.NDArray[numpy.uint8]:
        """
        Decode a batch of loss outcomes.
        """
    def get_dem_from_measurements(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8]) -> typing.Any:
        """
        Get the DEM given a specific set of measurement outcomes.
        """
    def get_dems_from_batch(self, arg0: SampleBatch) -> list[typing.Any]:
        """
        Get DEMs for an entire batch of samples.
        """
    def get_life_segment_dem(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], arg1: LifeSegment) -> typing.Any:
        """
        Get the DEM restricted to a specific life segment.
        """
    def get_nominal_dem(self, lost_qubits: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] = []) -> typing.Any:
        """
        Get the nominal DetectorErrorModel given a set of initially lost qubits.
        """
    @property
    def life_cycle_manager(self) -> LifeCycleManager:
        """
        Manager for qubit lifecycles.
        """
class PackedF2Matrix:
    """
    A bit-packed matrix over GF(2) for faster linear algebra.
    """
    def __getitem__(self, arg0: tuple) -> int:
        ...
    def __init__(self, rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Initialize an empty PackedF2Matrix with given dimensions.
        """
    def __matmul__(self, arg0: PackedF2Matrix) -> PackedF2Matrix:
        ...
    def __setitem__(self, arg0: tuple, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    def kernel(self) -> PackedF2Matrix:
        """
        Compute the kernel (nullspace) of the matrix.
        """
    def one(self, r: typing.SupportsInt | typing.SupportsIndex, c: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the bit at (r, c) to 1.
        """
    def rank(self) -> int:
        """
        Compute the rank of the matrix.
        """
    def rref(self) -> PackedF2Matrix:
        """
        Compute the reduced row echelon form.
        """
    def solve(self, b: PackedF2Matrix) -> PackedF2Matrix:
        """
        Solve the linear system Ax = b.
        """
    def to_list(self) -> list[list[int]]:
        """
        Convert the matrix into a list of lists of ints.
        """
    def xor_bit(self, r: typing.SupportsInt | typing.SupportsIndex, c: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Flip the bit at (r, c).
        """
    def xor_rows(self, dst: typing.SupportsInt | typing.SupportsIndex, src: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        XOR row `src` into row `dst`.
        """
    def zero(self, r: typing.SupportsInt | typing.SupportsIndex, c: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the bit at (r, c) to 0.
        """
    @property
    def T(self) -> PackedF2Matrix:
        """
        The transpose of the matrix.
        """
    @property
    def cols(self) -> int:
        """
        Number of columns.
        """
    @property
    def rows(self) -> int:
        """
        Number of rows.
        """
class PauliRerouter:
    def __init__(self, circuit: typing.Any) -> None:
        ...
    def get_S_and_L_matrices(self, observable_index: typing.SupportsInt | typing.SupportsIndex, lost_qubits: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], optimize: bool = False) -> tuple[PackedF2Matrix, PackedF2Matrix]:
        ...
    def reroute(self, observable_index: typing.SupportsInt | typing.SupportsIndex, lost_qubits: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], optimize: bool = False) -> list:
        ...
    def riroute(self, S: PackedF2Matrix, L: PackedF2Matrix, available_measurements: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> list[int]:
        ...
    @property
    def L(self) -> list:
        ...
    @property
    def final_measurements(self) -> list[int]:
        ...
    @property
    def meas_so_far(self) -> list[int]:
        ...
class SampleBatch:
    """
    A batch of circuit samples containing measurements, detectors, observables, and loss patterns.
    """
    def __getstate__(self) -> tuple[numpy.typing.NDArray[numpy.uint8], numpy.typing.NDArray[numpy.uint8], numpy.typing.NDArray[numpy.uint8], list[list[tuple[int, list[int]]]]]:
        ...
    @typing.overload
    def __init__(self, measurements: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], detectors: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], observables: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], loss_patterns: collections.abc.Sequence[collections.abc.Sequence[tuple[typing.SupportsInt | typing.SupportsIndex, collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]]]]) -> None:
        """
        Initialize SampleBatch with LossPatterns.
        """
    @typing.overload
    # `Mapping` keys are invariant, so stubgen's `SupportsInt | SupportsIndex`
    # key type would reject an ordinary `dict[int, ...]`. Narrowed to `int`.
    def __init__(self, measurements: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], detectors: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], observables: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], loss_patterns: collections.abc.Sequence[collections.abc.Mapping[int, collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]]]) -> None:
        """
        Initialize SampleBatch with mapped loss patterns.
        """
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def detectors(self) -> numpy.typing.NDArray[numpy.uint8]:
        ...
    @property
    def loss_patterns(self) -> list[dict[int, list[int]]]:
        ...
    @property
    def measurements(self) -> numpy.typing.NDArray[numpy.uint8]:
        ...
    @property
    def observables(self) -> numpy.typing.NDArray[numpy.uint8]:
        ...
class Sampler:
    """
    Base Sampler for quantum circuits.
    """
    def sample(self, num_samples: typing.SupportsInt | typing.SupportsIndex, reroute_observables: bool = False, optimize_retoute: bool = False) -> SampleBatch:
        """
        Generate samples from the circuit.
        """
    @property
    def circuit(self) -> LossyCircuit:
        """
        The circuit being sampled.
        """
def combine_circuits_into_dem(circuits: collections.abc.Sequence[typing.Any], weights: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None) -> typing.Any:
    """
    Combine multiple circuits into a single weighted DetectorErrorModel.
    """
def get_loss_dem(circuit: LossyCircuit, lost_qubits: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], life_segment: LifeSegment, optimize_rerouting: bool = False) -> typing.Any:
    """
    Get the effective DetectorErrorModel for a life segment with lost qubits.
    """
def get_loss_rewritten_circuits(circuit: LossyCircuit, lost_qubits: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], life_segment: LifeSegment, optimize_rerouting: bool = False) -> list[typing.Any]:
    """
    Get rewritten circuits accounting for lost qubits in a segment.
    """
def get_stabilizers(circuit: typing.Any) -> list:
    ...
