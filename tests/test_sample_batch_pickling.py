"""Tests for SampleBatch pickling and Python construction."""

import pickle

import numpy as np
import pytest

from qec_loss import ForwardSampler, LossyCircuit, SampleBatch


class TestSampleBatchConstruction:
    """Test constructing SampleBatch from Python."""

    def test_construct_minimal(self):
        """Test constructing a minimal SampleBatch."""
        n_samples = 10
        measurements = np.zeros((n_samples, 5), dtype=np.uint8)
        detectors = np.ones((n_samples, 3), dtype=np.uint8)
        observables = np.ones((n_samples, 2), dtype=np.uint8) * 2
        loss_patterns: list[dict[int, list[int]]] = [{} for _ in range(n_samples)]

        batch = SampleBatch(measurements, detectors, observables, loss_patterns)
        assert batch.measurements.shape == (n_samples, 5)
        assert batch.detectors.shape == (n_samples, 3)
        assert batch.observables.shape == (n_samples, 2)
        assert len(batch.loss_patterns) == n_samples


class TestSampleBatchPickle:
    """Test pickling SampleBatch."""

    def test_pickle_constructed_batch(self):
        """Test pickling a SampleBatch constructed from Python."""
        n_samples = 5
        measurements = np.zeros((n_samples, 10), dtype=np.uint8)
        detectors = np.ones((n_samples, 3), dtype=np.uint8)
        observables = np.ones((n_samples, 2), dtype=np.uint8) * 2
        loss_patterns = [{} for _ in range(n_samples)]

        batch = SampleBatch(measurements, detectors, observables, loss_patterns)

        pickled = pickle.dumps(batch)
        restored = pickle.loads(pickled)

        assert restored.measurements.shape == (5, 10)
        assert np.array_equal(restored.measurements, measurements)
        assert restored.detectors.shape == (5, 3)
        assert np.array_equal(restored.detectors, detectors)
        assert restored.observables.shape == (5, 2)
        assert np.array_equal(restored.observables, observables)
        assert len(restored.loss_patterns) == 5

    def test_pickle_round_trip_data_integrity(self):
        """Test that data is preserved through pickle round-trip."""
        n_samples = 5
        measurements = np.zeros((n_samples, 10), dtype=np.uint8)
        detectors = np.ones((n_samples, 3), dtype=np.uint8)
        observables = np.ones((n_samples, 2), dtype=np.uint8) * 2
        loss_patterns = [{} for _ in range(n_samples)]

        batch = SampleBatch(measurements, detectors, observables, loss_patterns)

        # Set some values
        batch.measurements[0, 0] = 1
        batch.measurements[1, 2] = 1
        batch.detectors[0, 0] = 1

        restored = pickle.loads(pickle.dumps(batch))

        assert restored.measurements[0, 0] == 1
        assert restored.measurements[1, 2] == 1
        assert restored.detectors[0, 0] == 1

    def test_repr_still_works(self):
        """Test that __repr__ still works after changes."""
        n_samples = 2
        measurements = np.zeros((n_samples, 3), dtype=np.uint8)
        detectors = np.ones((n_samples, 1), dtype=np.uint8)
        observables = np.ones((n_samples, 1), dtype=np.uint8)
        loss_patterns = [{} for _ in range(n_samples)]

        batch = SampleBatch(measurements, detectors, observables, loss_patterns)
        repr_str = repr(batch)
        assert repr_str.startswith("SampleBatch(measurements=array(")
        assert "detectors=" in repr_str
        assert "observables=" in repr_str
        assert "loss_patterns=" in repr_str
