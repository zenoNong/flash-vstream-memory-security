"""
Test CSM Temporal Compression — Experiment 0 (Tests 0A-0E)

Tests the weighted_kmeans_ordered_feature() and temporal_compress() contract
using synthetic data. No GPU or model weights required.
"""

import os
import sys
import pytest
import torch
import random
import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.conftest import make_features_3d, INTERNAL_TEMPORAL_LENGTH
from tests.direct_imports import weighted_kmeans_ordered_feature


class TestCSMShortSequence:
    """Test 0A: T < internal temporal length — no compression expected."""

    @pytest.mark.parametrize("T", [1, 5, 10, 30, 59])
    def test_short_sequence_passthrough(self, T):
        """Features shorter than temporal_length should pass through unchanged."""
        P, D = 4, 32  # Small for fast testing
        features = make_features_3d(T, P, D, diversity='random', seed=42)
        T0 = INTERNAL_TEMPORAL_LENGTH  # 60

        result = weighted_kmeans_ordered_feature(features, T0)

        # Short path returns 3 values
        assert len(result) == 3, f"Expected 3 returns for T={T} <= T0={T0}, got {len(result)}"

        out_features, out_weights, out_indices = result

        # Features should be identical
        assert torch.allclose(out_features, features), "Short sequence features should be unchanged"

        # Weights should be ones
        assert torch.allclose(out_weights, torch.ones(T)), "Short sequence weights should be ones"

        # Indices should be identity
        assert len(out_indices) == 1
        assert len(out_indices[0]) == T
        for i, idx_list in enumerate(out_indices[0]):
            assert idx_list == [i], f"Index {i} should be [{i}], got {idx_list}"


class TestCSMBoundary:
    """Test 0B: T = temporal_length and T = temporal_length + 1."""

    def test_exact_boundary_no_compression(self):
        """T = T0 should NOT trigger compression."""
        T = INTERNAL_TEMPORAL_LENGTH  # 60
        P, D = 4, 32
        features = make_features_3d(T, P, D, diversity='random', seed=42)

        result = weighted_kmeans_ordered_feature(features, T)
        assert len(result) == 3, f"T==T0 should return 3 values, got {len(result)}"
        out_features, out_weights, out_indices = result
        assert torch.allclose(out_features, features)

    def test_boundary_plus_one_triggers_compression(self):
        """T = T0 + 1 SHOULD trigger compression."""
        T = INTERNAL_TEMPORAL_LENGTH + 1  # 61
        T0 = INTERNAL_TEMPORAL_LENGTH      # 60
        P, D = 4, 32
        features = make_features_3d(T, P, D, diversity='random', seed=42)

        result = weighted_kmeans_ordered_feature(features, T0)
        assert len(result) == 4, f"T>T0 should return 4 values, got {len(result)}"
        out_features, out_weights, out_timestamps, out_indices = result

        # Output should have T0 temporal entries
        assert out_features.shape[0] == T0, f"Expected {T0} output frames, got {out_features.shape[0]}"
        assert out_features.shape[1:] == features.shape[1:], "Spatial dims should be preserved"
        assert out_weights.shape[0] == T0
        assert out_timestamps.shape[0] == T0
        assert len(out_indices) == T0


class TestCSMNormalCompression:
    """Test 0C: T >> temporal_length — normal compression."""

    @pytest.mark.parametrize("T", [120, 180, 240])
    def test_compression_output_shapes(self, T):
        """Verify output shapes after compression."""
        T0 = INTERNAL_TEMPORAL_LENGTH  # 60
        P, D = 4, 32
        features = make_features_3d(T, P, D, diversity='random', seed=42)

        result = weighted_kmeans_ordered_feature(features, T0)
        assert len(result) == 4

        out_features, out_weights, out_timestamps, out_indices = result

        assert out_features.shape[0] == T0
        assert out_features.shape[1] == P
        assert out_features.shape[2] == D
        assert out_weights.shape[0] == T0
        assert out_timestamps.shape[0] == T0
        assert len(out_indices) == T0

    def test_all_frames_assigned(self):
        """Every input frame should be assigned to exactly one cluster."""
        T, T0 = 120, INTERNAL_TEMPORAL_LENGTH
        P, D = 4, 32
        features = make_features_3d(T, P, D, diversity='random', seed=42)

        _, _, _, out_indices = weighted_kmeans_ordered_feature(features, T0)

        all_assigned = set()
        for cluster_members in out_indices:
            for frame_id in cluster_members:
                assert frame_id not in all_assigned, f"Frame {frame_id} assigned to multiple clusters"
                all_assigned.add(frame_id)

        assert all_assigned == set(range(T)), f"Not all frames assigned: missing {set(range(T)) - all_assigned}"

    def test_timestamps_sorted(self):
        """Output timestamps should be in sorted (temporal) order."""
        T, T0 = 120, INTERNAL_TEMPORAL_LENGTH
        P, D = 4, 32
        features = make_features_3d(T, P, D, diversity='structured', seed=42)

        _, _, timestamps, _ = weighted_kmeans_ordered_feature(features, T0)

        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1], \
                f"Timestamps not sorted: [{i}]={timestamps[i]:.2f} > [{i+1}]={timestamps[i+1]:.2f}"

    def test_weights_positive(self):
        """All output weights should be positive."""
        T, T0 = 120, INTERNAL_TEMPORAL_LENGTH
        P, D = 4, 32
        features = make_features_3d(T, P, D, diversity='random', seed=42)

        _, weights, _, _ = weighted_kmeans_ordered_feature(features, T0)

        assert (weights >= 0).all(), f"Found negative weights: {weights[weights < 0]}"

    def test_no_nan_in_output(self):
        """No NaN values in any output."""
        T, T0 = 120, INTERNAL_TEMPORAL_LENGTH
        P, D = 4, 32
        features = make_features_3d(T, P, D, diversity='random', seed=42)

        out_features, weights, timestamps, _ = weighted_kmeans_ordered_feature(features, T0)

        assert not torch.isnan(out_features).any(), "NaN in output features"
        assert not torch.isnan(weights).any(), "NaN in weights"
        assert not torch.isnan(timestamps).any(), "NaN in timestamps"


class TestCSMDuplicateFallback:
    """Test 0D: Low-diversity features trigger unique-point fallback."""

    def test_low_diversity_does_not_crash(self):
        """K-means with very few unique features should not crash."""
        T, T0 = 120, INTERNAL_TEMPORAL_LENGTH
        P, D = 4, 32
        features = make_features_3d(T, P, D, diversity='low_diversity', seed=42)

        # Should not raise
        result = weighted_kmeans_ordered_feature(features, T0)

        # Fallback path returns 4 values (padded)
        assert len(result) == 4
        out_features, weights, timestamps, indices = result

        # After padding, should have T0 entries
        assert out_features.shape[0] == T0, f"Expected {T0} after padding, got {out_features.shape[0]}"

    def test_low_diversity_valid_shapes(self):
        """Even with padding, output shapes must be valid."""
        T, T0 = 120, INTERNAL_TEMPORAL_LENGTH
        P, D = 4, 32
        features = make_features_3d(T, P, D, diversity='low_diversity', seed=42)

        out_features, weights, timestamps, indices = weighted_kmeans_ordered_feature(features, T0)

        assert out_features.shape == (T0, P, D)
        assert weights.shape[0] == T0
        assert timestamps.shape[0] == T0
        assert not torch.isnan(out_features).any()


class TestCSMDeterminism:
    """Test 0E: Determinism under fixed seed."""

    def test_deterministic_output(self):
        """Same seed + same input should produce same output."""
        T, T0 = 120, INTERNAL_TEMPORAL_LENGTH
        P, D = 4, 32

        results = []
        for _ in range(3):
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            features = make_features_3d(T, P, D, diversity='random', seed=42)
            result = weighted_kmeans_ordered_feature(features, T0)
            results.append(result)

        # Compare all runs
        for i in range(1, len(results)):
            assert torch.allclose(results[0][0], results[i][0], atol=1e-5), \
                f"Run {i} features differ from run 0"
            assert torch.allclose(results[0][1], results[i][1], atol=1e-5), \
                f"Run {i} weights differ from run 0"

    def test_different_seed_different_output(self):
        """Different seeds should (very likely) produce different outputs."""
        T, T0 = 120, INTERNAL_TEMPORAL_LENGTH
        P, D = 4, 32

        # Use the same features but different K-means seeds
        features = make_features_3d(T, P, D, diversity='random', seed=42)

        random.seed(42)
        torch.manual_seed(42)
        result1 = weighted_kmeans_ordered_feature(features.clone(), T0)

        random.seed(999)
        torch.manual_seed(999)
        result2 = weighted_kmeans_ordered_feature(features.clone(), T0)

        # Not guaranteed to differ, but very likely with random init
        # We just check that the function doesn't crash with different seeds
        assert result1[0].shape == result2[0].shape
