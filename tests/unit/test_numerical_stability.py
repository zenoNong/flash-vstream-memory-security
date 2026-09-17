"""
Test Numerical Stability — Candidate E

Tests for NaN, negative sqrt, divide-by-zero, and other numerical issues
in distance and similarity computations.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from research.adapters.dam_adapter import (
    efficient_euclidean_distance,
    cosine_similarity_matrix,
)


class TestEuclideanStability:
    """Numerical stability tests for Euclidean distance."""

    def test_identical_vectors_no_nan(self):
        """Identical vectors: A_2 + B_2 - 2*AB ≈ 0, sqrt(0) should be fine."""
        A = torch.randn(20, 128)
        dist = efficient_euclidean_distance(A, A)
        assert not torch.isnan(dist).any(), "NaN from identical vectors"
        assert not torch.isinf(dist).any(), "Inf from identical vectors"

    def test_near_identical_no_nan(self):
        """Near-identical vectors can produce tiny negative values before sqrt."""
        A = torch.randn(20, 128, dtype=torch.float32)
        B = A + torch.randn_like(A) * 1e-8
        dist = efficient_euclidean_distance(A, B)
        assert not torch.isnan(dist).any(), "NaN from near-identical vectors (negative sqrt)"

    def test_large_magnitude_no_nan(self):
        """Very large feature values should not cause overflow."""
        A = torch.randn(10, 64) * 1e4
        B = torch.randn(10, 64) * 1e4
        dist = efficient_euclidean_distance(A, B)
        assert not torch.isnan(dist).any(), "NaN from large-magnitude features"

    def test_zero_vectors(self):
        """Zero vectors should produce valid distances."""
        A = torch.zeros(3, 16)
        B = torch.randn(5, 16)
        dist = efficient_euclidean_distance(A, B)
        assert not torch.isnan(dist).any(), "NaN with zero vectors"

    def test_all_distances_non_negative(self):
        """All distances should be >= 0 (no negative from sqrt issue)."""
        A = torch.randn(50, 64)
        B = torch.randn(30, 64)
        dist = efficient_euclidean_distance(A, B)
        assert (dist >= 0).all(), f"Found negative distances: {dist[dist < 0]}"


class TestCosineStability:
    """Numerical stability tests for cosine similarity."""

    def test_normalized_range(self):
        """Cosine similarity should be in [-1, 1]."""
        A = torch.randn(20, 64)
        B = torch.randn(15, 64)
        sim = cosine_similarity_matrix(A, B)
        assert (sim >= -1.0 - 1e-5).all(), f"Similarity below -1: {sim.min()}"
        assert (sim <= 1.0 + 1e-5).all(), f"Similarity above 1: {sim.max()}"

    def test_zero_norm_handling(self):
        """Zero-norm vectors in cosine similarity should produce NaN or be handled."""
        A = torch.zeros(1, 16)  # Zero vector
        B = torch.randn(5, 16)
        sim = cosine_similarity_matrix(A, B)
        # F.normalize handles zero vectors by keeping them as zeros
        # So similarity should be 0, not NaN
        # Note: torch F.normalize returns 0 for zero-norm vectors
        assert not torch.isinf(sim).any(), "Inf with zero-norm vectors"

    def test_identical_vectors(self):
        """Identical vectors should have similarity ≈ 1.0."""
        A = torch.randn(5, 32)
        sim = cosine_similarity_matrix(A, A)
        diag = torch.diag(sim)
        assert torch.allclose(diag, torch.ones(5), atol=1e-5), \
            f"Self-similarity should be ~1.0, got {diag}"

    def test_single_dimension(self):
        """Edge case: single-dimensional features."""
        A = torch.tensor([[3.0]])
        B = torch.tensor([[5.0]])
        sim = cosine_similarity_matrix(A, B)
        assert abs(sim[0, 0].item() - 1.0) < 1e-5, "Same-sign 1D should have sim=1"
