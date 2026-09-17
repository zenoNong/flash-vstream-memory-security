"""
Test Cosine Retrieval Bug — Candidate B Regression Test

Proves that:
1. argmin on Euclidean distance retrieves the CORRECT (nearest) frame
2. argmin on cosine similarity retrieves the WRONG (least similar) frame
3. argmax on cosine similarity retrieves the CORRECT (most similar) frame
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


class TestCosineBugRegression:
    """
    Regression test for Candidate B: cosine retrieval in spatial_enhance()
    uses argmin, which retrieves the LEAST similar frame instead of the
    MOST similar frame.

    The default 'klarge_retrieve' (Euclidean) path is NOT affected.
    Only 'klarge_retrieve_cos' (cosine) is affected.
    """

    def _setup_known_scenario(self):
        """
        Create a scenario where we know exactly which frame should be retrieved.

        Centroid: [1, 0, 0, ...]
        Frame A (similar): [0.9, 0.1, 0, ...]  — cosine similarity ≈ 0.99
        Frame B (dissimilar): [-0.5, 0.8, 0.3, ...] — cosine similarity ≈ -0.2

        Correct retrieval: Frame A (most similar)
        Bug behavior: Frame B (least similar, argmin on cosine)
        """
        D = 16
        centroid = torch.zeros(1, D)
        centroid[0, 0] = 1.0

        frame_a = torch.zeros(1, D)  # Similar to centroid
        frame_a[0, 0] = 0.9
        frame_a[0, 1] = 0.1

        frame_b = torch.zeros(1, D)  # Dissimilar to centroid
        frame_b[0, 0] = -0.5
        frame_b[0, 1] = 0.8
        frame_b[0, 2] = 0.3

        candidates = torch.cat([frame_a, frame_b], dim=0)  # [2, D]

        return centroid, candidates

    def test_euclidean_argmin_correct(self):
        """Euclidean argmin should retrieve the nearest (most similar) frame."""
        centroid, candidates = self._setup_known_scenario()

        dist = efficient_euclidean_distance(centroid, candidates)  # [1, 2]
        idx = torch.argmin(dist, dim=1)

        # Frame A (index 0) is closer in Euclidean space
        assert idx[0].item() == 0, \
            f"Euclidean argmin should select frame A (idx 0), got idx {idx[0].item()}"

    def test_cosine_argmin_wrong(self):
        """
        Cosine argmin retrieves the LEAST similar frame — this is the BUG.
        This test documents the bug behavior.
        """
        centroid, candidates = self._setup_known_scenario()

        sim = cosine_similarity_matrix(centroid, candidates)  # [1, 2]
        idx_argmin = torch.argmin(sim, dim=1)

        # argmin on cosine similarity selects the LEAST similar → Frame B (idx 1)
        assert idx_argmin[0].item() == 1, \
            f"Cosine argmin should (incorrectly) select frame B (idx 1), got idx {idx_argmin[0].item()}"

    def test_cosine_argmax_correct(self):
        """
        Cosine argmax retrieves the MOST similar frame — this is the FIX.
        """
        centroid, candidates = self._setup_known_scenario()

        sim = cosine_similarity_matrix(centroid, candidates)  # [1, 2]
        idx_argmax = torch.argmax(sim, dim=1)

        # argmax on cosine similarity selects the MOST similar → Frame A (idx 0)
        assert idx_argmax[0].item() == 0, \
            f"Cosine argmax should correctly select frame A (idx 0), got idx {idx_argmax[0].item()}"

    def test_euclidean_not_affected(self):
        """Confirm that the default Euclidean path works correctly with argmin."""
        torch.manual_seed(42)
        D = 32
        # Create centroid and 10 candidate frames
        centroid = torch.randn(1, D)
        candidates = torch.randn(10, D)

        # Make candidate 3 very close to centroid
        candidates[3] = centroid[0] + torch.randn(D) * 0.01

        dist = efficient_euclidean_distance(centroid, candidates)
        idx = torch.argmin(dist, dim=1)

        assert idx[0].item() == 3, \
            f"Euclidean should retrieve the deliberately-close frame (idx 3), got {idx[0].item()}"

    def test_reference_code_uses_argmin_for_both(self):
        """
        Document that the reference code uses argmin for both Euclidean and cosine.
        This test reads the source to verify the bug exists in the reference.
        """
        ref_path = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'Flash-VStream', 'Flash-VStream-Qwen', 'models', 'vstream_qwen2vl_model.py'
        )

        with open(ref_path, 'r') as f:
            source = f.read()

        # The reference code has exactly one argmin in spatial_enhance
        # which is applied regardless of whether Euclidean or cosine is used
        assert 'torch.argmin(dist, dim=1)' in source, \
            "Reference code should use argmin (documenting the bug)"
        assert 'cosine_similarity' in source, \
            "Reference code should define cosine_similarity function"
        assert 'klarge_retrieve_cos' in source, \
            "Reference code should reference klarge_retrieve_cos method"
