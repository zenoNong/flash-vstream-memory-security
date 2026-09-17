"""
Test DAM Retrieval — Experiment 0 (DAM Reproduction)

Tests spatial_enhance() with synthetic data where nearest frames are known.
Verifies CSM centroid → DAM retrieval → correct nearest frame.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from research.adapters.dam_adapter import (
    efficient_euclidean_distance,
    cosine_similarity_matrix,
    research_spatial_enhance,
)


class TestEuclideanDistance:
    """Test the Euclidean distance function."""

    def test_identical_vectors_zero_distance(self):
        A = torch.randn(3, 16)
        dist = efficient_euclidean_distance(A, A)
        # Diagonal should be near-zero — the expanded formula (A²+B²-2AB)
        # has known floating-point cancellation, so we use relaxed tolerance.
        # This is the Candidate E issue: our research adapter clamps negative values.
        for i in range(3):
            assert dist[i, i].item() < 0.01, f"Self-distance should be ~0, got {dist[i,i]}"

    def test_known_distance(self):
        A = torch.tensor([[0.0, 0.0]])
        B = torch.tensor([[3.0, 4.0]])
        dist = efficient_euclidean_distance(A, B)
        assert abs(dist[0, 0].item() - 5.0) < 1e-5, f"Distance should be 5.0, got {dist[0,0]}"

    def test_no_nan_near_identical(self):
        """Near-identical vectors should not produce NaN (Candidate E)."""
        A = torch.randn(10, 64)
        B = A + torch.randn_like(A) * 1e-7  # Very small perturbation
        dist = efficient_euclidean_distance(A, B)
        assert not torch.isnan(dist).any(), "NaN in distances for near-identical vectors"

    def test_no_nan_identical(self):
        """Identical vectors must not produce NaN from negative sqrt."""
        A = torch.randn(10, 64)
        dist = efficient_euclidean_distance(A, A)
        assert not torch.isnan(dist).any(), "NaN in self-distance (negative sqrt issue)"


class TestCosineSimilarity:
    """Test the cosine similarity function."""

    def test_identical_vectors_similarity_one(self):
        A = torch.randn(3, 16)
        sim = cosine_similarity_matrix(A, A)
        for i in range(3):
            assert abs(sim[i, i].item() - 1.0) < 1e-5, f"Self-similarity should be ~1.0"

    def test_orthogonal_vectors(self):
        A = torch.tensor([[1.0, 0.0]])
        B = torch.tensor([[0.0, 1.0]])
        sim = cosine_similarity_matrix(A, B)
        assert abs(sim[0, 0].item()) < 1e-5, "Orthogonal vectors should have ~0 similarity"

    def test_opposite_vectors(self):
        A = torch.tensor([[1.0, 0.0]])
        B = torch.tensor([[-1.0, 0.0]])
        sim = cosine_similarity_matrix(A, B)
        assert abs(sim[0, 0].item() + 1.0) < 1e-5, "Opposite vectors should have ~-1 similarity"


class TestDAMRetrieval:
    """Test DAM retrieval with synthetic known-nearest-frame scenario."""

    def _make_dam_test_data(self, T=100, spatial_length=10, num_csm=20, seed=42):
        """
        Create synthetic data where we know exactly which frames should be retrieved.

        Strategy: Create T frames. CSM centroids are copies of specific frames
        (the "target" frames). DAM should retrieve those exact frames.
        """
        torch.manual_seed(seed)
        H, W = 8, 8  # Small for testing
        xdim = 32     # Small feature dim
        P_full = H // 2 * W // 2 * 2 * 2
        P_small = H // 4 * W // 4 * 2 * 2

        # Full-resolution features
        x = torch.randn(T * P_full, xdim)
        thw = torch.tensor([T, H, W], dtype=torch.long)

        # Small (pooled) features — different from full-res
        small_x = torch.randn(T * P_small, xdim)
        small_thw = torch.tensor([T, H // 2, W // 2], dtype=torch.long)

        # CSM output: take centroids as exact copies of certain small_x frames
        target_frames = torch.randperm(T)[:num_csm]
        small_x_3d = small_x.reshape(T, P_small, xdim)
        tem_x = small_x_3d[target_frames].reshape(-1, xdim)  # CSM centroids = exact frame features
        tem_thw = torch.tensor([num_csm, H // 2, W // 2], dtype=torch.long)

        # Weights: random but positive
        tem_weights = torch.rand(num_csm) + 0.1
        tem_positions = target_frames.float()

        return {
            'x': x, 'thw': thw,
            'small_x': small_x, 'small_thw': small_thw,
            'tem_x': tem_x, 'tem_thw': tem_thw,
            'tem_weights': tem_weights,
            'tem_positions': tem_positions,
            'target_frames': target_frames,
            'spatial_length': spatial_length,
        }

    def test_euclidean_retrieves_correct_frames(self):
        """When centroids ARE exact copies of frames, Euclidean should retrieve those frames."""
        data = self._make_dam_test_data(T=100, spatial_length=10, num_csm=20, seed=42)

        result = research_spatial_enhance(
            x=data['x'], small_x=data['small_x'], thw=data['thw'],
            tem_x=data['tem_x'], tem_thw=data['tem_thw'],
            tem_weights=data['tem_weights'], tem_positions=data['tem_positions'],
            spatial_length=data['spatial_length'],
            spatial_method='klarge_retrieve',
        )

        retrieved = result['spa_positions']
        metadata = result['retrieval_metadata']

        assert metadata['retrieval_needed'] is True
        assert len(retrieved) == data['spatial_length']

        # The retrieved frames should match the target frames for the top-weight centroids
        sorted_idx = torch.argsort(data['tem_weights'], descending=True)
        top_k = sorted_idx[:data['spatial_length']]
        expected_frames = data['target_frames'][top_k]

        for i, (ret, exp) in enumerate(zip(retrieved, expected_frames)):
            assert ret.item() == exp.item(), \
                f"Retrieval {i}: got frame {ret.item()}, expected {exp.item()}"

    def test_cosine_retrieves_correct_frames(self):
        """When centroids ARE exact copies of frames, cosine should also retrieve those frames."""
        data = self._make_dam_test_data(T=100, spatial_length=10, num_csm=20, seed=42)

        result = research_spatial_enhance(
            x=data['x'], small_x=data['small_x'], thw=data['thw'],
            tem_x=data['tem_x'], tem_thw=data['tem_thw'],
            tem_weights=data['tem_weights'], tem_positions=data['tem_positions'],
            spatial_length=data['spatial_length'],
            spatial_method='klarge_retrieve_cos',
        )

        retrieved = result['spa_positions']
        sorted_idx = torch.argsort(data['tem_weights'], descending=True)
        top_k = sorted_idx[:data['spatial_length']]
        expected_frames = data['target_frames'][top_k]

        for i, (ret, exp) in enumerate(zip(retrieved, expected_frames)):
            assert ret.item() == exp.item(), \
                f"Cosine retrieval {i}: got frame {ret.item()}, expected {exp.item()}"

    def test_short_sequence_no_retrieval(self):
        """When T <= spatial_length, all frames should be kept without retrieval."""
        data = self._make_dam_test_data(T=5, spatial_length=10, num_csm=5, seed=42)

        result = research_spatial_enhance(
            x=data['x'], small_x=data['small_x'], thw=data['thw'],
            tem_x=data['tem_x'], tem_thw=data['tem_thw'],
            tem_weights=data['tem_weights'], tem_positions=data['tem_positions'],
            spatial_length=data['spatial_length'],
            spatial_method='klarge_retrieve',
        )

        assert result['retrieval_metadata']['retrieval_needed'] is False
        assert result['spa_thw'][0].item() == 5

    def test_retrieval_metadata_populated(self):
        """Metadata dict should contain all expected fields."""
        data = self._make_dam_test_data(T=100, spatial_length=10, num_csm=20, seed=42)

        result = research_spatial_enhance(
            x=data['x'], small_x=data['small_x'], thw=data['thw'],
            tem_x=data['tem_x'], tem_thw=data['tem_thw'],
            tem_weights=data['tem_weights'], tem_positions=data['tem_positions'],
            spatial_length=data['spatial_length'],
            spatial_method='klarge_retrieve',
        )

        meta = result['retrieval_metadata']
        assert 'selected_csm_ids' in meta
        assert 'retrieved_frame_ids' in meta
        assert 'retrieval_details' in meta
        assert len(meta['retrieval_details']) == data['spatial_length']
