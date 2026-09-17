"""
Regression Tests — Frozen Reference Behavior

Guards against accidental behavioral changes in the reference code.
Uses specific seeds and inputs to assert specific outputs.
"""

import os
import sys
import pytest
import torch
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.direct_imports import weighted_kmeans_ordered_feature, DEFAULT_FLASH_MEMORY_CONFIG
from tests.conftest import INTERNAL_TEMPORAL_LENGTH


class TestReferenceIntegrity:
    """Tests that reference code files haven't been modified."""

    def test_reference_files_exist(self):
        """All key reference files should exist."""
        base = os.path.join(os.path.dirname(__file__), '..', '..',
                           'Flash-VStream', 'Flash-VStream-Qwen', 'models')

        expected_files = [
            'compress_functions.py',
            'flash_memory_constants.py',
            'vstream_qwen2vl_model.py',
            'vstream_qwen2vl_realtime.py',
            'vstream_qwen2vl_processor.py',
            '__init__.py',
        ]

        for f in expected_files:
            path = os.path.join(base, f)
            assert os.path.exists(path), f"Reference file missing: {f}"

    def test_default_config_unchanged(self):
        """Default config should match known values."""
        assert DEFAULT_FLASH_MEMORY_CONFIG['flash_memory_temporal_length'] == 120
        assert DEFAULT_FLASH_MEMORY_CONFIG['flash_memory_temporal_method'] == 'kmeans_ordered'
        assert DEFAULT_FLASH_MEMORY_CONFIG['flash_memory_temporal_poolsize'] == 2
        assert DEFAULT_FLASH_MEMORY_CONFIG['flash_memory_temporal_pca_dim'] == 32
        assert DEFAULT_FLASH_MEMORY_CONFIG['flash_memory_spatial_length'] == 60
        assert DEFAULT_FLASH_MEMORY_CONFIG['flash_memory_spatial_method'] == 'klarge_retrieve'


class TestFrozenBehavior:
    """Frozen behavior tests with specific seeds."""

    def test_short_sequence_identity(self):
        """Seed 42, T=10: short path should return identity."""
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        T, T0 = 10, INTERNAL_TEMPORAL_LENGTH
        P, D = 4, 16
        features = torch.randn(T, P, D)

        result = weighted_kmeans_ordered_feature(features, T0)
        assert len(result) == 3

        out_feat, out_weights, out_indices = result
        assert torch.allclose(out_feat, features)
        assert out_feat.shape == (T, P, D)

    def test_compression_output_is_reproducible(self):
        """Same seed + input → same output. Tests K-means reproducibility."""
        T, T0 = 100, INTERNAL_TEMPORAL_LENGTH
        P, D = 4, 16

        outputs = []
        for _ in range(2):
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            features = torch.randn(T, P, D)

            random.seed(456)
            np.random.seed(456)
            torch.manual_seed(456)
            result = weighted_kmeans_ordered_feature(features, T0)
            outputs.append(result)

        assert torch.allclose(outputs[0][0], outputs[1][0], atol=1e-5), \
            "Features should be identical across runs with same seeds"
        assert torch.allclose(outputs[0][1], outputs[1][1], atol=1e-5), \
            "Weights should be identical across runs with same seeds"
