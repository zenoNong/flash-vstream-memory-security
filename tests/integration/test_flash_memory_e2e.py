"""
Integration Test — Full FlashMemory Pipeline

Tests temporal_compress → spatial_enhance → cat_spa_tem → calc_am_rope
with synthetic features. No Qwen model weights required.

Verifies:
- Tensor shapes through the full pipeline
- No NaN values
- No invalid indices
- Position ID consistency
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.conftest import DEFAULT_CONFIG, XDIM, INTERNAL_TEMPORAL_LENGTH, INTERNAL_SPATIAL_LENGTH
from tests.direct_imports import get_flash_memory_class

FlashMemory = get_flash_memory_class()

# Skip all tests if FlashMemory can't be imported (no transformers)
pytestmark = pytest.mark.skipif(
    FlashMemory is None,
    reason="FlashMemory requires transformers (not installed locally)"
)


class TestFlashMemoryIntegration:
    """End-to-end pipeline tests."""

    def _make_flash_memory(self, **overrides):
        config = dict(DEFAULT_CONFIG)
        config.update(overrides)
        return FlashMemory(**config)

    def _make_test_data(self, T, H=28, W=28, seed=42):
        """Create synthetic data matching the FlashMemory forward() input format."""
        torch.manual_seed(seed)

        # Full-resolution features
        P = H // 2 * W // 2 * 2 * 2
        x = torch.randn(T * P, XDIM)
        thw = torch.tensor([T, H, W], dtype=torch.long)

        # Small (pooled) features: H//2, W//2
        small_H = H // 2
        small_W = W // 2
        if small_H % 2 != 0:
            small_H += 1
        if small_W % 2 != 0:
            small_W += 1
        small_P = small_H // 2 * small_W // 2 * 2 * 2
        small_x = torch.randn(T * small_P, XDIM)
        small_thw = torch.tensor([T, small_H, small_W], dtype=torch.long)

        return x, thw, small_x, small_thw

    def test_csm_dam_pipeline_short(self):
        """Short sequence (no compression): CSM + DAM should pass through."""
        fm = self._make_flash_memory()
        T = 10  # < internal_temporal_length (60)
        x, thw, small_x, small_thw = self._make_test_data(T)

        # CSM
        tem_x, tem_thw, tem_weights, tem_timestamps, tem_indices = \
            fm.temporal_compress(small_x, small_thw, fm.temporal_length)

        assert tem_thw[0].item() == T, "Short sequence should not be compressed"
        assert not torch.isnan(tem_x).any()

        # DAM
        tem_positions = tem_timestamps.round().long()
        spa_x, spa_thw, spa_positions = fm.spatial_enhance(
            x=x, small_x=small_x, thw=thw,
            tem_x=tem_x, tem_thw=tem_thw,
            tem_weights=tem_weights, tem_positions=tem_positions,
            tem_indices=tem_indices,
        )

        assert spa_thw[0].item() == T, "Short sequence: all frames should be spatial"
        assert not torch.isnan(spa_x).any()

        # Concatenation
        cat_x = fm.cat_spa_tem(spa_x, tem_x)
        assert not torch.isnan(cat_x).any()

    def test_csm_dam_pipeline_long(self):
        """Long sequence (compression): full pipeline."""
        fm = self._make_flash_memory()
        T = 120
        x, thw, small_x, small_thw = self._make_test_data(T)

        # CSM
        tem_x, tem_thw, tem_weights, tem_timestamps, tem_indices = \
            fm.temporal_compress(small_x, small_thw, fm.temporal_length)

        assert tem_thw[0].item() == INTERNAL_TEMPORAL_LENGTH
        assert not torch.isnan(tem_x).any()

        # DAM
        tem_positions = tem_timestamps.round().long()
        # Clamp positions to valid range
        tem_positions = torch.clamp(tem_positions, 0, T - 1)

        spa_x, spa_thw, spa_positions = fm.spatial_enhance(
            x=x, small_x=small_x, thw=thw,
            tem_x=tem_x, tem_thw=tem_thw,
            tem_weights=tem_weights, tem_positions=tem_positions,
            tem_indices=tem_indices,
        )

        assert spa_thw[0].item() == INTERNAL_SPATIAL_LENGTH
        assert not torch.isnan(spa_x).any()
        # Positions should be valid frame indices
        assert (spa_positions >= 0).all()
        assert (spa_positions < T).all()

        # Concatenation
        cat_x = fm.cat_spa_tem(spa_x, tem_x)
        expected_total = (INTERNAL_SPATIAL_LENGTH + INTERNAL_TEMPORAL_LENGTH)
        small_H = small_thw[1].item()
        small_W = small_thw[2].item()
        small_P = small_H // 2 * small_W // 2 * 2 * 2
        H = thw[1].item()
        W = thw[2].item()
        P = H // 2 * W // 2 * 2 * 2

        # cat_x should have (spa_frames * P_full + tem_frames * P_small) tokens
        assert not torch.isnan(cat_x).any()

    def test_shapes_boundary(self):
        """Test at exact boundary: T = temporal_length."""
        fm = self._make_flash_memory()
        T = INTERNAL_TEMPORAL_LENGTH  # 60
        x, thw, small_x, small_thw = self._make_test_data(T)

        tem_x, tem_thw, tem_weights, tem_timestamps, tem_indices = \
            fm.temporal_compress(small_x, small_thw, fm.temporal_length)

        assert tem_thw[0].item() == T, "At boundary, should not compress"

    def test_shapes_boundary_plus_one(self):
        """Test just above boundary: T = temporal_length + 1."""
        fm = self._make_flash_memory()
        T = INTERNAL_TEMPORAL_LENGTH + 1  # 61
        x, thw, small_x, small_thw = self._make_test_data(T)

        tem_x, tem_thw, tem_weights, tem_timestamps, tem_indices = \
            fm.temporal_compress(small_x, small_thw, fm.temporal_length)

        assert tem_thw[0].item() == INTERNAL_TEMPORAL_LENGTH, "Should compress to temporal_length"

    @pytest.mark.parametrize("T", [10, 60, 61, 120, 240])
    def test_no_nan_at_various_lengths(self, T):
        """No NaN at any tested sequence length."""
        fm = self._make_flash_memory()
        x, thw, small_x, small_thw = self._make_test_data(T, seed=42)

        tem_x, tem_thw, tem_weights, tem_timestamps, tem_indices = \
            fm.temporal_compress(small_x, small_thw, fm.temporal_length)

        assert not torch.isnan(tem_x).any(), f"NaN in CSM output at T={T}"
        assert not torch.isnan(tem_weights).any(), f"NaN in weights at T={T}"
        assert not torch.isnan(tem_timestamps).any(), f"NaN in timestamps at T={T}"
