"""
Test temporal_pool() — Shape and value validation.

temporal_pool expects xdim == 3 * 2 * 14 * 14 = 1176 and poolsize == 2.
It performs spatial restructuring and average pooling.

NOTE: These tests require `transformers` to import FlashMemory.
They will be skipped if transformers is not installed locally.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.conftest import XDIM, DEFAULT_CONFIG
from tests.direct_imports import get_flash_memory_class

FlashMemory = get_flash_memory_class()

# Skip all tests in this module if FlashMemory can't be imported (no transformers)
pytestmark = pytest.mark.skipif(
    FlashMemory is None,
    reason="FlashMemory requires transformers (not installed locally)"
)


class TestTemporalPool:
    """Test temporal_pool shape transformations."""

    def _make_flash_memory(self):
        return FlashMemory(**DEFAULT_CONFIG)

    @pytest.mark.parametrize("T,H,W", [
        (4, 28, 28),
        (8, 28, 28),
        (2, 28, 28),
        (10, 56, 56),
    ])
    def test_output_shape(self, T, H, W):
        """Verify output shape after temporal pooling."""
        fm = self._make_flash_memory()
        P = H // 2 * W // 2 * 2 * 2  # patches per frame
        x = torch.randn(T * P, XDIM)
        thw = torch.tensor([T, H, W], dtype=torch.long)

        out_x, out_thw = fm.temporal_pool(x, thw)

        expected_new_h = H // 2
        expected_new_w = W // 2
        if expected_new_h % 2 != 0:
            expected_new_h += 1
        if expected_new_w % 2 != 0:
            expected_new_w += 1

        new_P = expected_new_h // 2 * expected_new_w // 2 * 2 * 2

        assert out_x.shape == (T * new_P, XDIM), \
            f"Expected ({T * new_P}, {XDIM}), got {out_x.shape}"
        assert out_thw[0].item() == T
        assert out_thw[1].item() == expected_new_h * 2  # stored as grid dims * 2
        assert out_thw[2].item() == expected_new_w * 2

    def test_no_nan_in_output(self):
        """No NaN in pooled output."""
        fm = self._make_flash_memory()
        T, H, W = 4, 28, 28
        P = H // 2 * W // 2 * 2 * 2
        x = torch.randn(T * P, XDIM)
        thw = torch.tensor([T, H, W], dtype=torch.long)

        out_x, _ = fm.temporal_pool(x, thw)
        assert not torch.isnan(out_x).any(), "NaN in temporal_pool output"

    def test_requires_xdim_1176(self):
        """temporal_pool should assert xdim == 3 * 2 * 14 * 14."""
        fm = self._make_flash_memory()
        wrong_xdim = 512
        T, H, W = 4, 28, 28
        P = H // 2 * W // 2 * 2 * 2
        x = torch.randn(T * P, wrong_xdim)
        thw = torch.tensor([T, H, W], dtype=torch.long)

        with pytest.raises(AssertionError):
            fm.temporal_pool(x, thw)
