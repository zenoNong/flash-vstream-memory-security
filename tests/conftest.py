"""
Shared test fixtures for Flash-VStream memory pipeline testing.

Provides synthetic data generators, FlashMemory instantiation helpers,
and seed management.
"""

import os
import sys
import pytest
import torch
import random
import numpy as np

# Add project root and reference code to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_ROOT = os.path.join(PROJECT_ROOT, 'Flash-VStream', 'Flash-VStream-Qwen')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, QWEN_ROOT)


# --- Default Configuration ---

DEFAULT_CONFIG = dict(
    flash_memory_temporal_length=120,
    flash_memory_temporal_method='kmeans_ordered',
    flash_memory_temporal_poolsize=2,
    flash_memory_temporal_pca_dim=32,
    flash_memory_spatial_length=60,
    flash_memory_spatial_method='klarge_retrieve',
)

# Internal lengths (halved by FlashMemory.__init__)
INTERNAL_TEMPORAL_LENGTH = 60
INTERNAL_SPATIAL_LENGTH = 30

# Qwen2VL feature dimension: 3 channels * 2 temporal * 14 * 14 spatial
XDIM = 3 * 2 * 14 * 14  # 1176


@pytest.fixture
def default_config():
    """Return a copy of the default Flash Memory configuration."""
    return dict(DEFAULT_CONFIG)


@pytest.fixture
def seed_everything():
    """Fixture that sets all random seeds to 42 for determinism."""
    def _seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        return seed
    return _seed


def make_synthetic_features(T, H=28, W=28, xdim=XDIM, diversity='random', seed=42, device='cpu'):
    """
    Generate synthetic feature tensors mimicking Flash-VStream's internal format.

    Args:
        T: Number of temporal frames
        H: Height grid (default 28 = 14*2 for Qwen2VL)
        W: Width grid (default 28 = 14*2 for Qwen2VL)
        xdim: Feature dimension (default 1176 = 3*2*14*14)
        diversity: 'random' | 'structured' | 'low_diversity'
        seed: Random seed
        device: torch device

    Returns:
        x: [T * H//2 * W//2 * 2 * 2, xdim] flat feature tensor
        thw: [3] tensor (T, H, W)

    The shape follows the reference code's internal format where features
    are flattened as [T * spatial_patches, xdim].
    """
    torch.manual_seed(seed)

    P = H // 2 * W // 2 * 2 * 2  # patches per frame after spatial merge

    if diversity == 'random':
        # Fully random features — high diversity
        features = torch.randn(T, P, xdim, device=device)
    elif diversity == 'structured':
        # Features with temporal structure — nearby frames are similar
        base = torch.randn(T, P, xdim, device=device) * 0.1
        trend = torch.linspace(0, 1, T, device=device).view(T, 1, 1).expand(T, P, xdim)
        features = base + trend
    elif diversity == 'low_diversity':
        # Very few unique features — triggers K-means unique-point fallback
        unique = torch.randn(5, P, xdim, device=device)
        indices = torch.randint(0, 5, (T,))
        features = unique[indices]
    else:
        raise ValueError(f"Unknown diversity mode: {diversity}")

    # Flatten to reference format: [T * P, xdim]
    x = features.reshape(-1, xdim)
    thw = torch.tensor([T, H, W], dtype=torch.long)

    return x, thw


def make_small_features(T, H=28, W=28, xdim=XDIM, seed=42, device='cpu'):
    """
    Generate small (pooled) features for DAM retrieval.

    After temporal_pool with poolsize=2, spatial dims are halved.
    Small features have H//2, W//2.
    """
    torch.manual_seed(seed)
    small_H = H // 2
    small_W = W // 2
    if small_H % 2 != 0:
        small_H += 1
    if small_W % 2 != 0:
        small_W += 1

    P = small_H // 2 * small_W // 2 * 2 * 2
    features = torch.randn(T, P, xdim, device=device)
    x = features.reshape(-1, xdim)
    thw = torch.tensor([T, small_H, small_W], dtype=torch.long)
    return x, thw


def make_features_3d(T, P, D, diversity='random', seed=42, device='cpu'):
    """
    Generate 3D features [T, P, D] for direct use with compression functions.

    This is the format expected by weighted_kmeans_ordered_feature() etc.
    """
    torch.manual_seed(seed)

    if diversity == 'random':
        return torch.randn(T, P, D, device=device)
    elif diversity == 'structured':
        base = torch.randn(T, P, D, device=device) * 0.1
        trend = torch.linspace(0, 1, T, device=device).view(T, 1, 1).expand(T, P, D)
        return base + trend
    elif diversity == 'low_diversity':
        unique = torch.randn(min(5, T), P, D, device=device)
        indices = torch.randint(0, min(5, T), (T,))
        return unique[indices]
    else:
        raise ValueError(f"Unknown diversity: {diversity}")
