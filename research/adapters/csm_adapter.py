"""
CSM (Context/Spatial Memory) Adapter

Provides a normalized interface to the reference weighted_kmeans_ordered_feature()
and temporal_compress() functions, handling the return-arity inconsistency documented
in Candidate A of the audit.

DOES NOT modify the reference implementation.
"""

import sys
import os
import torch

# Add reference code to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QWEN_MODELS = os.path.join(_REPO_ROOT, 'Flash-VStream', 'Flash-VStream-Qwen', 'models')
if _QWEN_MODELS not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'Flash-VStream', 'Flash-VStream-Qwen'))


def call_weighted_kmeans_ordered(img_feature, video_max_frames, weights=None, times=None):
    """
    Normalized wrapper around weighted_kmeans_ordered_feature().

    The reference function returns different numbers of values:
      - T <= T0: returns (features, weights, indices) — 3 values
      - T >  T0: returns (features, weights, timestamps, indices) — 4 values

    This adapter always returns a consistent 5-tuple:
      (features, weights, timestamps, indices, was_compressed)

    The 'was_compressed' flag indicates whether actual K-means was run.
    """
    from models.compress_functions import weighted_kmeans_ordered_feature

    T = img_feature.shape[0]
    T0 = video_max_frames

    if T <= T0:
        # Short-sequence path: returns 3 values
        result = weighted_kmeans_ordered_feature(img_feature, video_max_frames, weights=weights, times=times)
        assert len(result) == 3, f"Expected 3 returns for T<=T0 path, got {len(result)}"
        features, out_weights, step_indices = result
        # Construct synthetic timestamps and normalize indices
        timestamps = torch.arange(T, device=features.device, dtype=torch.float32)
        # step_indices is [[[0], [1], ...]] — list of list of lists
        flat_indices = step_indices[0] if len(step_indices) == 1 else step_indices
        return features, out_weights, timestamps, flat_indices, False
    else:
        # Compression path: returns 4 values
        result = weighted_kmeans_ordered_feature(img_feature, video_max_frames, weights=weights, times=times)
        assert len(result) == 4, f"Expected 4 returns for T>T0 path, got {len(result)}"
        features, out_weights, timestamps, step_indices = result
        return features, out_weights, timestamps, step_indices, True


def call_temporal_compress(flash_memory, x, thw, temporal_length=None):
    """
    Wrapper around FlashMemory.temporal_compress() that ensures 5-value return.

    The reference temporal_compress always returns 5 values:
      (x, thw, weights, timestamps, indices)

    This adapter calls it directly but validates the contract.
    """
    if temporal_length is None:
        temporal_length = flash_memory.temporal_length

    result = flash_memory.temporal_compress(x, thw, temporal_length)
    assert len(result) == 5, f"temporal_compress should return 5 values, got {len(result)}"
    tem_x, tem_thw, tem_weights, tem_timestamps, tem_indices = result

    return {
        'features': tem_x,
        'thw': tem_thw,
        'weights': tem_weights,
        'timestamps': tem_timestamps,
        'indices': tem_indices,
        'input_t': thw[0].item(),
        'output_t': tem_thw[0].item(),
        'was_compressed': thw[0].item() > temporal_length,
    }
