"""
DAM Retrieval Metrics

Measures whether target frames are successfully retrieved by the DAM
and provides distance/rank analysis.
"""

import torch
from typing import List, Optional


def dam_target_retrieved(target_frame_id: int, retrieved_ids: List[int]) -> bool:
    """Check if the target frame was retrieved by DAM."""
    return target_frame_id in retrieved_ids


def dam_retrieval_rank(target_frame_id: int, distances: torch.Tensor, centroid_idx: int) -> int:
    """
    Get the rank of the target frame among all candidates for a given centroid.

    Args:
        target_frame_id: Original frame index
        distances: [t] distances from the centroid to all frames
        centroid_idx: Not used directly, kept for API consistency

    Returns:
        Rank (1 = nearest). Returns -1 if target_frame_id is out of range.
    """
    if target_frame_id >= len(distances):
        return -1
    target_dist = distances[target_frame_id]
    rank = (distances <= target_dist).sum().item()
    return rank


def dam_competing_distance(
    target_features: torch.Tensor,
    competitor_features: torch.Tensor,
    centroid: torch.Tensor,
) -> dict:
    """
    Compare distance of target vs competitor to a centroid.

    Args:
        target_features: [P*D] target frame features
        competitor_features: [P*D] competitor frame features
        centroid: [P*D] CSM centroid features

    Returns:
        dict with target_distance, competitor_distance, target_closer (bool)
    """
    target_dist = torch.dist(target_features, centroid, p=2).item()
    competitor_dist = torch.dist(competitor_features, centroid, p=2).item()

    return {
        'target_distance': target_dist,
        'competitor_distance': competitor_dist,
        'target_closer': target_dist < competitor_dist,
        'distance_ratio': target_dist / max(competitor_dist, 1e-8),
    }


def dam_retrieval_coverage(
    target_frame_ids: List[int],
    retrieved_ids: List[int],
) -> dict:
    """
    Measure what fraction of target frames were retrieved.

    Args:
        target_frame_ids: List of frame IDs that should be retrieved
        retrieved_ids: List of frame IDs that were actually retrieved

    Returns:
        dict with coverage, retrieved_targets, missed_targets
    """
    target_set = set(target_frame_ids)
    retrieved_set = set(retrieved_ids)
    retrieved_targets = target_set & retrieved_set
    missed_targets = target_set - retrieved_set

    return {
        'coverage': len(retrieved_targets) / max(len(target_set), 1),
        'retrieved_targets': sorted(list(retrieved_targets)),
        'missed_targets': sorted(list(missed_targets)),
        'total_targets': len(target_set),
        'total_retrieved': len(retrieved_set),
    }
