"""
CSM Retention Metrics

Measures whether target frames/events are retained in the CSM after temporal compression.
"""

import torch
from typing import List, Optional, Tuple


def csm_target_in_cluster(target_frame_id: int, cluster_indices: List[List[int]]) -> Tuple[bool, int]:
    """
    Check if a target frame is represented in any CSM cluster.

    Args:
        target_frame_id: Original frame index to look for
        cluster_indices: List of lists, each containing frame indices in a cluster

    Returns:
        (is_retained, cluster_id) — cluster_id is -1 if not found
    """
    for cluster_id, members in enumerate(cluster_indices):
        if target_frame_id in members:
            return True, cluster_id
    return False, -1


def csm_nearest_centroid(target_features: torch.Tensor, centroids: torch.Tensor) -> dict:
    """
    Find the nearest CSM centroid to a target feature vector.

    Args:
        target_features: [P*D] or [1, P*D] feature vector
        centroids: [K, P*D] centroid features

    Returns:
        dict with 'distance', 'nearest_id', 'rank', 'all_distances'
    """
    if target_features.ndim == 1:
        target_features = target_features.unsqueeze(0)

    dists = torch.cdist(target_features, centroids, p=2).squeeze(0)  # [K]
    nearest_id = torch.argmin(dists).item()
    sorted_ids = torch.argsort(dists)
    rank = (sorted_ids == nearest_id).nonzero(as_tuple=True)[0].item() + 1

    return {
        'distance': dists[nearest_id].item(),
        'nearest_id': nearest_id,
        'rank': rank,
        'all_distances': dists.tolist(),
    }


def csm_weight_for_target(target_cluster_id: int, weights: torch.Tensor) -> float:
    """Get the CSM weight for the cluster containing the target frame."""
    if target_cluster_id < 0 or target_cluster_id >= len(weights):
        return 0.0
    return weights[target_cluster_id].item()


def csm_timestamp_for_target(target_cluster_id: int, timestamps: torch.Tensor) -> float:
    """Get the CSM timestamp for the cluster containing the target frame."""
    if target_cluster_id < 0 or target_cluster_id >= len(timestamps):
        return -1.0
    return timestamps[target_cluster_id].item()


def csm_weight_rank(target_cluster_id: int, weights: torch.Tensor) -> int:
    """
    Get the rank of the target cluster by weight (1 = highest weight).

    Returns -1 if target_cluster_id is invalid.
    """
    if target_cluster_id < 0 or target_cluster_id >= len(weights):
        return -1
    sorted_ids = torch.argsort(weights, descending=True)
    rank = (sorted_ids == target_cluster_id).nonzero(as_tuple=True)[0].item() + 1
    return rank


def csm_cluster_sizes(cluster_indices: List[List[int]]) -> List[int]:
    """Return the size of each cluster."""
    return [len(members) for members in cluster_indices]
