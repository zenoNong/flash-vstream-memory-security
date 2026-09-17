"""
Representation Similarity Metrics

Measures feature-space similarity before and after intervention.
"""

import torch
import torch.nn.functional as F


def cosine_representation_similarity(features_before: torch.Tensor, features_after: torch.Tensor) -> float:
    """
    Compute cosine similarity between two feature representations.

    Args:
        features_before: [D] or [N, D] features before intervention
        features_after: [D] or [N, D] features after intervention

    Returns:
        Mean cosine similarity (scalar)
    """
    if features_before.ndim == 1:
        features_before = features_before.unsqueeze(0)
    if features_after.ndim == 1:
        features_after = features_after.unsqueeze(0)

    sim = F.cosine_similarity(features_before, features_after, dim=-1)
    return sim.mean().item()


def euclidean_representation_distance(features_before: torch.Tensor, features_after: torch.Tensor) -> float:
    """
    Compute mean Euclidean distance between feature representations.

    Args:
        features_before: [N, D] features before intervention
        features_after: [N, D] features after intervention

    Returns:
        Mean Euclidean distance (scalar)
    """
    if features_before.ndim == 1:
        features_before = features_before.unsqueeze(0)
    if features_after.ndim == 1:
        features_after = features_after.unsqueeze(0)

    dists = torch.norm(features_before - features_after, p=2, dim=-1)
    return dists.mean().item()


def representation_shift_summary(
    baseline_centroids: torch.Tensor,
    intervention_centroids: torch.Tensor,
) -> dict:
    """
    Summarize how much CSM centroids shifted between baseline and intervention.

    Args:
        baseline_centroids: [K, D] baseline CSM centroids
        intervention_centroids: [K, D] post-intervention CSM centroids

    Returns:
        dict with mean/max/min shift, cosine similarity
    """
    if baseline_centroids.shape != intervention_centroids.shape:
        return {
            'error': 'Shape mismatch',
            'baseline_shape': list(baseline_centroids.shape),
            'intervention_shape': list(intervention_centroids.shape),
        }

    shifts = torch.norm(baseline_centroids - intervention_centroids, p=2, dim=-1)
    cosine_sims = F.cosine_similarity(baseline_centroids, intervention_centroids, dim=-1)

    return {
        'mean_euclidean_shift': shifts.mean().item(),
        'max_euclidean_shift': shifts.max().item(),
        'min_euclidean_shift': shifts.min().item(),
        'std_euclidean_shift': shifts.std().item(),
        'mean_cosine_similarity': cosine_sims.mean().item(),
        'min_cosine_similarity': cosine_sims.min().item(),
        'num_centroids': baseline_centroids.shape[0],
    }
