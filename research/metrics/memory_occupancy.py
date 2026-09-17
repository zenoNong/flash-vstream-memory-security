"""
Memory Occupancy Metrics

Measures slot utilization, cluster sizes, and weight distributions.
"""

import torch
from typing import List, Dict, Any


def slot_counts(cluster_indices: List[List[int]]) -> dict:
    """
    Count temporal slots and their utilization.

    Args:
        cluster_indices: List of lists, each containing frame indices in a cluster

    Returns:
        dict with total_slots, non_empty_slots, empty_slots, sizes
    """
    sizes = [len(members) for members in cluster_indices]
    non_empty = sum(1 for s in sizes if s > 0)

    return {
        'total_slots': len(cluster_indices),
        'non_empty_slots': non_empty,
        'empty_slots': len(cluster_indices) - non_empty,
        'sizes': sizes,
        'mean_size': sum(sizes) / max(len(sizes), 1),
        'max_size': max(sizes) if sizes else 0,
        'min_size': min(sizes) if sizes else 0,
    }


def weight_distribution(weights: torch.Tensor) -> dict:
    """
    Analyze the distribution of CSM temporal weights.

    Args:
        weights: [K] tensor of cluster weights

    Returns:
        dict with distribution statistics
    """
    if weights.numel() == 0:
        return {'empty': True}

    return {
        'mean': weights.mean().item(),
        'std': weights.std().item(),
        'min': weights.min().item(),
        'max': weights.max().item(),
        'median': weights.median().item(),
        'total': weights.sum().item(),
        'num_slots': weights.numel(),
        'normalized_entropy': _normalized_entropy(weights),
    }


def _normalized_entropy(weights: torch.Tensor) -> float:
    """Compute normalized entropy of weight distribution (0=concentrated, 1=uniform)."""
    if weights.numel() <= 1:
        return 0.0
    # Normalize to probability distribution
    probs = weights / weights.sum()
    probs = probs[probs > 0]
    entropy = -(probs * torch.log2(probs)).sum().item()
    max_entropy = torch.log2(torch.tensor(float(weights.numel()))).item()
    return entropy / max_entropy if max_entropy > 0 else 0.0


def memory_capacity_summary(
    temporal_slots: int,
    spatial_slots: int,
    cluster_indices: List[List[int]],
    weights: torch.Tensor,
) -> dict:
    """
    Complete summary of memory utilization.

    Args:
        temporal_slots: Number of temporal (CSM) slots
        spatial_slots: Number of spatial (DAM) slots
        cluster_indices: CSM cluster membership
        weights: CSM weights

    Returns:
        Comprehensive memory state summary
    """
    return {
        'temporal': {
            'configured_slots': temporal_slots,
            'occupancy': slot_counts(cluster_indices),
            'weight_distribution': weight_distribution(weights),
        },
        'spatial': {
            'configured_slots': spatial_slots,
        },
        'total_memory_slots': temporal_slots + spatial_slots,
    }
