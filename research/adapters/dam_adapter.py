"""
DAM (Detail/Augmentation Memory) Adapter

Provides a research-friendly interface to FlashMemory.spatial_enhance()
that exposes retrieval metadata (distances, ranks, selected indices).

Also provides a corrected cosine retrieval path (Candidate B fix).

DOES NOT modify the reference implementation.
"""

import sys
import os
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QWEN_MODELS = os.path.join(_REPO_ROOT, 'Flash-VStream', 'Flash-VStream-Qwen', 'models')
if _QWEN_MODELS not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'Flash-VStream', 'Flash-VStream-Qwen'))


def efficient_euclidean_distance(A, B):
    """Compute pairwise Euclidean distance. Returns [A_rows, B_rows] distance matrix."""
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1]
    A_2 = torch.sum(A ** 2, dim=1, keepdim=True)
    B_2 = torch.sum(B ** 2, dim=1, keepdim=True)
    AB = A @ B.T
    dists_2 = A_2 + B_2.T - 2 * AB
    # Clamp to avoid NaN from floating-point errors (Candidate E)
    dists_2 = torch.clamp(dists_2, min=0.0)
    return torch.sqrt(dists_2)


def cosine_similarity_matrix(A, B):
    """Compute pairwise cosine similarity. Returns [A_rows, B_rows] similarity matrix."""
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1]
    A_norm = F.normalize(A, p=2, dim=-1)
    B_norm = F.normalize(B, p=2, dim=-1)
    return torch.matmul(A_norm, B_norm.T)


def research_spatial_enhance(
    x, small_x, thw, tem_x, tem_thw, tem_weights, tem_positions,
    spatial_length, spatial_method='klarge_retrieve'
):
    """
    Research-instrumented DAM retrieval.

    Differences from reference:
    1. Returns detailed retrieval metadata
    2. Fixes cosine retrieval to use argmax (Candidate B)
    3. Clamps Euclidean distance to avoid NaN (Candidate E)

    Returns:
        dict with keys:
            spa_x: retrieved spatial features
            spa_thw: spatial grid dimensions
            spa_positions: frame indices of retrieved features
            retrieval_metadata: dict with distances, ranks, selected CSM IDs, etc.
    """
    t, h, w = thw
    xdim = x.shape[-1]
    x = x.reshape(t, h // 2 * w // 2 * 2 * 2, xdim)
    small_x = small_x.reshape(t, h // 4 * w // 4 * 2 * 2, xdim)
    st, sh, sw = tem_thw
    tem_x = tem_x.reshape(st, sh // 2 * sw // 2 * 2 * 2, xdim)

    metadata = {
        'method': spatial_method,
        'input_t': t,
        'spatial_length': spatial_length,
        'csm_length': st,
    }

    if t <= spatial_length:
        spa_x = x
        spa_positions = torch.arange(t, device=x.device).long()
        metadata['retrieval_needed'] = False
    else:
        metadata['retrieval_needed'] = True

        if spatial_method == 'sample':
            idx = torch.linspace(0, t - 1, spatial_length, device=x.device).round().long()
            spa_x = x[idx]
            spa_positions = idx
        elif spatial_method == 'nearest':
            sorted_indices = torch.argsort(tem_weights, descending=True)
            klarge_indices = sorted_indices[:spatial_length]
            idx = tem_positions[klarge_indices]
            spa_x = x[idx]
            spa_positions = idx
            metadata['selected_csm_ids'] = klarge_indices.tolist()
        elif spatial_method in ('klarge_retrieve', 'klarge_retrieve_cos'):
            centroids = tem_x.reshape(st, -1)
            sorted_indices = torch.argsort(tem_weights, descending=True)
            klarge_indices = sorted_indices[:spatial_length]
            selected_centroids = centroids[klarge_indices]  # [spatial_length, P*D]
            flat_small_x = small_x.reshape(t, -1)

            if spatial_method == 'klarge_retrieve':
                dist = efficient_euclidean_distance(selected_centroids, flat_small_x)  # [k, t]
                idx = torch.argmin(dist, dim=1)  # Correct: smallest distance = most similar
            elif spatial_method == 'klarge_retrieve_cos':
                sim = cosine_similarity_matrix(selected_centroids, flat_small_x)  # [k, t]
                idx = torch.argmax(sim, dim=1)  # FIXED: largest similarity = most similar
                dist = 1.0 - sim  # Convert to distance for metadata

            spa_x = x[idx]
            spa_positions = idx

            # Detailed retrieval metadata
            metadata['selected_csm_ids'] = klarge_indices.tolist()
            metadata['selected_csm_weights'] = tem_weights[klarge_indices].tolist()
            metadata['retrieved_frame_ids'] = idx.tolist()

            # Compute per-retrieval distance info
            retrieval_distances = []
            for i in range(spatial_length):
                d = dist[i]
                retrieved_dist = d[idx[i]].item()
                rank = (d <= retrieved_dist).sum().item() if spatial_method == 'klarge_retrieve' else (d >= retrieved_dist).sum().item()
                retrieval_distances.append({
                    'csm_id': klarge_indices[i].item(),
                    'retrieved_frame': idx[i].item(),
                    'distance': retrieved_dist,
                    'rank': rank,
                })
            metadata['retrieval_details'] = retrieval_distances
        else:
            raise ValueError(f"Unknown spatial_method: {spatial_method}")

    spa_thw = thw.clone()
    spa_thw[0] = spa_x.shape[0]

    return {
        'spa_x': spa_x,
        'spa_thw': spa_thw,
        'spa_positions': spa_positions,
        'retrieval_metadata': metadata,
    }
