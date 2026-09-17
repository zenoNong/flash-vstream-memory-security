"""
Memory State Recorder

Captures memory state at each stage of the Flash-VStream pipeline
for research analysis. Disabled by default.

Usage:
    recorder = MemoryStateRecorder(enabled=True)
    recorder.record_csm_state(...)
    recorder.record_dam_state(...)
    recorder.save('path/to/output.json')
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CSMState:
    """Recorded state from CSM (temporal compression)."""
    input_temporal_length: int = 0
    output_temporal_length: int = 0
    was_compressed: bool = False
    weights: Optional[List[float]] = None
    timestamps: Optional[List[float]] = None
    membership_indices: Optional[List[List[int]]] = None
    config: Optional[Dict[str, Any]] = None
    random_seed: Optional[int] = None
    dtype: Optional[str] = None
    device: Optional[str] = None


@dataclass
class DAMState:
    """Recorded state from DAM (spatial retrieval)."""
    method: str = ''
    candidate_csm_ids: Optional[List[int]] = None
    candidate_csm_weights: Optional[List[float]] = None
    retrieved_frame_ids: Optional[List[int]] = None
    retrieval_distances: Optional[List[float]] = None
    retrieval_ranks: Optional[List[int]] = None
    spatial_length: int = 0
    retrieval_needed: bool = False


@dataclass
class MemorySnapshot:
    """Complete memory state snapshot for one video/sequence."""
    experiment_id: str = ''
    sequence_id: str = ''
    timestamp_utc: str = ''
    csm: Optional[CSMState] = None
    dam: Optional[DAMState] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryStateRecorder:
    """
    Configurable recorder for Flash-VStream memory states.

    Args:
        enabled: Whether recording is active (default False)
        save_metadata: Whether to save lightweight metadata (default True)
        save_full_tensors: Whether to save full tensor data (default False, saves disk)
    """

    def __init__(self, enabled=False, save_metadata=True, save_full_tensors=False):
        self.enabled = enabled
        self.save_metadata = save_metadata
        self.save_full_tensors = save_full_tensors
        self.snapshots: List[MemorySnapshot] = []
        self._current_csm: Optional[CSMState] = None
        self._current_dam: Optional[DAMState] = None

    def record_csm_state(
        self,
        input_t: int,
        output_t: int,
        weights=None,
        timestamps=None,
        indices=None,
        config=None,
        seed=None,
        dtype=None,
        device=None,
    ):
        """Record CSM compression state."""
        if not self.enabled:
            return

        state = CSMState(
            input_temporal_length=input_t,
            output_temporal_length=output_t,
            was_compressed=(input_t > output_t),
            config=config,
            random_seed=seed,
            dtype=str(dtype) if dtype is not None else None,
            device=str(device) if device is not None else None,
        )

        if self.save_metadata:
            if weights is not None:
                state.weights = weights.detach().cpu().tolist() if hasattr(weights, 'tolist') else list(weights)
            if timestamps is not None:
                state.timestamps = timestamps.detach().cpu().tolist() if hasattr(timestamps, 'tolist') else list(timestamps)
            if indices is not None:
                state.membership_indices = [list(idx) for idx in indices] if indices else None

        self._current_csm = state

    def record_dam_state(
        self,
        method: str,
        candidate_ids=None,
        candidate_weights=None,
        retrieved_ids=None,
        distances=None,
        ranks=None,
        spatial_length=0,
        retrieval_needed=False,
    ):
        """Record DAM retrieval state."""
        if not self.enabled:
            return

        state = DAMState(
            method=method,
            spatial_length=spatial_length,
            retrieval_needed=retrieval_needed,
        )

        if self.save_metadata:
            if candidate_ids is not None:
                state.candidate_csm_ids = list(candidate_ids)
            if candidate_weights is not None:
                state.candidate_csm_weights = (
                    candidate_weights.detach().cpu().tolist()
                    if hasattr(candidate_weights, 'tolist') else list(candidate_weights)
                )
            if retrieved_ids is not None:
                state.retrieved_frame_ids = list(retrieved_ids)
            if distances is not None:
                state.retrieval_distances = (
                    distances.detach().cpu().tolist()
                    if hasattr(distances, 'tolist') else list(distances)
                )
            if ranks is not None:
                state.retrieval_ranks = list(ranks)

        self._current_dam = state

    def commit_snapshot(self, experiment_id='', sequence_id='', config=None, metadata=None):
        """Commit current CSM and DAM states as a snapshot."""
        if not self.enabled:
            return

        snapshot = MemorySnapshot(
            experiment_id=experiment_id,
            sequence_id=sequence_id,
            timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            csm=self._current_csm,
            dam=self._current_dam,
            config=config,
            metadata=metadata,
        )
        self.snapshots.append(snapshot)
        self._current_csm = None
        self._current_dam = None

    def save(self, path, format='json'):
        """Save all recorded snapshots to file."""
        if not self.snapshots:
            return

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        if format == 'json':
            data = [asdict(s) for s in self.snapshots]
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unknown format: {format}")

    def clear(self):
        """Clear all recorded snapshots."""
        self.snapshots.clear()
        self._current_csm = None
        self._current_dam = None

    def __len__(self):
        return len(self.snapshots)
