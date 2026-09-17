"""
General Intervention Framework

Provides a structured way to run baseline vs. intervention comparisons
on the Flash-VStream memory pipeline.

This framework does NOT hard-code a single attack type.
It supports general independent variables.
"""

import json
import os
import time
import copy
from typing import Any, Callable, Dict, List, Optional

import yaml


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML or JSON."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"Unknown config format: {config_path}")


class ExperimentResult:
    """Container for experiment results."""

    def __init__(self, experiment_id: str, config: dict):
        self.experiment_id = experiment_id
        self.config = config
        self.baseline_state: Optional[dict] = None
        self.intervention_state: Optional[dict] = None
        self.comparison: Optional[dict] = None
        self.metadata: dict = {
            'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }

    def to_dict(self) -> dict:
        return {
            'experiment_id': self.experiment_id,
            'config': self.config,
            'baseline_state': self.baseline_state,
            'intervention_state': self.intervention_state,
            'comparison': self.comparison,
            'metadata': self.metadata,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class InterventionExperiment:
    """
    Framework for running controlled intervention experiments.

    Conceptually:
        baseline video + controlled competing content
        → same Flash-VStream pipeline
        → compare memory states

    Independent variables that can be controlled:
        - competition_amount: how much competing content
        - competition_duration: temporal extent
        - temporal_location: where in the video
        - feature_similarity: how similar to target
        - repetition: how many times repeated
        - density: concentration of competing content
        - distance_from_target: temporal gap to target event
    """

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None):
        if config_path is not None:
            self.config = load_config(config_path)
        elif config is not None:
            self.config = config
        else:
            self.config = {}

        self.results: List[ExperimentResult] = []

    def run_baseline(self, features, pipeline_fn: Callable, **kwargs) -> dict:
        """
        Run the baseline (no intervention) through the pipeline.

        Args:
            features: Input feature tensor [T, P, D]
            pipeline_fn: Function that runs the memory pipeline and returns state dict
            **kwargs: Additional pipeline arguments

        Returns:
            dict of baseline memory state
        """
        return pipeline_fn(features, **kwargs)

    def run_with_intervention(
        self,
        features,
        intervention_fn: Callable,
        pipeline_fn: Callable,
        **kwargs,
    ) -> dict:
        """
        Apply intervention to features, then run through pipeline.

        Args:
            features: Input feature tensor [T, P, D]
            intervention_fn: Function that modifies features (returns modified tensor)
            pipeline_fn: Function that runs the memory pipeline
            **kwargs: Additional arguments

        Returns:
            dict of post-intervention memory state
        """
        modified_features = intervention_fn(features)
        return pipeline_fn(modified_features, **kwargs)

    def compare(self, baseline_state: dict, intervention_state: dict) -> dict:
        """
        Compare baseline and intervention memory states.

        Returns a comparison dict. The actual metrics used depend on
        what's available in the state dicts.
        """
        comparison = {
            'baseline_csm_length': baseline_state.get('csm_output_t', None),
            'intervention_csm_length': intervention_state.get('csm_output_t', None),
        }

        # Compare weights if available
        if 'csm_weights' in baseline_state and 'csm_weights' in intervention_state:
            import torch
            bw = baseline_state['csm_weights']
            iw = intervention_state['csm_weights']
            if isinstance(bw, list):
                bw = torch.tensor(bw)
            if isinstance(iw, list):
                iw = torch.tensor(iw)
            if bw.shape == iw.shape:
                comparison['weight_correlation'] = torch.corrcoef(
                    torch.stack([bw.float(), iw.float()])
                )[0, 1].item()

        return comparison

    def run_experiment(
        self,
        experiment_id: str,
        features,
        pipeline_fn: Callable,
        intervention_fn: Optional[Callable] = None,
        **kwargs,
    ) -> ExperimentResult:
        """
        Run a complete experiment: baseline + optional intervention + comparison.

        Args:
            experiment_id: Unique experiment identifier
            features: Input features
            pipeline_fn: Memory pipeline function
            intervention_fn: Optional intervention function
            **kwargs: Additional arguments

        Returns:
            ExperimentResult with all recorded data
        """
        result = ExperimentResult(experiment_id, self.config)

        # Baseline
        result.baseline_state = self.run_baseline(features, pipeline_fn, **kwargs)

        # Intervention (if provided)
        if intervention_fn is not None:
            result.intervention_state = self.run_with_intervention(
                features, intervention_fn, pipeline_fn, **kwargs
            )
            result.comparison = self.compare(result.baseline_state, result.intervention_state)

        self.results.append(result)
        return result

    def save_all_results(self, output_dir: str):
        """Save all experiment results to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        for result in self.results:
            path = os.path.join(output_dir, f"{result.experiment_id}.json")
            result.save(path)
