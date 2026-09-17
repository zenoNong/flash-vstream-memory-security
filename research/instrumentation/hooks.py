"""
Non-invasive hook wrappers for FlashMemory.

These hooks intercept FlashMemory's temporal_compress() and spatial_enhance()
to record memory state without modifying the reference code.

Usage:
    from research.instrumentation.hooks import instrument_flash_memory
    from research.instrumentation.memory_recorder import MemoryStateRecorder

    recorder = MemoryStateRecorder(enabled=True)
    flash_memory = FlashMemory(**config)
    instrument_flash_memory(flash_memory, recorder)

    # Now flash_memory.temporal_compress/spatial_enhance will auto-record state
"""

import functools
import torch


def instrument_flash_memory(flash_memory, recorder, experiment_id='', sequence_id=''):
    """
    Monkey-patch a FlashMemory instance to record state via the given recorder.

    Args:
        flash_memory: FlashMemory instance to instrument
        recorder: MemoryStateRecorder instance
        experiment_id: ID for this experiment run
        sequence_id: ID for this video/sequence
    """
    original_temporal_compress = flash_memory.temporal_compress
    original_spatial_enhance = flash_memory.spatial_enhance

    @functools.wraps(original_temporal_compress)
    def hooked_temporal_compress(x, thw, temporal_length):
        result = original_temporal_compress(x, thw, temporal_length)
        tem_x, tem_thw, tem_weights, tem_timestamps, tem_indices = result

        recorder.record_csm_state(
            input_t=thw[0].item(),
            output_t=tem_thw[0].item(),
            weights=tem_weights,
            timestamps=tem_timestamps,
            indices=tem_indices,
            config=flash_memory.config,
            dtype=str(x.dtype),
            device=str(x.device),
        )
        return result

    @functools.wraps(original_spatial_enhance)
    def hooked_spatial_enhance(x, small_x, thw, tem_x, tem_thw, tem_weights, tem_positions, tem_indices):
        result = original_spatial_enhance(x, small_x, thw, tem_x, tem_thw, tem_weights, tem_positions, tem_indices)
        spa_x, spa_thw, spa_positions = result

        recorder.record_dam_state(
            method=flash_memory.spatial_method,
            retrieved_ids=spa_positions.tolist() if hasattr(spa_positions, 'tolist') else list(spa_positions),
            spatial_length=flash_memory.spatial_length,
            retrieval_needed=(thw[0].item() > flash_memory.spatial_length),
        )

        recorder.commit_snapshot(
            experiment_id=experiment_id,
            sequence_id=sequence_id,
            config=flash_memory.config,
        )
        return result

    flash_memory.temporal_compress = hooked_temporal_compress
    flash_memory.spatial_enhance = hooked_spatial_enhance


def remove_hooks(flash_memory):
    """Remove instrumentation hooks by restoring original methods."""
    if hasattr(flash_memory.temporal_compress, '__wrapped__'):
        flash_memory.temporal_compress = flash_memory.temporal_compress.__wrapped__
    if hasattr(flash_memory.spatial_enhance, '__wrapped__'):
        flash_memory.spatial_enhance = flash_memory.spatial_enhance.__wrapped__
