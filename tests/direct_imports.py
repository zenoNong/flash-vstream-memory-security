"""
Direct importers for Flash-VStream reference code.

Bypasses models/__init__.py which requires `transformers` (heavy dependency).
Uses importlib to load specific files directly.
"""

import importlib.util
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_REPO_ROOT, 'Flash-VStream', 'Flash-VStream-Qwen', 'models')


def _load_module_from_file(name, filepath):
    """Load a Python module directly from a file path, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so internal relative imports within the module work
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_compress_functions():
    """Import compress_functions.py directly (no transformers needed)."""
    return _load_module_from_file(
        'compress_functions_direct',
        os.path.join(_MODELS_DIR, 'compress_functions.py')
    )


def get_flash_memory_constants():
    """Import flash_memory_constants.py directly."""
    return _load_module_from_file(
        'flash_memory_constants_direct',
        os.path.join(_MODELS_DIR, 'flash_memory_constants.py')
    )


def get_flash_memory_class():
    """
    Import the FlashMemory class directly from vstream_qwen2vl_model.py.

    This DOES require transformers since FlashMemory is defined in the same
    file as the full model. We use a mock approach instead.
    """
    # First try direct import (requires transformers)
    try:
        mod = _load_module_from_file(
            'vstream_qwen2vl_model_direct',
            os.path.join(_MODELS_DIR, 'vstream_qwen2vl_model.py')
        )
        return mod.FlashMemory
    except (ImportError, ModuleNotFoundError):
        # If transformers not available, return None
        return None


# Pre-load the lightweight modules
compress_functions = get_compress_functions()
flash_memory_constants = get_flash_memory_constants()

# Re-export commonly used functions
weighted_kmeans_ordered_feature = compress_functions.weighted_kmeans_ordered_feature
fast_weighted_kmeans_ordered_feature = compress_functions.fast_weighted_kmeans_ordered_feature
DEFAULT_FLASH_MEMORY_CONFIG = flash_memory_constants.DEFAULT_FLASH_MEMORY_CONFIG
