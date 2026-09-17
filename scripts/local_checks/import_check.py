"""
Import Check — verifies all research modules can be imported without GPU dependencies.

Usage: python3 scripts/local_checks/import_check.py
"""

import importlib
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODULES_TO_CHECK = [
    'research.instrumentation.memory_recorder',
    'research.metrics.csm_metrics',
    'research.metrics.dam_metrics',
    'research.metrics.representation',
    'research.metrics.memory_occupancy',
    'research.interventions.framework',
    # Adapters require torch but NOT the full model
    'research.adapters.dam_adapter',
]

def check_imports():
    errors = 0
    for module_name in MODULES_TO_CHECK:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {module_name}")
        except Exception as e:
            print(f"  ✗ {module_name}: {e}")
            errors += 1

    return errors

if __name__ == '__main__':
    print("Checking research module imports...")
    errors = check_imports()
    if errors == 0:
        print("  All imports passed.")
    else:
        print(f"  {errors} import(s) failed!")
        sys.exit(1)
