"""
Colab Bootstrap Script

Verifies the Colab environment matches expected versions and sets up
all necessary paths and directories.

Usage (in Colab):
    !python scripts/colab/bootstrap.py

Or import and call:
    from scripts.colab.bootstrap import verify_environment, setup_drive
"""

import os
import sys
import json
import time


# Expected versions from setup.sh
EXPECTED = {
    'python': '3.10',           # Major.minor match
    'torch': '2.6.0',
    'torch_cuda': '12.4',
    'torchvision': '0.21.0',
    'transformers': '4.45.0',
    'flash_attn': '2.7.4',      # Prefix match (post1 suffix varies)
}


def verify_environment():
    """
    Verify all critical package versions.
    Returns (all_pass, results_dict).
    """
    results = {}
    all_pass = True

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    match = py_ver == EXPECTED['python']
    results['python'] = {'expected': EXPECTED['python'], 'actual': py_ver, 'match': match}
    if not match:
        all_pass = False

    # PyTorch
    try:
        import torch
        torch_ver = torch.__version__.split('+')[0]
        match = torch_ver == EXPECTED['torch']
        results['torch'] = {'expected': EXPECTED['torch'], 'actual': torch_ver, 'match': match}
        if not match:
            all_pass = False

        # CUDA build
        cuda_ver = torch.version.cuda or 'None'
        match = cuda_ver.startswith(EXPECTED['torch_cuda'])
        results['torch_cuda'] = {'expected': EXPECTED['torch_cuda'], 'actual': cuda_ver, 'match': match}
        if not match:
            all_pass = False

        # GPU
        if torch.cuda.is_available():
            results['gpu'] = {
                'name': torch.cuda.get_device_name(0),
                'capability': list(torch.cuda.get_device_capability(0)),
                'memory_gb': round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1),
            }
        else:
            results['gpu'] = {'error': 'CUDA not available'}
            all_pass = False
    except ImportError:
        results['torch'] = {'error': 'torch not installed'}
        all_pass = False

    # torchvision
    try:
        import torchvision
        tv_ver = torchvision.__version__.split('+')[0]
        match = tv_ver == EXPECTED['torchvision']
        results['torchvision'] = {'expected': EXPECTED['torchvision'], 'actual': tv_ver, 'match': match}
        if not match:
            all_pass = False
    except ImportError:
        results['torchvision'] = {'error': 'torchvision not installed'}
        all_pass = False

    # transformers
    try:
        import transformers
        tf_ver = transformers.__version__
        match = tf_ver == EXPECTED['transformers']
        results['transformers'] = {'expected': EXPECTED['transformers'], 'actual': tf_ver, 'match': match}
        if not match:
            all_pass = False
    except ImportError:
        results['transformers'] = {'error': 'transformers not installed'}
        all_pass = False

    # flash_attn
    try:
        import flash_attn
        fa_ver = flash_attn.__version__
        match = fa_ver.startswith(EXPECTED['flash_attn'])
        results['flash_attn'] = {'expected': EXPECTED['flash_attn'], 'actual': fa_ver, 'match': match}
        if not match:
            all_pass = False
    except ImportError:
        results['flash_attn'] = {'error': 'flash_attn not installed'}
        all_pass = False

    return all_pass, results


def setup_drive(drive_root=None):
    """
    Set up Google Drive directory structure.

    Args:
        drive_root: Root path on Drive. If None, uses DRIVE_ROOT env var.
    """
    if drive_root is None:
        drive_root = os.environ.get('DRIVE_ROOT', '/content/drive/MyDrive/FlashVStream-Research')

    dirs = [
        '01_environment',
        '02_models',
        '03_datasets',
        '04_memory_states',
        '05_results',
        '06_logs',
    ]

    for d in dirs:
        path = os.path.join(drive_root, d)
        os.makedirs(path, exist_ok=True)
        print(f"  ✓ {path}")

    # Set HF_HOME to Drive for persistent model cache
    hf_home = os.path.join(drive_root, '02_models')
    os.environ['HF_HOME'] = hf_home
    os.environ['DRIVE_ROOT'] = drive_root
    os.environ['RESULTS_ROOT'] = os.path.join(drive_root, '05_results')
    os.environ['LOG_ROOT'] = os.path.join(drive_root, '06_logs')

    print(f"\n  HF_HOME = {hf_home}")
    print(f"  DRIVE_ROOT = {drive_root}")

    return drive_root


def save_environment_log(output_dir=None):
    """Save environment verification results to a JSON file."""
    all_pass, results = verify_environment()

    log = {
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'all_pass': all_pass,
        'versions': results,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'environment_verification.json')
        with open(path, 'w') as f:
            json.dump(log, f, indent=2)
        print(f"\nSaved to {path}")

    return log


if __name__ == '__main__':
    print("=" * 60)
    print("Flash-VStream Research — Environment Verification")
    print("=" * 60)

    all_pass, results = verify_environment()

    for key, info in results.items():
        if 'error' in info:
            print(f"  ✗ {key}: {info['error']}")
        elif 'match' in info:
            status = "✓" if info['match'] else "✗"
            print(f"  {status} {key}: expected={info['expected']}, actual={info['actual']}")
        else:
            print(f"  ℹ {key}: {info}")

    print()
    if all_pass:
        print("ALL CHECKS PASSED ✓")
    else:
        print("SOME CHECKS FAILED ✗")
        print("The environment may not match the expected Flash-VStream setup.")
        print("Check the results above and install missing/mismatched packages.")
        sys.exit(1)
