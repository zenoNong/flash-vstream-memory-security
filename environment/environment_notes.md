# Environment Notes

## Local Development Machine

- **CPU**: Intel i3
- **RAM**: 8 GB
- **GPU**: GeForce 840M (NOT used for model inference)
- **Python**: System Python 3.14.4 (tests run in venv)
- **Purpose**: Code audit, testing, instrumentation — NO heavy model execution

## Colab Execution Environment

### Required Setup

The official Flash-VStream `setup.sh` specifies:

```bash
conda create -n vstream python=3.10.14
pip install torch==2.6 torchvision==0.21.0
pip install transformers==4.45.0
pip install flash-attn (v2.7.4.post1)
```

### Colab Bootstrap

Run `scripts/colab/bootstrap.py` first. It will verify:

1. Python version (3.10.14)
2. PyTorch version and CUDA build (2.6.0 + cu124)
3. torchvision (0.21.0)
4. transformers (4.45.0)
5. flash_attn version (2.7.4.post1)
6. GPU availability and capability

### Google Drive Layout

```
FlashVStream-Research/
├── 01_environment/          # Environment verification logs
├── 02_models/               # HuggingFace model cache (set HF_HOME here)
├── 03_datasets/             # Video datasets
├── 04_memory_states/        # Saved memory state recordings
├── 05_results/              # Experiment results (JSON)
└── 06_logs/                 # Execution logs
```

### Environment Variables

Set these in Colab before running experiments:

```python
import os
os.environ['PROJECT_ROOT'] = '/content/Adversarial-FlashVStream'
os.environ['DRIVE_ROOT'] = '/content/drive/MyDrive/FlashVStream-Research'
os.environ['HF_HOME'] = os.path.join(os.environ['DRIVE_ROOT'], '02_models')
os.environ['RESULTS_ROOT'] = os.path.join(os.environ['DRIVE_ROOT'], '05_results')
os.environ['LOG_ROOT'] = os.path.join(os.environ['DRIVE_ROOT'], '06_logs')
```

### Dependency Notes

- `flash-attn`: Requires CUDA. Do NOT install a system CUDA toolkit in Colab — use the pre-built wheel matching Colab's CUDA version.
- `peft` and `deepspeed`: Only needed for training. Install from `requirements-colab-optional.txt` if needed.
- `decord`: Video decoding library. Required for loading videos.
- Do NOT install unnecessary training packages for inference-only experiments.

### Known Colab Issues

- Colab runtime may disconnect after extended periods. Save results to Drive frequently.
- Colab's default Python may not be 3.10.14. Use conda or verify the version.
- GPU memory on free Colab is limited (T4 = 16GB). Monitor with `torch.cuda.memory_summary()`.
