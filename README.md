# Flash-VStream Memory Security / Robustness Research

## What is This?

This repository investigates the robustness properties of bounded visual-memory pipelines for long-video understanding, using [Flash-VStream](https://github.com/IVGSZ/Flash-VStream) (Qwen2VL variant) as a concrete, inspectable memory architecture.

### Research Question

> Can the bounded visual-memory mechanism used by Flash-VStream be systematically interfered with by controlled competing visual information, causing measurable loss, displacement, or misretrieval of target information?

This is an **open research question**. We do not assume the hypothesis is proven.

## What is Flash-VStream?

Flash-VStream introduces a Flash Memory mechanism for efficient long-video understanding using:
- **CSM (Context/Spatial Memory)**: Temporal compression via weighted K-means clustering
- **DAM (Detail/Augmentation Memory)**: Spatial enhancement via nearest-neighbor retrieval from high-weight CSM clusters

The system maintains a bounded memory capacity even for very long video streams. We use it as a well-documented, inspectable baseline — **not as an assumption of current state-of-the-art** (the research landscape has continued to evolve in 2025-2026).

## Repository Structure

```
├── Flash-VStream/              Reference implementation (read-only)
├── research/                   Research code
│   ├── adapters/               Normalized test interfaces for CSM/DAM
│   ├── instrumentation/        Memory state recording
│   ├── metrics/                CSM retention, DAM retrieval, etc.
│   └── interventions/          General intervention framework
├── tests/                      Test suite (CPU-only, lightweight)
│   ├── unit/                   CSM, DAM, boundary, stability tests
│   ├── integration/            Full pipeline tests
│   └── regression/             Frozen reference behavior tests
├── experiments/                Experiment configs and results
├── configs/                    YAML configurations
├── scripts/                    Local checks and Colab bootstrap
├── environment/                Dependency specifications
└── reference/                  Reference manifest with file hashes
```

## Architecture

```
LOCAL (this repo)               GITHUB                 GOOGLE COLAB
code audit / fixes / tests  →   branch + commit    →   GPU execution
                                                        ↓
                                                    GOOGLE DRIVE
                                                    models / results
```

**Local machine is weak** (i3, 8GB, 840M) — no model inference here.
All heavy execution is deferred to Colab.

## How to Run Local Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/ -v --tb=short

# Run specific test suite
pytest tests/unit/test_csm_compress.py -v       # CSM compression tests
pytest tests/unit/test_dam_retrieval.py -v       # DAM retrieval tests
pytest tests/unit/test_cosine_bug.py -v          # Cosine bug regression
pytest tests/integration/test_flash_memory_e2e.py -v  # Integration
```

## How to Run in Colab

1. Clone this repository
2. Run `scripts/colab/bootstrap.py` to verify environment
3. Mount Google Drive
4. Run experiments via config files

See `environment/environment_notes.md` for detailed Colab setup instructions.

## Experiment Stages

| Stage | Description | Status |
|-------|-------------|--------|
| Exp 0 | Memory reproduction (synthetic data) | Ready for local test |
| Exp 1 | Baseline (real model, real video) | Colab required |
| Exp 2 | Controls (irrelevant/similar content) | Colab required |
| Exp 3 | Interference experiments | Colab required |
| Exp 4 | Ablation studies | Colab required |

## Scientific Approach

```
REPRODUCE → UNDERSTAND → INSTRUMENT → CONTROL → INTERVENE → MEASURE → FALSIFY/VALIDATE
```

**H0**: Adding competing content merely changes memory through ordinary compression effects.
**H1**: Certain competing content causes disproportionate target displacement via memory-mechanism interaction.

We test H0 before claiming H1.

## Reference Commit

Repository: `https://github.com/IVGSZ/Flash-VStream.git`
Commit: `8f6bde2f397f4846505df4d03364d97617fc02ee`

## License

Research code is provided for academic research purposes.
Flash-VStream is licensed under Apache 2.0 (see `Flash-VStream/LICENSE`).
