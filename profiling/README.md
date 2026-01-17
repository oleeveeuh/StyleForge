# StyleForge Profiling Directory

This directory contains scripts for profiling CUDA kernels with NVIDIA Nsight Compute.

## Prerequisites

Install NVIDIA Nsight Compute:

```bash
# Ubuntu/Debian
sudo apt-get install nvidia-nsight-compute

# Or download from:
# https://developer.nvidia.com/nsight-compute
```

Verify installation:
```bash
ncu --version
```

## Quick Start

### 1. Profile a Kernel

```bash
# Make script executable
chmod +x profile.sh

# Profile InstanceNorm kernel
./profile.sh instance_norm

# Profile ConvFusion kernel
./profile.sh conv_fusion

# Profile all kernels
./profile.sh all
```

### 2. Analyze Results

```bash
# Analyze the generated CSV metrics
python analyze_profile.py nsight_reports/profile_instance_norm_metrics_*.csv
```

### 3. Generate Portfolio Report

```bash
# Generate a professional markdown report
python generate_profile_report.py \
    nsight_reports/profile_instance_norm_metrics_*_summary.json \
    PROFILING_REPORT.md
```

## Files

| File | Description |
|------|-------------|
| [profile_kernels.py](profile_kernels.py) | Profiling target - runs kernels with markers |
| [profile.sh](profile.sh) | Automated Nsight Compute profiling script |
| [analyze_profile.py](analyze_profile.py) | Parse and analyze ncu CSV output |
| [generate_profile_report.py](generate_profile_report.py) | Generate portfolio-ready reports |

## Profiling Modes

The `profile.sh` script supports multiple profiling modes:

```bash
# Quick profiling (basic metrics only)
./profile.sh instance_norm --quick

# Full profiling (all metrics - default)
./profile.sh instance_norm --full

# Memory bandwidth focused
./profile.sh instance_norm --memory

# Compute utilization focused
./profile.sh instance_norm --compute
```

## Available Kernels

- `instance_norm` - Fused InstanceNorm2d kernel
- `conv_fusion` - Fused Conv2d + InstanceNorm2d + ReLU kernel
- `ffn` - Fused Feed-Forward Network kernel
- `pytorch_baseline` - PyTorch baseline for comparison
- `all` - Profile all available kernels

## Understanding Output

### Nsight Compute Report (.ncu-rep)

Open in Nsight Compute UI for interactive visualization:
```bash
ncu-ui nsight_reports/profile_instance_norm_full_*.ncu-rep
```

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| GPU Utilization | % of peak compute used | > 70% |
| Memory Bandwidth | GB/s achieved | > 500 GB/s (RTX 3080+) |
| Occupancy | % of warps active | > 50% |
| Warp Stalls | % of warps stalled | < 25% |

## Resume-Ready Claims

After successful profiling, you can claim:

- "Achieved X% GPU utilization through kernel fusion and memory optimization"
- "Measured X GB/s memory bandwidth using Nsight Compute"
- "Profiled with NVIDIA Nsight Compute, optimizing warp occupancy and memory access patterns"

## Troubleshooting

### "ncu: command not found"

Install Nsight Compute from NVIDIA's website.

### "No CUDA-capable device is detected"

Ensure you have a CUDA-compatible GPU and drivers installed.

### Profile is empty/incomplete

- Check that the kernel is actually launching (add print statements)
- Ensure sufficient iterations (default: 50)
- Try running with `--quick` mode first
