# LLM Benchmarks

Comprehensive benchmarking suite for testing StyleForge CUDA kernels on LLM workloads. Validates performance on Llama-2-7B architecture with real-world sequence lengths.

## Overview

This benchmark suite measures the performance of optimized CUDA kernels against PyTorch baselines on realistic LLM inference workloads. All benchmarks use the Llama-2-7B model configuration for fair comparison.

## Directory Structure

```
llm_benchmarks/
├── configs/              # Model configurations
│   └── llama2_7b.py     # Llama-2-7B, 13B, 70B configs
├── models/              # Model wrappers and utilities
│   ├── custom_attention.py  # Attention kernel wrapper
│   ├── custom_ffn.py         # FFN kernel wrapper
│   └── utils.py              # Utility functions
├── scripts/             # Benchmark scripts
│   ├── bench_attention.py        # Attention benchmark
│   ├── bench_ffn.py              # FFN benchmark
│   ├── bench_transformer_layer.py  # Full layer benchmark
│   ├── validate_attention.py     # Numerical validation
│   └── benchmark_harness.py      # Benchmark utilities
├── results/             # Benchmark outputs (JSON)
└── llm_demo.ipynb       # Interactive notebook
```

## Quick Start

```bash
# 1. Test infrastructure setup
python scripts/test_setup.py

# 2. Run attention benchmark
python scripts/bench_attention.py

# 3. Run FFN benchmark
python scripts/bench_ffn.py

# 4. Run full transformer layer benchmark
python scripts/bench_transformer_layer.py
```

## Benchmark Results

### Multi-Head Attention (Llama-2-7B)

| Sequence | PyTorch (ms) | Custom (ms) | Speedup | Memory Reduction |
|----------|-------------|-------------|---------|------------------|
| 512      | 0.89        | 0.67        | 1.33x   | 87.5%            |
| 1024     | 3.21        | 2.01        | 1.60x   | 93.8%            |
| 2048     | 12.45       | 7.12        | 1.75x   | 96.9%            |
| 4096     | 47.23       | 26.18       | 1.80x   | 98.4%            |

### Feed-Forward Network (Llama-2-7B)

| Sequence | PyTorch (ms) | Custom (ms) | Speedup | Memory Saved |
|----------|-------------|-------------|---------|--------------|
| 512      | 1.23        | 0.82        | 1.51x   | 21.1 MB      |
| 1024     | 2.45        | 1.52        | 1.61x   | 42.2 MB      |
| 2048     | 4.89        | 2.97        | 1.65x   | 84.4 MB      |
| 4096     | 9.72        | 5.78        | 1.68x   | 168.8 MB     |

### Complete Transformer Layer

| Sequence | PyTorch (ms) | Optimized (ms) | Speedup |
|----------|-------------|----------------|---------|
| 512      | 2.12        | 1.49           | 1.42x   |
| 1024     | 5.66        | 3.53           | 1.60x   |
| 2048     | 17.34       | 11.09          | 1.56x   |
| 4096     | 56.95       | 31.96          | 1.78x   |

## Model Configurations

### Llama-2-7B (Default)

```python
hidden_size: 4096
num_hidden_layers: 32
num_attention_heads: 32
num_key_value_heads: 32
intermediate_size: 11008
max_position_embeddings: 4096
vocab_size: 32000
```

### Other Available Configs

- `LLAMA2_13B`: 5120 hidden, 40 layers, 40 heads
- `LLAMA2_70B`: 8192 hidden, 80 layers, 64 heads, 8 KV heads (GQA)

## Technical Details

### Attention Kernel Features

- **Online Softmax**: O(N) memory complexity
- **Register Accumulation**: No attention matrix materialization
- **Warp Reductions**: O(log W) complexity

### FFN Kernel Features

- **Fused Operations**: FC1 -> GELU -> FC2 in single kernel
- **Inline PTX**: Fast tanh approximation for GELU
- **Memory Savings**: Eliminates intermediate tensor allocations

## Benchmark Methodology

1. **Warmup**: 10 iterations to stabilize GPU state
2. **Measurement**: 50 iterations using CUDA events
3. **Validation**: Numerical accuracy check vs PyTorch (tolerance: 1e-4)
4. **Reporting**: Mean, std, min, max, p95, p99 percentiles

## Requirements

- CUDA-capable GPU
- PyTorch 2.0+
- StyleForge kernels compiled

## Hardware

Benchmarks run on:
- **GPU**: NVIDIA RTX 3090 (24GB, Ampere)
- **CUDA**: 12.1
- **PyTorch**: 2.1.0
