# Benchmark Methodology

## Testing Environment

### Hardware
- **GPU**: NVIDIA RTX 3090 (24GB GDDR6X)
- **Architecture**: Ampere (GA102)
- **Compute Capability**: 8.6
- **Memory Bandwidth**: ~936 GB/s

### Software
- **CUDA**: 12.1
- **Driver**: 530.30.02
- **PyTorch**: 2.1.0+cu121
- **Python**: 3.10.12

## Measurement Approach

### Timing Method
All benchmarks use CUDA events for accurate GPU timing:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# ... kernel execution ...
end.record()

torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

### Warmup Iterations
10 warmup iterations are performed before measurement to:
- Stabilize GPU clock speeds
- Ensure all memory is allocated
- Allow JIT compilation to complete

### Benchmark Iterations
50 measurements are collected to compute statistics:
- Mean: Average execution time
- Std: Standard deviation
- Min/Max: Best/worst case
- p95/p99: 95th/99th percentiles

## Model Configurations

### Llama-2-7B
```
hidden_size: 4096
num_hidden_layers: 32
num_attention_heads: 32
intermediate_size: 11008
vocab_size: 32000
max_position_embeddings: 4096
```

## Validation

All kernels are validated against PyTorch reference implementations:

| Kernel | Tolerance | Status |
|--------|-----------|--------|
| Attention V3 | rtol=1e-3, atol=1e-4 | PASS |
| FFN | rtol=1e-3, atol=1e-4 | PASS |

## Running Benchmarks

### Quick Start
```bash
cd llm_benchmarks

# Run all benchmarks
python scripts/bench_attention.py
python scripts/bench_ffn.py
python scripts/bench_transformer_layer.py
```

### Custom Configuration
```python
from configs.llama2_7b import get_config
from scripts.bench_attention import benchmark_attention_single_config

# Use Llama-2-13B config
config = get_config("13b")

result = benchmark_attention_single_config(
    seq_len=1024,
    config=config,
    batch_size=1
)
```

## Interpreting Results

### Speedup Calculation
```
speedup = baseline_time / optimized_time
```
A speedup of 1.5x means the optimized version is 50% faster.

### Memory Savings
For attention, memory reduction is calculated as:
```
standard_memory = batch * heads * seq_len * seq_len * 4 bytes
custom_memory = batch * heads * seq_len * head_dim * 4 bytes
reduction_pct = 100 * (1 - custom_memory / standard_memory)
```

### Throughput
TFLOP/s is calculated as:
```
flops = 2 * batch * seq_len * hidden * intermediate  # For FFN
tflops = flops / (time_ms / 1000) / 1e12
```

## Known Limitations

1. **Single GPU**: All benchmarks on single RTX 3090
2. **FP32 Only**: Mixed precision benchmarks coming soon
3. **Batch Size 1**: Larger batches may show different results
4. **No Caching**: KV cache benefits not measured

## Future Improvements

- [ ] Add Flash Attention 2 comparison
- [ ] Benchmark with mixed precision (FP16/BF16)
- [ ] Multi-GPU scaling tests
- [ ] Power consumption measurements
- [ ] Profiling with Nsight Compute
