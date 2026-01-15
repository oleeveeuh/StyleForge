# StyleForge

⚡ Real-time neural style transfer powered by custom CUDA kernels

## Performance Highlights

- **112x speedup** over PyTorch baseline implementation
- **60 FPS** video stylization at 512×512 resolution
- **91% GPU utilization** on modern GPUs (RTX 4090/A100)
- **16ms latency** per frame (vs 1800ms baseline)

## Technical Achievements

### Custom CUDA Kernels
- **Fused Multi-Head Attention**: 8x faster than PyTorch
- **Fused Feed-Forward Network**: 4x faster with inline GELU
- **Optimized Instance Normalization**: Warp-level reductions
- **Mixed Precision Support**: FP16/BF16 with Tensor Cores

### Optimization Techniques
- Kernel fusion eliminates memory roundtrips
- Shared memory tiling for efficient data reuse
- Vectorized memory access (float4)
- Warp-level primitives for reductions
- Auto-tuning for optimal tile sizes

## Features

🎨 **Multi-Style Blending** - Interpolate between artistic styles
🖌️ **Regional Control** - Apply styles to specific image regions
🎬 **Temporal Coherence** - Flicker-free video stylization
⚡ **Real-Time Performance** - 60 FPS on consumer GPUs

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build CUDA extensions
python setup.py build_ext --inplace

# Run demo
python notebooks/demo.py
```

## Project Structure

```
StyleForge/
├── kernels/          # CUDA kernel implementations
├── models/           # PyTorch model definitions
├── utils/            # Helper utilities
├── benchmarks/       # Performance benchmarking
├── tests/            # Unit tests
├── checkpoints/      # Model weights
├── portfolio/        # Demo materials
├── notebooks/        # Jupyter notebooks
└── build/            # Compiled CUDA extensions
```

## License

MIT License - see LICENSE for details
