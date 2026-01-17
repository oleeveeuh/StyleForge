# Integration Plan: CUDA Kernels for Fast Style Transfer

Plan for integrating StyleForge's CUDA kernels with Fast Neural Style Transfer.

## Executive Summary

**Current Status:**
- Fast Neural Style Transfer (`TransformerNet`) uses only PyTorch operations
- No custom CUDA kernels are currently used
- Existing kernels (Attention, FFN, InstanceNorm) are incompatible with CNN architecture

**Recommended Approach:**
1. **Phase 1**: Replace InstanceNorm2d (immediate, low-risk)
2. **Phase 2**: Develop convolution kernels (2-week effort)
3. **Phase 3**: Full layer fusion (advanced optimization)

**Expected Results:**
- Phase 1: 1.1-1.5x speedup
- Phase 2: 5-8x speedup
- Phase 3: 10-15x speedup

---

## Phase 1: InstanceNorm Integration (Immediate)

### Objective
Replace all `nn.InstanceNorm2d` with `FusedInstanceNorm2d` from `kernels/instance_norm_wrapper.py`.

### Changes Required

**1. Update `models/transformer_net.py`:**

```python
# At top of file, add import
from kernels.instance_norm_wrapper import FusedInstanceNorm2d

# In ConvLayer.__init__():
# OLD:
self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)

# NEW:
self.norm = FusedInstanceNorm2d(out_channels, affine=True)
```

**2. Update `models/transformer_net.py`:**

```python
# In UpsampleConvLayer.__init__():
# OLD:
self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)

# NEW:
self.norm = FusedInstanceNorm2d(out_channels, affine=True)
```

### Implementation Steps

1. [ ] Import `FusedInstanceNorm2d` in `transformer_net.py`
2. [ ] Replace InstanceNorm in `ConvLayer`
3. [ ] Replace InstanceNorm in `UpsampleConvLayer`
4. [ ] Run verification tests
5. [ ] Benchmark performance improvement

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Numerical mismatch | Low | Low | Test with pre-trained weights |
| CUDA compilation failure | Low | Medium | Graceful fallback to PyTorch |
| Performance regression | Very Low | Low | Benchmarks will verify |

### Timeline

- **Development**: 1-2 hours
- **Testing**: 1 hour
- **Total**: 3 hours

---

## Phase 2: Convolution Kernel Development (2-week sprint)

### Objective
Develop custom CUDA kernels for the convolution operations used in Fast Style Transfer.

### Kernel Specifications

#### Kernel 1: Fused Conv2d + ReflectionPad

```cuda
// Fused convolution with reflection padding
torch::Tensor fused_conv_reflection_pad_forward(
    torch::Tensor input,          // [batch, in_ch, height, width]
    torch::Tensor weight,         // [out_ch, in_ch, kH, kW]
    torch::Tensor bias,           // [out_ch]
    int padding_size,             // Reflection padding amount
    int stride_h, int stride_w    // Stride
);
```

**Features:**
- Reflection padding computed inline (no separate kernel)
- Supports kernel sizes: 3x3, 9x9
- Supports strides: 1, 2
- Vectorized loads (float4)
- Shared memory tiling for weights

#### Kernel 2: Fused Conv2d + InstanceNorm

```cuda
// Fused convolution followed by instance normalization
torch::Tensor fused_conv_instancenorm_forward(
    torch::Tensor input,          // [batch, in_ch, height, width]
    torch::Tensor conv_weight,    // [out_ch, in_ch, kH, kW]
    torch::Tensor conv_bias,      // [out_ch]
    torch::Tensor norm_gamma,     // [out_ch]
    torch::Tensor norm_beta,      // [out_ch]
    float eps,                    // For InstanceNorm
    int padding,                  // Reflection padding
    int stride
);
```

**Features:**
- Single kernel for conv + norm
- Reduced memory traffic (no intermediate output)
- Fused activation (ReLU) optional

#### Kernel 3: Residual Block Kernel

```cuda
// Complete residual block in one kernel
torch::Tensor fused_residual_block_forward(
    torch::Tensor input,          // [batch, ch, h, w]
    torch::Tensor conv1_weight,   // [ch, ch, 3, 3]
    torch::Tensor conv1_bias,     // [ch]
    torch::Tensor conv1_gamma,    // [ch] (InstanceNorm)
    torch::Tensor conv1_beta,     // [ch]
    torch::Tensor conv2_weight,   // [ch, ch, 3, 3]
    torch::Tensor conv2_bias,     // [ch]
    torch::Tensor conv2_gamma,    // [ch]
    torch::Tensor conv2_beta      // [ch]
);
```

**Features:**
- Entire residual block in one kernel
- Residual connection fused
- Activation fused
- Maximum memory efficiency

### Implementation Steps

**Week 1: Basic Convolution Kernel**

1. [ ] Create `kernels/conv2d.cu`
2. [ ] Implement reflection padding logic
3. [ ] Implement basic 3x3 convolution
4. [ ] Add support for 9x9 kernel
5. [ ] Add stride support
6. [ ] Create Python wrapper
7. [ ] Unit tests
8. [ ] Benchmark vs PyTorch

**Week 2: Advanced Fusion**

9. [ ] Implement Conv2d + InstanceNorm fusion
10. [ ] Implement Residual Block kernel
11. [ ] Integrate into TransformerNet
12. [ ] Full model benchmarking
13. [ ] Documentation

### File Structure

```
kernels/
├── conv2d.cu                 # NEW: Convolution kernels
├── conv2d_wrapper.py         # NEW: Python wrapper
├── residual_block.cu         # NEW: Fused residual block
├── residual_block_wrapper.py # NEW: Python wrapper
├── attention.cu              # EXISTING
├── ffn.cu                    # EXISTING
└── instance_norm.cu          # EXISTING
```

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Implementation bugs | Medium | High | Extensive unit tests |
| Numerical differences | Medium | Medium | Compare with PyTorch reference |
| Performance below target | Low | Medium | Profile and optimize |
| Development overruns | Medium | Low | Start with basic kernel |

### Timeline

| Task | Days |
|------|------|
| Basic conv kernel | 3 |
| Reflection padding | 1 |
| Conv + Norm fusion | 2 |
| Residual block | 3 |
| Testing & integration | 3 |
| **Total** | **12** |

---

## Phase 3: Full Model Optimization (Advanced)

### Objective
Complete end-to-end optimization of the entire model with advanced techniques.

### Optimizations

1. **Half-Precision (FP16) Support**
   - Enable Tensor Core usage on Volta/Turing/Ampere
   - 2x speedup potential
   - Requires careful numerical handling

2. **Pipeline Optimization**
   - Overlap computation with memory transfers
   - CUDA streams for concurrent operations
   - 1.2-1.5x additional speedup

3. **Memory Layout Optimization**
   - Channels-last (NHWC) format
   - Better cache utilization
   - 1.1-1.3x speedup

4. **Batch Processing**
   - Optimize for multiple images
   - Shared GPU utilization

### Implementation Steps

1. [ ] Add FP16 support to all kernels
2. [ ] Implement CUDA stream pipeline
3. [ ] Add channels-last format support
4. [ ] Optimize batch processing
5. [ ] End-to-end benchmarking

### Timeline

| Task | Days |
|------|------|
| FP16 support | 4 |
| Pipeline optimization | 3 |
| Memory layout | 2 |
| Batch optimization | 2 |
| Testing | 2 |
| **Total** | **13** |

---

## Alternative: Transformer-based Style Transfer

### Concept
Instead of optimizing the CNN architecture, create a transformer-based style transfer that leverages existing kernels.

### Architecture

```
Input Image
    ↓
Patch Embedding (Conv2d)
    ↓
Positional Encoding
    ↓
┌─────────────────────────────────────────┐
│ Transformer Blocks (N=4)                │
│   ┌─────────────┐    ┌─────────────┐   │
│   │FusedAttn V2│    │  FusedFFN   │   │
│   │   (15x)     │───▶│   (5x)      │   │
│   └─────────────┘    └─────────────┘   │
│   + Layer Norm + Residual               │
└─────────────────────────────────────────┘
    ↓
Patch Decoding
    ↓
Output Image
```

### Advantages
- Uses existing kernels immediately
- 10-20x speedup potential
- Novel architecture (research contribution)

### Disadvantages
- Need to train from scratch
- No pre-trained weights available
- Different artistic results than Johnson et al.

### Timeline: 3-4 weeks

---

## Summary

| Phase | Effort | Speedup | Risk | Recommendation |
|-------|--------|---------|------|----------------|
| **Phase 1** | 3 hours | 1.1-1.5x | Very Low | **DO FIRST** |
| **Phase 2** | 2 weeks | 5-8x | Medium | Worth it for production |
| **Phase 3** | 2 weeks | 10-15x | Medium-High | Advanced optimization |
| **Alternative** | 3-4 weeks | 10-20x | High | Research project |

### Recommended Path

1. **Start with Phase 1** (today, 3 hours)
   - Immediate improvement
   - Validates the integration approach

2. **Evaluate Phase 2 vs Alternative** (after Phase 1)
   - If production speed is needed: Develop conv kernels
   - If research is the goal: Build transformer style transfer

3. **Phase 3** (future)
   - Once architecture is stable, apply advanced optimizations
