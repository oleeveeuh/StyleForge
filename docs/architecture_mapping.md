# Architecture Mapping: Style Transfer Kernels

Mapping Fast Neural Style Transfer operations to available CUDA kernels.

## Fast Neural Style Transfer Architecture

The `TransformerNet` model (Johnson et al., 2016) consists of:

```
Input (3, H, W)
    ↓
┌─────────────────────────────────────┐
│ Encoder                              │
│   ConvLayer(3→32, 9x9, stride=1)    │
│   ConvLayer(32→64, 3x3, stride=2)   │
│   ConvLayer(64→128, 3x3, stride=2)  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Residual Blocks (5 blocks)          │
│   Each: ConvLayer → ConvLayer → +   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Decoder                              │
│   UpsampleConvLayer(128→64)         │
│   UpsampleConvLayer(64→32)          │
│   ConvLayer(32→3, 9x9)              │
└─────────────────────────────────────┘
    ↓
Output (3, H, W)
```

## Layer Decomposition

### ConvLayer

```python
class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        self.pad = nn.ReflectionPad2d(padding)    # ← No kernel
        self.conv = nn.Conv2d(in_ch, out_ch, ...)  # ← No kernel
        self.norm = nn.InstanceNorm2d(out_ch)      # ← HAS KERNEL!
        self.activation = nn.ReLU(inplace=True)    # ← No kernel (trivial)
```

### ResidualBlock

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        self.conv1 = ConvLayer(channels, channels, 3, 1, 1)
        self.conv2 = ConvLayer(channels, channels, 3, 1, 1)
        # residual = x; out = conv2(conv1(x)); return residual + out
```

### UpsampleConvLayer

```python
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upsample):
        self.upsample = nn.Upsample(scale_factor=2)  # ← No kernel
        self.pad = nn.ReflectionPad2d(padding)        # ← No kernel
        self.conv = nn.Conv2d(in_ch, out_ch, ...)     # ← No kernel
        self.norm = nn.InstanceNorm2d(out_ch)         # ← HAS KERNEL!
        self.activation = nn.ReLU(inplace=True)       # ← No kernel
```

## Operation-to-Kernel Mapping

| Operation | Count per Model | CUDA Kernel | Speedup Potential |
|-----------|-----------------|-------------|-------------------|
| ReflectionPad2d | 16 | ❌ None | Low (simple op) |
| Conv2d (3x3) | ~12 | ❌ None | **HIGH** (80% of time) |
| Conv2d (9x9) | 2 | ❌ None | **HIGH** (80% of time) |
| InstanceNorm2d | 16 | ✅ FusedInstanceNorm2d | 3-5x |
| ReLU | ~16 | ⚠️ Fused in FFN only | Low |
| Upsample (nearest) | 2 | ❌ None | Low |
| Residual (+) | 7 | ⚠️ Fused in FFN only | Low |

## Kernel Compatibility Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Style Transfer → CUDA Kernels                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TransformerNet Layer          Available CUDA Kernel               │
│  ──────────────────          ──────────────────────────            │
│                                                                     │
│  ┌─────────────┐              ┌──────────────┐                     │
│  │ Reflection  │              │   NONE       │                     │
│  │   Pad2d     │─────────────▶│  (could be   │                     │
│  │             │              │   fused)     │                     │
│  └─────────────┘              └──────────────┘                     │
│        │                                                         │
│        ▼                                                         │
│  ┌─────────────┐              ┌──────────────┐                     │
│  │   Conv2d    │              │   NONE       │  ◀── MISSING!       │
│  │  (3x3/9x9)  │─────────────▶│  (critical   │                     │
│  │             │              │   bottleneck)│                     │
│  └─────────────┘              └──────────────┘                     │
│        │                                                         │
│        ▼                                                         │
│  ┌─────────────┐              ┌──────────────┐                     │
│  │ Instance    │              │   YES!       │  ◀── DROP-IN READY   │
│  │   Norm2d    │─────────────▶│ FusedInst... │                     │
│  │             │              │   (3-5x)     │                     │
│  └─────────────┘              └──────────────┘                     │
│        │                                                         │
│        ▼                                                         │
│  ┌─────────────┐              ┌──────────────┐                     │
│  │    ReLU     │              │   Fused      │                     │
│  │             │─────────────▶│  (in FFN)    │                     │
│  └─────────────┘              └──────────────┘                     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                           Fused Attention                           │
│  (Not used in Fast Style Transfer, but available in StyleForge)    │
│                                                                     │
│  ┌─────────────┐              ┌──────────────┐                     │
│  │ Multi-head  │              │   YES!       │  ◀── 8-20x          │
│  │ Attention   │─────────────▶│ FusedAttn... │                     │
│  │             │              │   (V2: 15x)  │                     │
│  └─────────────┘              └──────────────┘                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Performance Breakdown Estimate

Based on typical style transfer models:

```
Total inference time: 100%
│
├─ Conv2d operations:     80%  ◀── NO KERNEL (major bottleneck!)
├─ InstanceNorm2d:        10%  ◀── HAS KERNEL (3-5x speedup)
├─ Memory operations:      5%
└─ Other (ReLU, pad, +):   5%
```

**With current kernels:** ~10-15% overall speedup (InstanceNorm only)

**With conv kernels:** ~5-8x overall speedup

## Opportunities for Kernel Development

### High Priority

1. **Fused Conv2d + InstanceNorm2d**
   - Combine convolution and normalization
   - Reduce memory traffic (one pass instead of two)
   - Estimated speedup: 2-3x over separate ops

2. **Fused Conv2d + ReflectionPad**
   - Implement padding within convolution kernel
   - Avoid extra memory allocation
   - Estimated speedup: 1.2-1.5x

3. **Residual Block Kernel**
   - Fuse: Conv → Norm → ReLU → Conv → Norm → Add
   - Single kernel for entire residual block
   - Estimated speedup: 3-4x over PyTorch

### Medium Priority

4. **Upsample + Conv Fusion**
   - Combine nearest-neighbor upsampling with convolution
   - Avoid intermediate memory allocation

### Low Priority

5. **Standalone ReLU kernel**
   - Not worth it (too simple)
   - Better to fuse with other operations

## Transformer vs CNN Style Transfer

### Transformer-based Style Transfer (future)

Could leverage existing attention kernel:

```
Patch Embedding → Positional Encoding → Transformer Blocks → Decode
                                    ↓
                            ┌─────────────────┐
                            │ FusedAttention  │ ← USES OUR KERNEL!
                            │   (8-20x)       │
                            └─────────────────┘
                                    ↓
                            ┌─────────────────┐
                            │   FusedFFN      │ ← USES OUR KERNEL!
                            │   (4-5x)        │
                            └─────────────────┘
```

### CNN-based Fast Style Transfer (current)

Cannot use attention/FFN kernels - needs convolution kernels:

```
Input → Conv → Norm → ReLU → Conv → Norm → ReLU → ... → Output
         ↓
      NO KERNEL!
```

## Conclusion

**Immediate Integration:**
- Replace `nn.InstanceNorm2d` with `FusedInstanceNorm2d`
- Effort: 1-2 hours
- Speedup: 1.1-1.5x overall

**Maximum Potential (with new conv kernels):**
- Develop fused Conv2d kernels
- Effort: 1-2 weeks
- Speedup: 5-8x overall

**Alternative approach:**
- Develop transformer-based style transfer
- Leverages existing attention/FFN kernels
- Effort: 2-3 weeks
- Speedup: 10-20x overall (with existing kernels)
