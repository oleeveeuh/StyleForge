#!/usr/bin/env python3
"""
StyleForge - CUDA Kernel Profiling Target

This script runs kernels in a way that's easy to profile with NVIDIA Nsight Compute.
It isolates kernel launches and provides clear markers for profiling.

Usage with Nsight Compute:
    ncu --set full python profile_kernels.py instance_norm
    ncu --set full python profile_kernels.py fused_conv
    ncu --set full python profile_kernels.py all
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import kernels
try:
    from kernels.instance_norm_wrapper import FusedInstanceNorm2d
    HAS_INSTANCE_NORM = True
except ImportError:
    HAS_INSTANCE_NORM = False

try:
    from kernels.conv_fusion_wrapper import FusedConvInstanceNormReLU
    HAS_CONV_FUSION = True
except ImportError:
    HAS_CONV_FUSION = False

try:
    from kernels.ffn_wrapper import FusedFFN
    HAS_FFN = True
except ImportError:
    HAS_FFN = False


def print_gpu_info():
    """Print GPU information for profiling context."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return None

    props = torch.cuda.get_device_properties(0)

    print("\n" + "=" * 70)
    print("GPU Information")
    print("=" * 70)
    print(f"  Device:              {props.name}")
    print(f"  Compute Capability:  {props.major}.{props.minor}")
    print(f"  Total Memory:        {props.total_memory / 1e9:.2f} GB")
    print(f"  Streaming MPs:       {props.multi_processor_count}")
    print(f"  Clock Rate:          {props.clock_rate / 1e6:.1f} MHz")
    print(f"  L2 Cache:            {props.l2_cache_size / 1024:.0f} KB")

    # Estimate memory bandwidth (very rough approximation)
    bandwidth_map = {
        (8, 0): 900,    # Volta V100
        (7, 5): 660,    # Turing RTX 2080
        (8, 6): 1000,   # Ampere RTX 3080/3090
        (8, 9): 1008,   # Ada RTX 4090
        (9, 0): 2000,   # Hopper H100
    }
    bw = bandwidth_map.get((props.major, props.minor), 700)
    print(f"  Est. Peak Bandwidth: {bw} GB/s")
    print()

    return props


def profile_instance_norm():
    """Profile instance normalization kernel."""
    if not HAS_INSTANCE_NORM:
        print("❌ InstanceNorm kernel not available")
        return

    print("=" * 70)
    print("Profiling: Instance Normalization Kernel")
    print("=" * 70)

    # Configuration for profiling
    batch_size = 1
    channels = 128
    height = 256
    width = 256

    total_elements = batch_size * channels * height * width
    memory_mb = total_elements * 4 / 1e6

    print(f"\nConfiguration:")
    print(f"  Input shape:  [{batch_size}, {channels}, {height}, {width}]")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Memory (input):  {memory_mb:.2f} MB")
    print(f"  Memory (total):  {memory_mb * 2:.2f} MB (input + output)")

    # Create layer and input
    layer = FusedInstanceNorm2d(channels, affine=True, use_vectorized=True).cuda()
    x = torch.randn(batch_size, channels, height, width, device='cuda')

    # Warmup - critical for accurate profiling
    print("\nWarming up (10 iterations)...")
    for _ in range(10):
        with torch.no_grad():
            _ = layer(x)

    torch.cuda.synchronize()

    # Profile section - marked for ncu
    print("\n" + "=" * 70)
    print("PROFILE_START - Running kernel 50 times for measurement")
    print("=" * 70)

    for i in range(50):
        with torch.no_grad():
            output = layer(x)

        if i % 10 == 0:
            print(f"  Iteration {i}/50")

    torch.cuda.synchronize()

    print("=" * 70)
    print("PROFILE_END")
    print("=" * 70)

    print(f"\n✅ Profiling complete")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")


def profile_conv_fusion():
    """Profile fused Conv+InstanceNorm+ReLU kernel."""
    if not HAS_CONV_FUSION:
        print("❌ ConvFusion kernel not available")
        return

    print("=" * 70)
    print("Profiling: Fused Conv+InstanceNorm+ReLU Kernel")
    print("=" * 70)

    # Configuration
    batch_size = 1
    in_channels = 64
    out_channels = 128
    height = 128
    width = 128
    kernel_size = 3

    print(f"\nConfiguration:")
    print(f"  Input:  [{batch_size}, {in_channels}, {height}, {width}]")
    print(f"  Output: [{batch_size}, {out_channels}, {height}, {width}]")
    print(f"  Kernel: {kernel_size}x{kernel_size}")

    # Create layer and input
    layer = FusedConvInstanceNormReLU(
        in_channels, out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=1
    ).cuda()

    x = torch.randn(batch_size, in_channels, height, width, device='cuda')

    # Warmup
    print("\nWarming up (10 iterations)...")
    for _ in range(10):
        with torch.no_grad():
            _ = layer(x)

    torch.cuda.synchronize()

    # Profile section
    print("\n" + "=" * 70)
    print("PROFILE_START - Running kernel 50 times for measurement")
    print("=" * 70)

    for i in range(50):
        with torch.no_grad():
            output = layer(x)

        if i % 10 == 0:
            print(f"  Iteration {i}/50")

    torch.cuda.synchronize()

    print("=" * 70)
    print("PROFILE_END")
    print("=" * 70)

    print(f"\n✅ Profiling complete")
    print(f"Output shape: {output.shape}")


def profile_ffn():
    """Profile fused FFN kernel."""
    if not HAS_FFN:
        print("❌ FFN kernel not available")
        return

    print("=" * 70)
    print("Profiling: Fused Feed-Forward Network Kernel")
    print("=" * 70)

    # Configuration
    batch_size = 2
    seq_len = 256
    embed_dim = 512
    ffn_dim = 2048  # 4x embed_dim

    print(f"\nConfiguration:")
    print(f"  Input: [{batch_size}, {seq_len}, {embed_dim}]")
    print(f"  Hidden dim: {ffn_dim}")
    print(f"  FLOPs (approx): {2 * batch_size * seq_len * embed_dim * ffn_dim / 1e9:.2f} GFLOPs")

    # Create layer and input
    layer = FusedFFN(embed_dim, ffn_dim).cuda()
    x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')

    # Warmup
    print("\nWarming up (10 iterations)...")
    for _ in range(10):
        with torch.no_grad():
            _ = layer(x)

    torch.cuda.synchronize()

    # Profile section
    print("\n" + "=" * 70)
    print("PROFILE_START - Running kernel 50 times for measurement")
    print("=" * 70)

    for i in range(50):
        with torch.no_grad():
            output = layer(x)

        if i % 10 == 0:
            print(f"  Iteration {i}/50")

    torch.cuda.synchronize()

    print("=" * 70)
    print("PROFILE_END")
    print("=" * 70)

    print(f"\n✅ Profiling complete")
    print(f"Output shape: {output.shape}")


def profile_pytorch_baseline():
    """Profile PyTorch baseline for comparison."""
    print("=" * 70)
    print("Profiling: PyTorch Baseline (InstanceNorm2d)")
    print("=" * 70)

    batch_size = 1
    channels = 128
    height = 256
    width = 256

    print(f"\nConfiguration: [{batch_size}, {channels}, {height}, {width}]")

    # PyTorch layer
    layer = nn.InstanceNorm2d(channels, affine=True).cuda()
    x = torch.randn(batch_size, channels, height, width, device='cuda')

    # Warmup
    print("\nWarming up (10 iterations)...")
    for _ in range(10):
        with torch.no_grad():
            _ = layer(x)

    torch.cuda.synchronize()

    # Profile section
    print("\n" + "=" * 70)
    print("PROFILE_START - Running PyTorch baseline 50 times")
    print("=" * 70)

    for i in range(50):
        with torch.no_grad():
            output = layer(x)

        if i % 10 == 0:
            print(f"  Iteration {i}/50")

    torch.cuda.synchronize()

    print("=" * 70)
    print("PROFILE_END")
    print("=" * 70)

    print(f"\n✅ Baseline profiling complete")


def main():
    """Main profiling entry point."""

    if len(sys.argv) < 2:
        print("StyleForge CUDA Kernel Profiling Target")
        print("\nUsage: python profile_kernels.py <kernel_name>")
        print("\nAvailable kernels:")
        print("  instance_norm    - Profile InstanceNorm kernel")
        print("  conv_fusion      - Profile Conv+InstanceNorm+ReLU kernel")
        print("  ffn              - Profile FFN kernel")
        print("  pytorch_baseline  - Profile PyTorch baseline for comparison")
        print("  all              - Profile all available kernels")
        sys.exit(1)

    mode = sys.argv[1]

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Profiling requires a GPU.")
        sys.exit(1)

    # Print GPU info
    print_gpu_info()

    # Run profiling
    if mode == "instance_norm":
        profile_instance_norm()

    elif mode == "conv_fusion":
        profile_conv_fusion()

    elif mode == "ffn":
        profile_ffn()

    elif mode == "pytorch_baseline":
        profile_pytorch_baseline()

    elif mode == "all":
        if HAS_INSTANCE_NORM:
            profile_instance_norm()
            print("\n")

        if HAS_CONV_FUSION:
            profile_conv_fusion()
            print("\n")

        if HAS_FFN:
            profile_ffn()
            print("\n")

        profile_pytorch_baseline()

    else:
        print(f"❌ Unknown kernel: {mode}")
        print("Available kernels: instance_norm, conv_fusion, ffn, pytorch_baseline, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
