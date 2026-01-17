#!/usr/bin/env python3
"""
StyleForge - Instance Normalization Test Suite

Comprehensive correctness tests for the fused InstanceNorm kernel.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.instance_norm_wrapper import FusedInstanceNorm2d


def test_basic_correctness():
    """Test basic correctness vs PyTorch."""
    print("Test 1: Basic correctness comparison")
    print("-" * 60)

    # Create layers
    pytorch_layer = nn.InstanceNorm2d(64, affine=True).cuda().eval()
    fused_layer = FusedInstanceNorm2d(64, affine=True).cuda().eval()

    # Copy weights
    with torch.no_grad():
        fused_layer.gamma.copy_(pytorch_layer.weight)
        fused_layer.beta.copy_(pytorch_layer.bias)

    # Create input
    x = torch.randn(2, 64, 128, 128).cuda()

    # Forward pass
    with torch.no_grad():
        pytorch_out = pytorch_layer(x)
        fused_out = fused_layer(x)

    # Check correctness
    max_diff = torch.max(torch.abs(pytorch_out - fused_out)).item()
    mean_diff = torch.mean(torch.abs(pytorch_out - fused_out)).item()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pytorch_out.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-4:
        print("✓ PASSED\n")
        return True
    else:
        print("✗ FAILED\n")
        return False


def test_different_sizes():
    """Test different spatial sizes."""
    print("Test 2: Different spatial sizes")
    print("-" * 60)

    all_passed = True

    for h, w in [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256)]:
        print(f"  Testing size {h}x{w}...")

        channels = 64
        pytorch_layer = nn.InstanceNorm2d(channels, affine=True).cuda().eval()
        fused_layer = FusedInstanceNorm2d(channels, affine=True).cuda().eval()

        with torch.no_grad():
            fused_layer.gamma.copy_(pytorch_layer.weight)
            fused_layer.beta.copy_(pytorch_layer.bias)

        x = torch.randn(1, channels, h, w).cuda()

        with torch.no_grad():
            pytorch_out = pytorch_layer(x)
            fused_out = fused_layer(x)

        max_diff = torch.max(torch.abs(pytorch_out - fused_out)).item()

        if max_diff < 1e-4:
            print(f"    ✓ {h}x{w} PASSED")
        else:
            print(f"    ✗ {h}x{w} FAILED (diff={max_diff:.2e})")
            all_passed = False

    print()
    return all_passed


def test_different_channels():
    """Test different channel counts."""
    print("Test 3: Different channel counts")
    print("-" * 60)

    all_passed = True

    for channels in [3, 16, 32, 64, 128, 256]:
        print(f"  Testing {channels} channels...")

        pytorch_layer = nn.InstanceNorm2d(channels, affine=True).cuda().eval()
        fused_layer = FusedInstanceNorm2d(channels, affine=True).cuda().eval()

        with torch.no_grad():
            fused_layer.gamma.copy_(pytorch_layer.weight)
            fused_layer.beta.copy_(pytorch_layer.bias)

        x = torch.randn(1, channels, 64, 64).cuda()

        with torch.no_grad():
            pytorch_out = pytorch_layer(x)
            fused_out = fused_layer(x)

        max_diff = torch.max(torch.abs(pytorch_out - fused_out)).item()

        if max_diff < 1e-4:
            print(f"    ✓ {channels} channels PASSED")
        else:
            print(f"    ✗ {channels} channels FAILED (diff={max_diff:.2e})")
            all_passed = False

    print()
    return all_passed


def test_batch_processing():
    """Test different batch sizes."""
    print("Test 4: Batch processing")
    print("-" * 60)

    all_passed = True

    for batch_size in [1, 2, 4, 8]:
        print(f"  Testing batch_size={batch_size}...")

        channels = 64
        pytorch_layer = nn.InstanceNorm2d(channels, affine=True).cuda().eval()
        fused_layer = FusedInstanceNorm2d(channels, affine=True).cuda().eval()

        with torch.no_grad():
            fused_layer.gamma.copy_(pytorch_layer.weight)
            fused_layer.beta.copy_(pytorch_layer.bias)

        x = torch.randn(batch_size, channels, 64, 64).cuda()

        with torch.no_grad():
            pytorch_out = pytorch_layer(x)
            fused_out = fused_layer(x)

        max_diff = torch.max(torch.abs(pytorch_out - fused_out)).item()

        if max_diff < 1e-4:
            print(f"    ✓ batch_size={batch_size} PASSED")
        else:
            print(f"    ✗ batch_size={batch_size} FAILED (diff={max_diff:.2e})")
            all_passed = False

    print()
    return all_passed


def test_no_affine():
    """Test without affine parameters."""
    print("Test 5: Without affine parameters")
    print("-" * 60)

    pytorch_layer = nn.InstanceNorm2d(64, affine=False).cuda().eval()
    fused_layer = FusedInstanceNorm2d(64, affine=False).cuda().eval()

    x = torch.randn(2, 64, 64, 64).cuda()

    with torch.no_grad():
        pytorch_out = pytorch_layer(x)
        fused_out = fused_layer(x)

    max_diff = torch.max(torch.abs(pytorch_out - fused_out)).item()
    mean_diff = torch.mean(torch.abs(pytorch_out - fused_out)).item()

    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-4:
        print("✓ PASSED\n")
        return True
    else:
        print("✗ FAILED\n")
        return False


def test_vectorized_vs_scalar():
    """Test vectorized vs scalar implementations."""
    print("Test 6: Vectorized vs Scalar comparison")
    print("-" * 60)

    channels = 64
    pytorch_layer = nn.InstanceNorm2d(channels, affine=True).cuda().eval()

    fused_scalar = FusedInstanceNorm2d(channels, affine=True, use_vectorized=False).cuda().eval()
    fused_vec = FusedInstanceNorm2d(channels, affine=True, use_vectorized=True).cuda().eval()

    with torch.no_grad():
        fused_scalar.gamma.copy_(pytorch_layer.weight)
        fused_scalar.beta.copy_(pytorch_layer.bias)
        fused_vec.gamma.copy_(pytorch_layer.weight)
        fused_vec.beta.copy_(pytorch_layer.bias)

    x = torch.randn(2, channels, 128, 128).cuda()

    with torch.no_grad():
        pytorch_out = pytorch_layer(x)
        scalar_out = fused_scalar(x)
        vec_out = fused_vec(x)

    scalar_diff = torch.max(torch.abs(pytorch_out - scalar_out)).item()
    vec_diff = torch.max(torch.abs(pytorch_out - vec_out)).item()
    internal_diff = torch.max(torch.abs(scalar_out - vec_out)).item()

    print(f"Scalar vs PyTorch max diff: {scalar_diff:.2e}")
    print(f"Vectorized vs PyTorch max diff: {vec_diff:.2e}")
    print(f"Scalar vs Vectorized max diff: {internal_diff:.2e}")

    if scalar_diff < 1e-4 and vec_diff < 1e-4:
        print("✓ Both implementations match PyTorch\n")
        return True
    else:
        print("✗ Outputs differ from PyTorch\n")
        return False


def test_edge_cases():
    """Test edge cases."""
    print("Test 7: Edge cases")
    print("-" * 60)

    all_passed = True

    # Very small feature map
    print("  Testing 8x8 feature map...")
    pytorch_layer = nn.InstanceNorm2d(16, affine=True).cuda().eval()
    fused_layer = FusedInstanceNorm2d(16, affine=True).cuda().eval()

    with torch.no_grad():
        fused_layer.gamma.copy_(pytorch_layer.weight)
        fused_layer.beta.copy_(pytorch_layer.bias)

    x = torch.randn(1, 16, 8, 8).cuda()

    with torch.no_grad():
        pytorch_out = pytorch_layer(x)
        fused_out = fused_layer(x)

    max_diff = torch.max(torch.abs(pytorch_out - fused_out)).item()

    if max_diff < 1e-4:
        print("    ✓ 8x8 PASSED")
    else:
        print(f"    ✗ 8x8 FAILED (diff={max_diff:.2e})")
        all_passed = False

    # Odd dimensions
    print("  Testing 63x63 feature map...")
    pytorch_layer = nn.InstanceNorm2d(32, affine=True).cuda().eval()
    fused_layer = FusedInstanceNorm2d(32, affine=True).cuda().eval()

    with torch.no_grad():
        fused_layer.gamma.copy_(pytorch_layer.weight)
        fused_layer.beta.copy_(pytorch_layer.bias)

    x = torch.randn(1, 32, 63, 63).cuda()

    with torch.no_grad():
        pytorch_out = pytorch_layer(x)
        fused_out = fused_layer(x)

    max_diff = torch.max(torch.abs(pytorch_out - fused_out)).item()

    if max_diff < 1e-4:
        print("    ✓ 63x63 PASSED")
    else:
        print(f"    ✗ 63x63 FAILED (diff={max_diff:.2e})")
        all_passed = False

    print()
    return all_passed


def main():
    """Run all tests."""
    print("="*60)
    print("Instance Normalization Test Suite")
    print("="*60)
    print()

    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Tests require a GPU.")
        return False

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    results = []
    results.append(test_basic_correctness())
    results.append(test_different_sizes())
    results.append(test_different_channels())
    results.append(test_batch_processing())
    results.append(test_no_affine())
    results.append(test_vectorized_vs_scalar())
    results.append(test_edge_cases())

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
