#!/usr/bin/env python3
"""
StyleForge - Fused Conv+InstanceNorm+ReLU Test Suite

Quick correctness tests for the conv fusion kernel.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.conv_fusion_wrapper import FusedConvInstanceNormReLU, ResidualBlock


def test_basic_conv_fusion():
    """Test basic Conv+InstanceNorm+ReLU fusion."""
    print("Test 1: Basic Conv+InstanceNorm+ReLU fusion")
    print("-" * 60)

    # Create PyTorch layers
    conv = nn.Conv2d(64, 64, 3, padding=1, bias=True).cuda().eval()
    instance_norm = nn.InstanceNorm2d(64, affine=True).cuda().eval()
    relu = nn.ReLU(inplace=False).cuda().eval()

    # Create fused layer
    fused = FusedConvInstanceNormReLU(64, 64, 3).cuda().eval()

    # Copy weights for fair comparison
    with torch.no_grad():
        fused.weight.copy_(conv.weight)
        fused.bias.copy_(conv.bias)
        fused.gamma.copy_(instance_norm.weight)
        fused.beta.copy_(instance_norm.bias)

    # Create input
    x = torch.randn(1, 64, 64, 64).cuda()

    # Forward pass
    with torch.no_grad():
        pytorch_out = relu(instance_norm(conv(x)))
        fused_out = fused(x)

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


def test_residual_block():
    """Test ResidualBlock with fused layers."""
    print("Test 2: ResidualBlock with fused Conv+InstanceNorm+ReLU")
    print("-" * 60)

    # Create PyTorch residual block
    class PyTorchResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.in1 = nn.InstanceNorm2d(channels, affine=True)
            self.relu1 = nn.ReLU(inplace=False)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.in2 = nn.InstanceNorm2d(channels, affine=True)
            self.relu2 = nn.ReLU(inplace=False)

        def forward(self, x):
            residual = x
            out = self.relu1(self.in1(self.conv1(x)))
            out = self.relu2(self.in2(self.conv2(out)))
            return out + residual

    channels = 128
    pytorch_block = PyTorchResidualBlock(channels).cuda().eval()
    fused_block = ResidualBlock(channels).cuda().eval()

    # Copy weights
    with torch.no_grad():
        fused_block.conv1.weight.copy_(pytorch_block.conv1.weight)
        fused_block.conv1.bias.copy_(pytorch_block.conv1.bias)
        fused_block.conv1.gamma.copy_(pytorch_block.in1.weight)
        fused_block.conv1.beta.copy_(pytorch_block.in1.bias)

        fused_block.conv2.weight.copy_(pytorch_block.conv2.weight)
        fused_block.conv2.bias.copy_(pytorch_block.conv2.bias)
        fused_block.conv2.gamma.copy_(pytorch_block.in2.weight)
        fused_block.conv2.beta.copy_(pytorch_block.in2.bias)

    # Create input
    x = torch.randn(1, 128, 32, 32).cuda()

    # Forward pass
    with torch.no_grad():
        pytorch_out = pytorch_block(x)
        fused_out = fused_block(x)

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


def test_different_kernel_sizes():
    """Test different kernel sizes."""
    print("Test 3: Different kernel sizes")
    print("-" * 60)

    all_passed = True

    for kernel_size, padding in [(1, 0), (3, 1), (5, 2)]:
        print(f"  Testing kernel_size={kernel_size}, padding={padding}...")

        # Create PyTorch layers
        conv = nn.Conv2d(32, 32, kernel_size, padding=padding, bias=True).cuda().eval()
        instance_norm = nn.InstanceNorm2d(32, affine=True).cuda().eval()
        relu = nn.ReLU(inplace=False).cuda().eval()

        # Create fused layer
        fused = FusedConvInstanceNormReLU(
            32, 32, kernel_size, padding=padding
        ).cuda().eval()

        # Copy weights
        with torch.no_grad():
            fused.weight.copy_(conv.weight)
            fused.bias.copy_(conv.bias)
            fused.gamma.copy_(instance_norm.weight)
            fused.beta.copy_(instance_norm.bias)

        # Create input
        x = torch.randn(1, 32, 64, 64).cuda()

        # Forward pass
        with torch.no_grad():
            pytorch_out = relu(instance_norm(conv(x)))
            fused_out = fused(x)

        # Check correctness
        max_diff = torch.max(torch.abs(pytorch_out - fused_out)).item()

        if max_diff < 1e-4:
            print(f"    ✓ kernel_size={kernel_size} PASSED")
        else:
            print(f"    ✗ kernel_size={kernel_size} FAILED (diff={max_diff:.2e})")
            all_passed = False

    print()
    return all_passed


def test_stride():
    """Test stride 2 (downsampling)."""
    print("Test 4: Stride 2 (downsampling)")
    print("-" * 60)

    # Create PyTorch layers
    conv = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=True).cuda().eval()
    instance_norm = nn.InstanceNorm2d(128, affine=True).cuda().eval()
    relu = nn.ReLU(inplace=False).cuda().eval()

    # Create fused layer
    fused = FusedConvInstanceNormReLU(
        64, 128, 3, stride=2, padding=1
    ).cuda().eval()

    # Copy weights
    with torch.no_grad():
        fused.weight.copy_(conv.weight)
        fused.bias.copy_(conv.bias)
        fused.gamma.copy_(instance_norm.weight)
        fused.beta.copy_(instance_norm.bias)

    # Create input
    x = torch.randn(1, 64, 128, 128).cuda()

    # Forward pass
    with torch.no_grad():
        pytorch_out = relu(instance_norm(conv(x)))
        fused_out = fused(x)

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


def main():
    """Run all tests."""
    print("="*60)
    print("Fused Conv+InstanceNorm+ReLU Test Suite")
    print("="*60)
    print()

    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Tests require a GPU.")
        return False

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    results = []
    results.append(test_basic_conv_fusion())
    results.append(test_residual_block())
    results.append(test_different_kernel_sizes())
    results.append(test_stride())

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
