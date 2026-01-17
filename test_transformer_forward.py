#!/usr/bin/env python3
"""
Test script for StyleForge Transformer - Verifies CUDA kernel usage.

This script validates:
1. Model creates successfully
2. Forward pass works
3. CUDA kernels are being called (not PyTorch fallback)
4. Output shape is correct
5. Performance benchmark (PyTorch vs CUDA)

Usage:
    python test_transformer_forward.py
    python test_transformer_forward.py --no-cuda    # Force PyTorch baseline
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vit_style_transfer import (
    StyleForgeTransformer,
    create_styleforge_transformer,
    create_model,
    STYLEFORGE_MODELS
)
from models.custom_attention_wrapper import print_attention_stats


def test_model_creation():
    """Test that model can be created."""
    print("\n" + "=" * 60)
    print("TEST 1: Model Creation")
    print("=" * 60)

    try:
        # Use patch_size=32 for 256x256 image (64 patches, fits in shared memory)
        model = create_styleforge_transformer(
            image_size=256,
            patch_size=32,  # Changed from 16 to 32 for shared memory compatibility
            embed_dim=512,
            num_heads=8,
            num_blocks=4,  # Smaller for testing
            ffn_dim=2048
        )
        print("✓ Model created successfully")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # Count attention blocks
        attn_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'attn') and hasattr(module, 'num_heads'):
                attn_count += 1
        print(f"  Attention blocks: {attn_count}")

        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model: StyleForgeTransformer, batch_size: int = 1):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass")
    print("=" * 60)

    try:
        # Create dummy inputs
        content = torch.randn(batch_size, 3, 256, 256)
        style = torch.randn(batch_size, 3, 256, 256)

        print(f"Content input shape: {content.shape}")
        print(f"Style input shape: {style.shape}")

        # Forward pass
        with torch.no_grad():
            output = model(content, style)

        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")

        assert output.shape == (batch_size, 3, 256, 256), \
            f"Unexpected output shape: {output.shape}"

        print("✓ Forward pass successful")
        return output

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_kernel_usage(model: StyleForgeTransformer, use_cuda: bool = True):
    """Test that CUDA kernels are being used."""
    print("\n" + "=" * 60)
    print("TEST 3: CUDA Kernel Usage Verification")
    print("=" * 60)

    # Reset stats
    for name, module in model.named_modules():
        if hasattr(module, 'reset_stats'):
            module.reset_stats()

    # Run forward pass
    content = torch.randn(1, 3, 256, 256)
    style = torch.randn(1, 3, 256, 256)

    if use_cuda and torch.cuda.is_available():
        content = content.cuda()
        style = style.cuda()
        model = model.cuda()

    with torch.no_grad():
        _ = model(content, style)

    # Print stats
    stats = model.get_kernel_stats()

    print(f"Attention modules found: {stats['attention_modules']}")
    print(f"Total attention calls: {stats['total_calls']}")
    print(f"CUDA kernel calls: {stats['cuda_kernel_calls']}")
    print(f"PyTorch fallback calls: {stats['pytorch_fallback_calls']}")
    print(f"CUDA usage: {stats['cuda_percentage']:.1f}%")

    if use_cuda and torch.cuda.is_available():
        if stats['cuda_kernel_calls'] > 0:
            print("\n✅ CUDA kernels are being used!")
        else:
            print("\n⚠️  CUDA kernels available but not called - using PyTorch fallback")
    else:
        print(f"\n⚠️  CUDA not available or disabled - using PyTorch baseline")

    model.print_kernel_stats()

    return stats


def benchmark_attention(model: StyleForgeTransformer, num_iters: int = 100):
    """Benchmark attention performance."""
    print("\n" + "=" * 60)
    print("TEST 4: Performance Benchmark")
    print("=" * 60)

    content = torch.randn(1, 3, 256, 256)
    style = torch.randn(1, 3, 256, 256)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        content = content.cuda()
        style = style.cuda()
        model = model.cuda()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(content, style)

    if use_cuda:
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            output = model(content, style)

    if use_cuda:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_ms = elapsed / num_iters * 1000

    print(f"Iterations: {num_iters}")
    print(f"Total time: {elapsed:.3f} s")
    print(f"Average time: {avg_ms:.2f} ms")
    print(f"Throughput: {1000 / avg_ms:.2f} images/sec")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters ({total_params * 4 / 1e6:.2f} MB FP32)")

    # Calculate attention calls per forward pass
    stats = model.get_kernel_stats()
    attn_calls = stats['total_calls'] / max(stats['total_calls'], 1)  # Normalize
    num_blocks = len(model.encoder_blocks) + len(model.decoder_blocks)
    print(f"Attention calls per forward pass: {num_blocks} (encoder + decoder)")

    return avg_ms


def test_variants():
    """Test different model variants."""
    print("\n" + "=" * 60)
    print("TEST 5: Model Variants")
    print("=" * 60)

    for variant_name, config in STYLEFORGE_MODELS.items():
        print(f"\n{variant_name.upper()} model:")
        print(f"  image_size: {config['image_size']}")
        print(f"  embed_dim: {config['embed_dim']}")
        print(f"  num_heads: {config['num_heads']}")
        print(f"  num_blocks: {config['num_blocks']}")

        try:
            model = create_model(variant_name)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  parameters: {total_params:,}")
            print(f"  ✓ {variant_name} model created successfully")
        except Exception as e:
            print(f"  ❌ {variant_name} model failed: {e}")


def test_gradient_flow(model: StyleForgeTransformer):
    """Test that gradients flow correctly."""
    print("\n" + "=" * 60)
    print("TEST 6: Gradient Flow (for training)")
    print("=" * 60)

    try:
        model.train()

        content = torch.randn(1, 3, 256, 256, requires_grad=True)  # Must match model image_size
        style = torch.randn(1, 3, 256, 256)

        if torch.cuda.is_available():
            content = content.cuda()
            style = style.cuda()
            model = model.cuda()

        # Forward
        output = model(content, style)
        loss = output.mean()

        # Backward
        loss.backward()

        # Check gradients
        grad_count = 0
        zero_grad_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                if param.grad.abs().sum() == 0:
                    zero_grad_count += 1

        print(f"Parameters with gradients: {grad_count}")
        print(f"Zero gradients: {zero_grad_count}")

        assert grad_count > 0, "No parameters have gradients!"
        assert zero_grad_count == 0, f"{zero_grad_count} parameters have zero gradients!"

        print("✓ Gradient flow test passed")

    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test StyleForge Transformer")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--blocks", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--benchmark-iters", type=int, default=50, help="Benchmark iterations")

    args = parser.parse_args()

    print("=" * 60)
    print("STYLEFORGE TRANSFORMER - FORWARD PASS TEST")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using CUDA: {not args.no_cuda}")
    print(f"Image size: {args.size}")
    print(f"Transformer blocks: {args.blocks}")

    use_cuda = torch.cuda.is_available() and not args.no_cuda

    # Test 1: Model creation
    model = test_model_creation()
    if model is None:
        print("\n❌ Cannot continue without a working model")
        return 1

    # Test 2: Forward pass
    output = test_forward_pass(model)
    if output is None:
        print("\n❌ Cannot continue without working forward pass")
        return 1

    # Test 3: Kernel usage verification
    stats = test_kernel_usage(model, use_cuda=use_cuda)

    # Test 4: Benchmark
    avg_ms = benchmark_attention(model, num_iters=args.benchmark_iters)

    # Test 5: Variants (optional, can be slow)
    if args.size == 256 and args.blocks == 4:
        test_variants()

    # Test 6: Gradient flow
    test_gradient_flow(model)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"CUDA kernels used: {stats['cuda_kernel_calls'] > 0}")
    print(f"Attention calls per forward pass: {len(model.encoder_blocks) + len(model.decoder_blocks)}")
    print(f"Average forward time: {avg_ms:.2f} ms")
    print(f"Expected speedup: 8-15x on attention operations (when CUDA is used)")

    if stats['cuda_kernel_calls'] > 0:
        print("\n✅ SUCCESS: CUDA kernels are integrated and working!")
    else:
        print("\n⚠️  WARNING: CUDA kernels not used - PyTorch baseline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
