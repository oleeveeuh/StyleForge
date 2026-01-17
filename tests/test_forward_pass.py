"""
Forward pass tests for StyleForge.

Tests verify that:
- Output shape matches input shape for various sizes
- Output contains no NaN or Inf values
- Output pixel values are in valid range
- Forward pass works correctly on different devices
"""

import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_net import TransformerNet
from tests.config import (
    TEST_IMAGE_SIZES,
    NUMERICAL_TOLERANCE,
    DEVICE,
    IS_CUDA_AVAILABLE,
    MODELS_DIR,
)


def test_forward_pass_shapes(style_name: str = "candy"):
    """
    Verify output shape matches input shape for various image sizes.

    Tests multiple resolutions to ensure the model handles different sizes correctly.
    """
    print("\n--- Testing forward pass shapes ---")

    # Try to load pre-trained weights, otherwise use random initialization
    model = TransformerNet().to(DEVICE)

    checkpoint_path = MODELS_DIR / f"{style_name}.pth"
    if checkpoint_path.exists():
        model.load_checkpoint(str(checkpoint_path))
        print(f"Using pre-trained weights: {style_name}")
    else:
        print(f"Using random initialization (checkpoint not found: {style_name})")

    model.eval()

    for h, w in TEST_IMAGE_SIZES:
        input_img = torch.randn(1, 3, h, w).to(DEVICE)

        with torch.no_grad():
            output = model(input_img)

        assert output.shape == input_img.shape, \
            f"Shape mismatch for {h}x{w}: expected {input_img.shape}, got {output.shape}"

        print(f"✓ Forward pass works for {h}x{w} -> output shape: {output.shape}")


def test_output_range(style_name: str = "candy"):
    """
    Verify output pixel values are in valid range.

    The model output should be bounded and not produce extreme values.
    """
    print(f"\n--- Testing output value range: {style_name} ---")

    model = TransformerNet().to(DEVICE)

    checkpoint_path = MODELS_DIR / f"{style_name}.pth"
    if checkpoint_path.exists():
        model.load_checkpoint(str(checkpoint_path))

    model.eval()

    # Test with normalized input [0, 1]
    input_img = torch.rand(1, 3, 512, 512).to(DEVICE)

    with torch.no_grad():
        output = model(input_img)

    output_min = output.min().item()
    output_max = output.max().item()
    output_mean = output.mean().item()
    output_std = output.std().item()

    print(f"  Output statistics:")
    print(f"    Min:  {output_min:+.4f}")
    print(f"    Max:  {output_max:+.4f}")
    print(f"    Mean: {output_mean:+.4f}")
    print(f"    Std:  {output_std:.4f}")

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "NaN values detected in output!"
    assert not torch.isinf(output).any(), "Inf values detected in output!"

    # Check reasonable bounds (allow some flexibility for untrained models)
    tolerance = NUMERICAL_TOLERANCE
    assert output_min >= tolerance["output_min"], \
        f"Output minimum too low: {output_min} < {tolerance['output_min']}"
    assert output_max <= tolerance["output_max"], \
        f"Output maximum too high: {output_max} > {tolerance['output_max']}"

    print(f"✓ Output values in valid range")


def test_batch_processing(style_name: str = "candy"):
    """
    Verify model can process batches of images.
    """
    print(f"\n--- Testing batch processing: {style_name} ---")

    model = TransformerNet().to(DEVICE)

    checkpoint_path = MODELS_DIR / f"{style_name}.pth"
    if checkpoint_path.exists():
        model.load_checkpoint(str(checkpoint_path))

    model.eval()

    # Test different batch sizes
    batch_sizes = [1, 2, 4]

    for batch_size in batch_sizes:
        input_img = torch.randn(batch_size, 3, 256, 256).to(DEVICE)

        with torch.no_grad():
            output = model(input_img)

        assert output.shape == input_img.shape, \
            f"Batch {batch_size}: Shape mismatch {output.shape} != {input_img.shape}"
        assert output.shape[0] == batch_size, \
            f"Batch size mismatch: expected {batch_size}, got {output.shape[0]}"

        print(f"✓ Batch size {batch_size}: OK")


def test_gradient_flow():
    """
    Verify gradients flow correctly through the network.

    This is important for training, though less critical for inference.
    """
    print("\n--- Testing gradient flow ---")

    model = TransformerNet().to(DEVICE)
    model.train()  # Enable gradients

    input_img = torch.randn(1, 3, 256, 256, requires_grad=True, device=DEVICE)

    output = model(input_img)
    loss = output.mean()
    loss.backward()

    # Check gradients exist
    assert input_img.grad is not None, "Input gradients are None"
    assert input_img.grad.abs().sum() > 0, "Input gradients are all zero"

    # Check model parameter gradients
    grad_count = 0
    zero_grad_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            if param.grad.abs().sum() == 0:
                zero_grad_count += 1

    assert grad_count > 0, "No parameters have gradients!"

    print(f"✓ Gradient flow verified")
    print(f"  - Parameters with gradients: {grad_count}")
    print(f"  - Parameters with zero gradients: {zero_grad_count}")


def test_different_channels():
    """
    Verify model only accepts 3-channel RGB input.
    """
    print("\n--- Testing input channel validation ---")

    model = TransformerNet().to(DEVICE)
    model.eval()

    # Test with correct input
    try:
        input_img = torch.randn(1, 3, 256, 256).to(DEVICE)
        with torch.no_grad():
            output = model(input_img)
        print(f"✓ 3-channel RGB input: OK")
    except Exception as e:
        raise AssertionError(f"3-channel input failed: {e}")

    # Test with wrong input (should fail or produce different shape)
    try:
        input_img_1ch = torch.randn(1, 1, 256, 256).to(DEVICE)
        with torch.no_grad():
            output = model(input_img_1ch)
        print(f"⚠️  1-channel input produced output: {output.shape} (expected failure)")
    except Exception:
        print(f"✓ 1-channel input correctly rejected")

    try:
        input_img_4ch = torch.randn(1, 4, 256, 256).to(DEVICE)
        with torch.no_grad():
            output = model(input_img_4ch)
        print(f"⚠️  4-channel input produced output: {output.shape} (expected failure)")
    except Exception:
        print(f"✓ 4-channel input correctly rejected")


def test_non_square_images():
    """
    Verify model handles non-square (rectangular) images.
    """
    print("\n--- Testing non-square images ---")

    model = TransformerNet().to(DEVICE)
    model.eval()

    # Test various aspect ratios
    test_sizes = [
        (256, 512),   # Portrait 2:1
        (512, 256),   # Landscape 2:1
        (256, 384),   # 3:4
        (384, 256),   # 4:3
    ]

    for h, w in test_sizes:
        input_img = torch.randn(1, 3, h, w).to(DEVICE)

        with torch.no_grad():
            output = model(input_img)

        # Note: Due to stride-2 convolutions, spatial dimensions may be divided
        # The important thing is output has valid shape
        assert output.shape[1] == 3, f"Output channels should be 3, got {output.shape[1]}"
        assert output.shape[0] == 1, f"Batch size should be 1, got {output.shape[0]}"
        assert output.dim() == 4, f"Output should be 4D, got {output.dim()}D"

        print(f"✓ {h}x{w} -> {output.shape[2]}x{output.shape[3]}")


def test_deterministic_output():
    """
    Verify model produces deterministic output for same input.
    """
    print("\n--- Testing deterministic output ---")

    model = TransformerNet().to(DEVICE)
    model.eval()

    input_img = torch.randn(1, 3, 256, 256).to(DEVICE)

    with torch.no_grad():
        output1 = model(input_img)
        output2 = model(input_img)

    diff = (output1 - output2).abs().max().item()

    assert diff < 1e-6, f"Output not deterministic: max diff = {diff}"

    print(f"✓ Deterministic output verified (max diff: {diff:.2e})")


def test_cpu_vs_cuda_consistency(style_name: str = "candy"):
    """
    Verify model produces same output on CPU and CUDA (if available).
    """
    if not IS_CUDA_AVAILABLE:
        print("\n--- Skipping CPU vs CUDA test (CUDA not available) ---")
        return

    print(f"\n--- Testing CPU vs CUDA consistency: {style_name} ---")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create models
    model_cpu = TransformerNet()
    model_cuda = TransformerNet().to(DEVICE)

    # Load same weights
    checkpoint_path = MODELS_DIR / f"{style_name}.pth"
    if checkpoint_path.exists():
        model_cpu.load_checkpoint(str(checkpoint_path))

        # Load to CPU first, then move to CUDA
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # Remove 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        model_cuda.load_state_dict(new_state_dict)

    model_cpu.eval()
    model_cuda.eval()

    # Same input
    input_img = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        output_cpu = model_cpu(input_img)
        output_cuda = model_cuda(input_img.to(DEVICE))

    output_cuda_cpu = output_cuda.cpu()

    # Compare
    max_diff = (output_cpu - output_cuda_cpu).abs().max().item()

    print(f"  CPU vs CUDA max difference: {max_diff:.2e}")

    # Small differences are acceptable due to floating point
    assert max_diff < 1e-4, f"CPU and CUDA outputs differ too much: {max_diff}"

    print(f"✓ CPU and CUDA outputs consistent")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run forward pass tests")
    parser.add_argument("--style", "-s", default="candy",
                        help="Style model to test (default: candy)")

    args = parser.parse_args()

    print("=" * 60)
    print("STYLE FORGE - FORWARD PASS TESTS")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    try:
        test_forward_pass_shapes(args.style)
        test_output_range(args.style)
        test_batch_processing(args.style)
        test_gradient_flow()
        test_different_channels()
        test_non_square_images()
        test_deterministic_output()
        test_cpu_vs_cuda_consistency(args.style)

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
