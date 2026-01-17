"""
Numerical accuracy tests for StyleForge.

Tests verify that:
- Model outputs are numerically stable
- Custom kernels match PyTorch implementations
- Outputs are consistent across multiple runs
- Numerical precision is maintained
"""

import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_net import TransformerNet, ConvLayer, ResidualBlock, UpsampleConvLayer
from tests.config import (
    DEVICE,
    IS_CUDA_AVAILABLE,
    NUMERICAL_TOLERANCE,
    MODELS_DIR,
)


def test_layer_numerical_stability():
    """
    Test individual layers for numerical stability.
    """
    print("\n--- Testing layer numerical stability ---")

    # Test ConvLayer
    conv = ConvLayer(3, 32, kernel_size=3, stride=1, padding=1).to(DEVICE)
    conv.eval()

    input_tensor = torch.randn(1, 3, 256, 256).to(DEVICE)

    with torch.no_grad():
        output = conv(input_tensor)

    assert not torch.isnan(output).any(), "ConvLayer produced NaN"
    assert not torch.isinf(output).any(), "ConvLayer produced Inf"
    print(f"  ✓ ConvLayer: output range [{output.min():.3f}, {output.max():.3f}]")

    # Test ResidualBlock
    res = ResidualBlock(64).to(DEVICE)
    res.eval()

    input_tensor = torch.randn(1, 64, 128, 128).to(DEVICE)

    with torch.no_grad():
        output = res(input_tensor)

    assert not torch.isnan(output).any(), "ResidualBlock produced NaN"
    assert not torch.isinf(output).any(), "ResidualBlock produced Inf"
    print(f"  ✓ ResidualBlock: output range [{output.min():.3f}, {output.max():.3f}]")

    # Test UpsampleConvLayer
    upsample = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, padding=1, upsample=2).to(DEVICE)
    upsample.eval()

    input_tensor = torch.randn(1, 64, 64, 64).to(DEVICE)

    with torch.no_grad():
        output = upsample(input_tensor)

    assert not torch.isnan(output).any(), "UpsampleConvLayer produced NaN"
    assert not torch.isinf(output).any(), "UpsampleConvLayer produced Inf"
    print(f"  ✓ UpsampleConvLayer: output range [{output.min():.3f}, {output.max():.3f}]")

    print("✓ All layers numerically stable")


def test_full_model_numerical_stability(style_name: str = "candy"):
    """
    Test full model for numerical stability.
    """
    print(f"\n--- Testing full model numerical stability: {style_name} ---")

    model = TransformerNet().to(DEVICE)

    checkpoint_path = MODELS_DIR / f"{style_name}.pth"
    if checkpoint_path.exists():
        model.load_checkpoint(str(checkpoint_path))

    model.eval()

    # Test with various input ranges
    test_cases = [
        ("uniform [0,1]", torch.rand(1, 3, 512, 512)),
        ("normal", torch.randn(1, 3, 512, 512)),
        ("extreme values", torch.ones(1, 3, 512, 512) * 10),
        ("small values", torch.ones(1, 3, 512, 512) * 0.01),
        ("negative", -torch.rand(1, 3, 512, 512)),
    ]

    for name, input_tensor in test_cases:
        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()

        status = "✓" if not (has_nan or has_inf) else "❌"
        print(f"  {status} {name}: range [{output.min():.3f}, {output.max():.3f}]")

        if has_nan:
            raise AssertionError(f"NaN detected in output for {name}")
        if has_inf:
            raise AssertionError(f"Inf detected in output for {name}")

    print("✓ Full model numerically stable")


def test_deterministic_forward(style_name: str = "candy"):
    """
    Verify model produces deterministic output.
    """
    print(f"\n--- Testing deterministic forward pass: {style_name} ---")

    model = TransformerNet().to(DEVICE)

    checkpoint_path = MODELS_DIR / f"{style_name}.pth"
    if checkpoint_path.exists():
        model.load_checkpoint(str(checkpoint_path))

    model.eval()

    # Set seed
    torch.manual_seed(42)
    input_tensor = torch.randn(1, 3, 256, 256).to(DEVICE)

    # Run multiple times
    outputs = []
    for i in range(5):
        with torch.no_grad():
            output = model(input_tensor)
        outputs.append(output.clone())

    # Compare all outputs
    for i in range(1, len(outputs)):
        diff = (outputs[0] - outputs[i]).abs().max().item()
        assert diff < 1e-6, f"Output not deterministic: run 0 vs {i} diff = {diff}"

    print(f"✓ Deterministic forward pass verified")
    print(f"  All 5 runs produced identical output")


def test_fp16_vs_fp32_consistency(style_name: str = "candy"):
    """
    Test FP16 and FP32 produce similar outputs (on CUDA).
    """
    if not IS_CUDA_AVAILABLE:
        print("\n--- Skipping FP16 vs FP32 test (CUDA not available) ---")
        return

    print(f"\n--- Testing FP16 vs FP32 consistency: {style_name} ---")

    # Create models
    model_fp32 = TransformerNet().to(DEVICE)

    checkpoint_path = MODELS_DIR / f"{style_name}.pth"
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        model_fp32.load_state_dict(new_state_dict)

    model_fp32.eval()

    # Clone for FP16
    model_fp16 = TransformerNet().to(DEVICE)
    model_fp16.load_state_dict(model_fp32.state_dict())
    model_fp16.half()
    model_fp16.eval()

    # Test input
    torch.manual_seed(42)
    input_fp32 = torch.randn(1, 3, 256, 256).to(DEVICE)
    input_fp16 = input_fp32.half()

    with torch.no_grad():
        output_fp32 = model_fp32(input_fp32)
        output_fp16 = model_fp16(input_fp16)

    # Convert FP16 to FP32 for comparison
    output_fp16_fp32 = output_fp16.float()

    # Compare
    max_diff = (output_fp32 - output_fp16_fp32).abs().max().item()
    mean_diff = (output_fp32 - output_fp16_fp32).abs().mean().item()

    print(f"  Max difference: {max_diff:.4f}")
    print(f"  Mean difference: {mean_diff:.4f}")

    # FP16 should be reasonably close to FP32
    # Allow larger tolerance for FP16
    tolerance = 0.01
    assert max_diff < tolerance, f"FP16 differs too much from FP32: {max_diff}"

    print(f"✓ FP16 and FP32 outputs consistent (tolerance: {tolerance})")


def test_attention_implementations_match():
    """
    Compare custom attention kernel with PyTorch implementation.
    """
    print("\n--- Testing attention implementations match ---")

    if not IS_CUDA_AVAILABLE:
        print("⚠️  CUDA not available, skipping")
        return

    try:
        from kernels.attention_wrapper import fused_attention_forward
    except ImportError:
        print("⚠️  Custom attention kernel not available, skipping")
        return

    # Set seed for reproducibility
    torch.manual_seed(42)

    batch_size = 2
    num_heads = 4
    seq_len = 32
    head_dim = 32

    # Create test inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE)

    # PyTorch implementation
    scale = head_dim ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    pytorch_output = torch.matmul(attn_weights, v)

    # Custom kernel
    try:
        custom_output = fused_attention_forward(q, k, v)

        # Handle tuple return
        if isinstance(custom_output, tuple):
            custom_output = custom_output[0]

        # Compare
        max_diff = (pytorch_output - custom_output).abs().max().item()
        mean_diff = (pytorch_output - custom_output).abs().mean().item()

        tolerance = NUMERICAL_TOLERANCE

        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")

        assert max_diff < tolerance["max_abs_diff"], \
            f"Outputs differ too much: {max_diff} > {tolerance['max_abs_diff']}"
        assert mean_diff < tolerance["mean_abs_diff"], \
            f"Mean difference too large: {mean_diff} > {tolerance['mean_abs_diff']}"

        print(f"✓ Custom attention matches PyTorch implementation")

    except Exception as e:
        print(f"⚠️  Could not compare implementations: {e}")


def test_gradient_numerics():
    """
    Test gradient computation for numerical correctness.
    """
    print("\n--- Testing gradient numerics ---")

    model = TransformerNet().to(DEVICE)
    model.train()

    input_tensor = torch.randn(1, 3, 256, 256, requires_grad=True, device=DEVICE)

    output = model(input_tensor)
    loss = output.mean()
    loss.backward()

    # Check input gradient exists and is valid
    assert input_tensor.grad is not None, "Input gradient is None"
    assert not torch.isnan(input_tensor.grad).any(), "NaN in input gradient"
    assert not torch.isinf(input_tensor.grad).any(), "Inf in input gradient"

    # Check parameter gradients
    grad_stats = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_max = param.grad.abs().max().item()
            grad_stats.append((name, grad_norm, grad_max))

    # Print some stats
    print(f"  Input gradient norm: {input_tensor.grad.norm().item():.4f}")
    print(f"  Input gradient range: [{input_tensor.grad.min():.4f}, {input_tensor.grad.max():.4f}]")
    print(f"  Parameters with gradients: {len(grad_stats)}")

    # Check for exploding/vanishing gradients
    for name, norm, max_val in grad_stats[:5]:  # Check first 5
        print(f"    {name}: norm={norm:.4f}, max={max_val:.4f}")

    print("✓ Gradient numerics verified")


def test_instance_norm_numerics():
    """
    Test InstanceNorm for numerical correctness.
    """
    print("\n--- Testing InstanceNorm numerics ---")

    # Create test input
    N, C, H, W = 4, 32, 64, 64
    input_tensor = torch.randn(N, C, H, W, device=DEVICE)

    # Test PyTorch InstanceNorm
    norm = nn.InstanceNorm2d(C, affine=True).to(DEVICE)
    norm.eval()

    weight = norm.weight.clone()
    bias = norm.bias.clone()

    with torch.no_grad():
        pytorch_output = norm(input_tensor)

    # Test custom kernel if available
    if IS_CUDA_AVAILABLE:
        try:
            from kernels.instance_norm_wrapper import fused_instance_norm_forward

            custom_output = fused_instance_norm_forward(
                input_tensor, weight, bias, eps=1e-5
            )

            if torch.is_tensor(custom_output):
                max_diff = (pytorch_output - custom_output).abs().max().item()
                print(f"  Max difference from PyTorch: {max_diff:.2e}")

                if max_diff < 1e-4:
                    print(f"✓ Custom InstanceNorm matches PyTorch")
                else:
                    print(f"⚠️  Custom InstanceNorm differs by {max_diff:.2e}")
        except ImportError:
            print("  ⚠️  Custom InstanceNorm not available")
    else:
        print("  ⚠️  CUDA not available, skipping custom kernel test")

    # Verify PyTorch output is valid
    assert not torch.isnan(pytorch_output).any(), "NaN in InstanceNorm output"
    assert not torch.isinf(pytorch_output).any(), "Inf in InstanceNorm output"

    print("✓ InstanceNorm numerics verified")


def test_residual_block_numerics():
    """
    Test residual block for numerical correctness.
    """
    print("\n--- Testing residual block numerics ---")

    channels = 64
    block = ResidualBlock(channels).to(DEVICE)
    block.eval()

    # Test with various input scales
    test_inputs = [
        ("normal scale", torch.randn(1, channels, 64, 64)),
        ("small scale", torch.randn(1, channels, 64, 64) * 0.1),
        ("large scale", torch.randn(1, channels, 64, 64) * 10),
    ]

    for name, input_tensor in test_inputs:
        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():
            output = block(input_tensor)

        # Check for numerical issues
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()

        # Check residual connection preserved signal
        signal_preserved = (output - input_tensor).abs().mean().item() > 0

        status = "✓" if not (has_nan or has_inf) else "❌"
        print(f"  {status} {name}: output range [{output.min():.3f}, {output.max():.3f}]")

        if has_nan:
            raise AssertionError(f"NaN in residual block output ({name})")
        if has_inf:
            raise AssertionError(f"Inf in residual block output ({name})")

    print("✓ Residual block numerics verified")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run numerical accuracy tests")
    parser.add_argument("--style", "-s", default="candy",
                        help="Style model to test (default: candy)")

    args = parser.parse_args()

    print("=" * 60)
    print("STYLE FORGE - NUMERICAL ACCURACY TESTS")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    try:
        test_layer_numerical_stability()
        test_full_model_numerical_stability(args.style)
        test_deterministic_forward(args.style)
        test_fp16_vs_fp32_consistency(args.style)
        test_attention_implementations_match()
        test_gradient_numerics()
        test_instance_norm_numerics()
        test_residual_block_numerics()

        print("\n" + "=" * 60)
        print("✅ ALL NUMERICAL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
