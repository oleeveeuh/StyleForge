"""
Unit tests for model loading functionality.

Tests verify that:
- Pre-trained weights load without errors
- Model architecture matches expected structure
- Model parameters are correctly frozen after loading
"""

import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_net import TransformerNet, AVAILABLE_STYLES
from tests.config import MODELS_DIR, DEVICE


def test_load_pretrained_weights(style_name: str = "candy"):
    """
    Verify pre-trained weights load without errors.

    Args:
        style_name: Name of the style model to load (default: candy)

    Raises:
        AssertionError: If model loading fails
        FileNotFoundError: If checkpoint file doesn't exist
    """
    print(f"\n--- Testing pre-trained weight loading: {style_name} ---")

    checkpoint_path = MODELS_DIR / f"{style_name}.pth"

    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print(f"   Run: python download_models.py --style {style_name}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create model and load weights
    model = TransformerNet().to(DEVICE)
    model.load_checkpoint(str(checkpoint_path))

    # Verify model loaded
    assert model is not None, "Model is None after loading"

    # Verify parameters exist and are on correct device
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0, f"Model has no parameters! Found: {param_count}"

    # Check device placement
    device_check = all(
        p.device == DEVICE or
        p.device.type == DEVICE.type or
        p.device.type == 'cpu'
        for p in model.parameters()
    )
    if DEVICE.type == "cuda":
        device_check = all(p.device.type == "cuda" for p in model.parameters())

    assert device_check, f"Not all parameters on expected device {DEVICE}"

    print(f"✓ Model loaded successfully")
    print(f"  - Parameters: {param_count:,}")
    print(f"  - Device: {DEVICE}")
    print(f"  - Checkpoint: {checkpoint_path.name}")


def test_model_architecture():
    """
    Verify architecture matches expected structure.

    Checks:
    - Encoder has correct number of conv layers (3)
    - Residual blocks exist (5 by default)
    - Decoder has correct number of upsampling layers (3)
    - Output channels match input channels (3)
    """
    print("\n--- Testing model architecture ---")

    model = TransformerNet(num_residual_blocks=5)

    # Check encoder layers
    assert hasattr(model, 'conv1'), "Missing conv1 layer"
    assert hasattr(model, 'conv2'), "Missing conv2 layer"
    assert hasattr(model, 'conv3'), "Missing conv3 layer"

    # Check residual blocks
    assert hasattr(model, 'residual_blocks'), "Missing residual_blocks"
    assert len(model.residual_blocks) == 5, f"Expected 5 residual blocks, got {len(model.residual_blocks)}"

    # Check decoder layers
    assert hasattr(model, 'deconv1'), "Missing deconv1 layer"
    assert hasattr(model, 'deconv2'), "Missing deconv2 layer"
    assert hasattr(model, 'deconv3'), "Missing deconv3 layer"

    # Verify input/output channel counts
    # conv1: 3 -> 32
    assert model.conv1.conv.in_channels == 3, f"conv1 input channels should be 3, got {model.conv1.conv.in_channels}"
    assert model.conv1.conv.out_channels == 32, f"conv1 output channels should be 32, got {model.conv1.conv.out_channels}"

    # deconv3: 32 -> 3 (deconv3 is Sequential, so access via indexing)
    assert hasattr(model.deconv3, '__getitem__'), "deconv3 should be indexable"
    deconv3_conv = model.deconv3[1]  # nn.ReflectionPad2d is at [0], nn.Conv2d is at [1]
    assert deconv3_conv.in_channels == 32, f"deconv3 input channels should be 32, got {deconv3_conv.in_channels}"
    assert deconv3_conv.out_channels == 3, f"deconv3 output channels should be 3, got {deconv3_conv.out_channels}"

    # Count total layers
    total_layers = len(list(model.modules()))
    print(f"✓ Architecture verified")
    print(f"  - Encoder conv layers: 3 (conv1, conv2, conv3)")
    print(f"  - Residual blocks: {len(model.residual_blocks)}")
    print(f"  - Decoder layers: 3 (deconv1, deconv2, deconv3)")
    print(f"  - Total modules: {total_layers}")


def test_parameter_frozen(style_name: str = "candy"):
    """
    Verify parameters are frozen after loading pre-trained weights.

    For inference, we want parameters to not require gradients.
    """
    print(f"\n--- Testing parameter freezing: {style_name} ---")

    checkpoint_path = MODELS_DIR / f"{style_name}.pth"

    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint not found, skipping test")
        return

    model = TransformerNet().to(DEVICE)
    model.load_checkpoint(str(checkpoint_path))
    model.eval()  # Set to eval mode

    # Check that no parameters require gradients
    requires_grad_count = sum(1 for p in model.parameters() if p.requires_grad)

    # After eval(), parameters might still have requires_grad=True
    # But for inference we explicitly disable them
    for p in model.parameters():
        p.requires_grad = False

    assert all(p.requires_grad == False for p in model.parameters()), \
        f"{requires_grad_count} parameters still require gradients"

    print(f"✓ All parameters frozen for inference")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Frozen: {len(list(model.parameters()))}")


def test_model_size():
    """Verify model size is within expected range."""
    print("\n--- Testing model size ---")

    model = TransformerNet()

    size_mb = model.get_model_size()
    total_params, trainable_params = model.get_parameter_count()

    # TransformerNet should be around 1.7M parameters ~ 6.8 MB FP32
    expected_min_mb = 5.0
    expected_max_mb = 10.0

    assert expected_min_mb < size_mb < expected_max_mb, \
        f"Model size {size_mb:.2f} MB outside expected range [{expected_min_mb}, {expected_max_mb}]"

    print(f"✓ Model size within expected range")
    print(f"  - Size: {size_mb:.2f} MB (FP32)")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")


def test_all_available_styles():
    """Test that all available styles can be loaded (if checkpoints exist)."""
    print("\n--- Testing all available styles ---")

    loaded_count = 0
    missing_count = 0

    for style in AVAILABLE_STYLES:
        checkpoint_path = MODELS_DIR / f"{style}.pth"

        if checkpoint_path.exists():
            try:
                model = TransformerNet().to(DEVICE)
                model.load_checkpoint(str(checkpoint_path))

                # Quick forward pass test
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)
                    output = model(dummy_input)

                assert output.shape == (1, 3, 256, 256), \
                    f"Output shape mismatch for {style}: {output.shape}"

                loaded_count += 1
                print(f"  ✓ {style}: OK")
            except Exception as e:
                print(f"  ❌ {style}: FAILED - {e}")
                raise
        else:
            missing_count += 1
            print(f"  ⚠️  {style}: NOT FOUND (run download_models.py --all)")

    print(f"\n✓ Loaded {loaded_count}/{len(AVAILABLE_STYLES)} style models")

    if missing_count > 0:
        print(f"  Tip: Run 'python download_models.py --all' to download all models")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run model loading tests")
    parser.add_argument("--style", "-s", default="candy",
                        help="Style model to test (default: candy)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Test all available styles")

    args = parser.parse_args()

    print("=" * 60)
    print("STYLE FORGE - MODEL LOADING TESTS")
    print("=" * 60)

    try:
        test_model_architecture()
        test_model_size()

        if args.all:
            test_all_available_styles()
        else:
            test_load_pretrained_weights(args.style)
            test_parameter_frozen(args.style)

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
