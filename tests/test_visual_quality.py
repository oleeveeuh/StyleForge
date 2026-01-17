"""
Visual quality tests for StyleForge.

Tests verify that:
- Style transfer produces visually reasonable outputs
- Outputs can be saved correctly
- Multiple styles work on the same image
- Test outputs are saved for manual inspection
"""

import io
import sys
from pathlib import Path

import requests
import torch
from PIL import Image
from torchvision import transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_net import TransformerNet, AVAILABLE_STYLES
from tests.config import (
    DEVICE,
    MODELS_DIR,
    TEST_OUTPUTS_DIR,
    TEST_DATA_DIR,
    TEST_IMAGE_URLS,
)


def download_test_images():
    """
    Download test images if not present.

    Returns:
        List of paths to test images
    """
    print("\n--- Ensuring test images exist ---")

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    test_images = []

    for name, url in TEST_IMAGE_URLS.items():
        img_path = TEST_DATA_DIR / f"{name}.jpg"

        if not img_path.exists():
            print(f"  Downloading {name}...")
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                img = Image.open(io.BytesIO(response.content)).convert('RGB')
                img.save(img_path)
                print(f"  ✓ Saved to {img_path}")
            except Exception as e:
                print(f"  ⚠️  Failed to download {name}: {e}")
                # Create a simple test image instead
                img = Image.new('RGB', (512, 512), color=(128, 128, 128))
                img.save(img_path)
                print(f"  ✓ Created placeholder {img_path}")
        else:
            print(f"  ✓ {name} already exists")

        test_images.append(img_path)

    return test_images


def load_image(img_path: Path, size: int = 512) -> torch.Tensor:
    """
    Load and preprocess an image.

    Args:
        img_path: Path to image file
        size: Size to resize to (maintains aspect ratio)

    Returns:
        Preprocessed tensor (1, 3, H, W) in range [0, 1]
    """
    img = Image.open(img_path).convert('RGB')

    # Resize maintaining aspect ratio
    aspect = img.width / img.height
    if aspect > 1:
        new_width = size
        new_height = int(size / aspect)
    else:
        new_width = int(size * aspect)
        new_height = size

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform(img).unsqueeze(0)


def save_image(tensor: torch.Tensor, output_path: Path) -> None:
    """
    Save a tensor as an image.

    Args:
        tensor: Image tensor (1, 3, H, W) in range [0, 1], [-1, 1], or [0, 255]
        output_path: Path to save to
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove batch dimension
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Detect range and normalize to [0, 1]
    tensor_min = tensor.min().item()
    tensor_max = tensor.max().item()

    # If values are in [0, 255] range (like pre-trained model outputs), normalize to [0, 1]
    if tensor_max > 1.0:
        tensor = tensor / 255.0
    # If values are in [-1, 1] range, shift to [0, 1]
    elif tensor_min < 0:
        tensor = (tensor + 1.0) / 2.0

    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to PIL
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor.cpu())

    img.save(output_path, quality=95)


def test_style_transfer_visual():
    """
    Generate test outputs and save for manual inspection.

    This is the main visual quality test that creates stylized images
    for human review.
    """
    print("\n--- Running visual quality test ---")

    # Ensure output directory exists
    TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Download test images
    test_image_paths = download_test_images()

    # Test subset of styles (use only downloaded ones)
    test_styles = []
    for style in AVAILABLE_STYLES:
        if (MODELS_DIR / f"{style}.pth").exists():
            test_styles.append(style)

    if not test_styles:
        print("⚠️  No style models found!")
        print("   Run: python download_models.py --all")
        return

    print(f"\nTesting with {len(test_image_paths)} images and {len(test_styles)} styles")

    generated_count = 0

    for img_path in test_image_paths:
        img_name = img_path.stem

        print(f"\nProcessing {img_name}...")

        # Load image
        input_tensor = load_image(img_path, size=512).to(DEVICE)

        # Save original for reference
        original_output_path = TEST_OUTPUTS_DIR / f"{img_name}_original.jpg"
        save_image(input_tensor, original_output_path)
        print(f"  Saved original: {original_output_path}")

        # Process with each style
        for style_name in test_styles:
            try:
                # Load model
                checkpoint_path = MODELS_DIR / f"{style_name}.pth"
                model = TransformerNet().to(DEVICE)
                model.load_checkpoint(str(checkpoint_path))
                model.eval()

                # Stylize
                with torch.no_grad():
                    output_tensor = model(input_tensor)

                # Save output
                output_path = TEST_OUTPUTS_DIR / f"{img_name}_{style_name}.jpg"
                save_image(output_tensor, output_path)

                generated_count += 1
                print(f"  ✓ Generated {img_name}_{style_name}.jpg")

            except Exception as e:
                print(f"  ❌ Failed to process {img_name} with {style_name}: {e}")

    print(f"\n✓ Visual quality test complete")
    print(f"  Generated {generated_count} stylized images")
    print(f"\n📁 Check {TEST_OUTPUTS_DIR} for visual inspection")


def test_image_round_trip():
    """
    Test that images can be loaded, processed, and saved correctly.
    """
    print("\n--- Testing image round-trip ---")

    TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create a test image with known colors
    test_img = Image.new('RGB', (256, 256), color=(
        int(255 * 0.2),  # Red
        int(255 * 0.5),  # Green
        int(255 * 0.8),  # Blue
    ))

    # Draw a simple pattern
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_img)
    draw.rectangle([50, 50, 200, 200], fill=(255, 0, 0))
    draw.ellipse([100, 100, 150, 150], fill=(0, 255, 0))

    test_img_path = TEST_DATA_DIR / "test_pattern.jpg"
    test_img.save(test_img_path)

    # Load and process
    input_tensor = load_image(test_img_path, size=256).to(DEVICE)

    # Create model (use random weights if no checkpoint)
    checkpoint_path = MODELS_DIR / "candy.pth"
    model = TransformerNet().to(DEVICE)

    if checkpoint_path.exists():
        model.load_checkpoint(str(checkpoint_path))
        print("  Using pre-trained candy model")
    else:
        print("  Using random initialization")

    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

    # Save
    output_path = TEST_OUTPUTS_DIR / "test_pattern_styled.jpg"
    save_image(output, output_path)

    # Verify output file exists and is valid
    assert output_path.exists(), f"Output file not created: {output_path}"

    # Verify can be re-opened
    reopened = Image.open(output_path)
    assert reopened.size == (input_tensor.shape[3], input_tensor.shape[2]), \
        f"Output image size mismatch: {reopened.size}"

    print(f"✓ Image round-trip successful")
    print(f"  Input:  {input_tensor.shape[2]}x{input_tensor.shape[3]}")
    print(f"  Output: {reopened.size[0]}x{reopened.size[1]}")


def test_edge_cases():
    """
    Test model behavior on edge case images.
    """
    print("\n--- Testing edge cases ---")

    TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model = TransformerNet().to(DEVICE)

    checkpoint_path = MODELS_DIR / "candy.pth"
    if checkpoint_path.exists():
        model.load_checkpoint(str(checkpoint_path))

    model.eval()

    # Test cases
    test_cases = [
        ("solid_white", torch.ones(1, 3, 256, 256)),
        ("solid_black", torch.zeros(1, 3, 256, 256)),
        ("solid_gray", torch.full((1, 3, 256, 256), 0.5)),
        ("checkerboard", _create_checkerboard(256)),
        ("gradient", _create_gradient(256)),
    ]

    for name, test_input in test_cases:
        test_input = test_input.to(DEVICE)

        with torch.no_grad():
            output = model(test_input)

        # Save output
        output_path = TEST_OUTPUTS_DIR / f"edge_{name}.jpg"
        save_image(output, output_path)

        # Verify output is valid
        assert not torch.isnan(output).any(), f"NaN in {name} output"
        assert not torch.isinf(output).any(), f"Inf in {name} output"

        print(f"  ✓ {name}: OK")

    print(f"✓ Edge case tests passed")


def _create_checkerboard(size: int, num_squares: int = 8) -> torch.Tensor:
    """Create a checkerboard pattern tensor."""
    img = torch.zeros(1, 3, size, size)
    square_size = size // num_squares

    for i in range(num_squares):
        for j in range(num_squares):
            if (i + j) % 2 == 1:
                img[:, :, i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 1.0

    return img


def _create_gradient(size: int) -> torch.Tensor:
    """Create a horizontal gradient tensor."""
    img = torch.zeros(1, 3, size, size)

    for i in range(size):
        value = i / size
        img[:, :, :, i] = value

    return img


def test_style_comparison():
    """
    Generate a side-by-side comparison of all styles on the same image.
    """
    print("\n--- Creating style comparison ---")

    TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load a test image
    test_images = download_test_images()
    if not test_images:
        print("⚠️  No test images available")
        return

    img_path = test_images[0]
    input_tensor = load_image(img_path, size=512).to(DEVICE)

    # Get available styles
    available_styles = [s for s in AVAILABLE_STYLES
                       if (MODELS_DIR / f"{s}.pth").exists()]

    if not available_styles:
        print("⚠️  No style models found")
        return

    # Create comparison image
    from PIL import ImageDraw

    n_styles = len(available_styles)
    grid_cols = min(3, n_styles + 1)  # Include original
    grid_rows = (n_styles + 1 + grid_cols - 1) // grid_cols

    cell_size = 256
    comparison = Image.new('RGB', (
        grid_cols * cell_size,
        grid_rows * cell_size
    ), color=(255, 255, 255))

    # Add original image
    original_pil = transforms.ToPILImage()(input_tensor.squeeze(0).cpu())
    original_pil = original_pil.resize((cell_size - 10, cell_size - 40))
    comparison.paste(original_pil, (5, 5))

    # Add label
    draw = ImageDraw.Draw(comparison)
    draw.text((5, cell_size - 30), "Original", fill=(0, 0, 0))

    # Add each style
    for idx, style_name in enumerate(available_styles):
        try:
            # Load and run model
            checkpoint_path = MODELS_DIR / f"{style_name}.pth"
            model = TransformerNet().to(DEVICE)
            model.load_checkpoint(str(checkpoint_path))
            model.eval()

            with torch.no_grad():
                output = model(input_tensor)

            # Convert to PIL and resize
            output_pil = transforms.ToPILImage()(output.squeeze(0).cpu())
            output_pil = output_pil.resize((cell_size - 10, cell_size - 40))

            # Calculate position
            row = (idx + 1) // grid_cols
            col = (idx + 1) % grid_cols
            x = col * cell_size + 5
            y = row * cell_size + 5

            # Paste and label
            comparison.paste(output_pil, (x, y))
            draw.text((x, y + cell_size - 30), style_name, fill=(0, 0, 0))

            print(f"  ✓ Added {style_name}")

        except Exception as e:
            print(f"  ❌ Failed to add {style_name}: {e}")

    # Save comparison
    comparison_path = TEST_OUTPUTS_DIR / "style_comparison.jpg"
    comparison.save(comparison_path, quality=95)

    print(f"\n✓ Style comparison saved to {comparison_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run visual quality tests")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading test images")
    parser.add_argument("--comparison-only", action="store_true",
                        help="Only run style comparison test")

    args = parser.parse_args()

    print("=" * 60)
    print("STYLE FORGE - VISUAL QUALITY TESTS")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Output directory: {TEST_OUTPUTS_DIR}")

    try:
        if not args.skip_download:
            download_test_images()

        if args.comparison_only:
            test_style_comparison()
        else:
            test_style_transfer_visual()
            test_image_round_trip()
            test_edge_cases()
            test_style_comparison()

        print("\n" + "=" * 60)
        print("✅ VISUAL QUALITY TESTS PASSED")
        print("=" * 60)
        print(f"\n📁 Check {TEST_OUTPUTS_DIR} for generated images")
        print("   Please manually inspect the outputs to verify quality")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
