#!/usr/bin/env python3
"""
Baseline PyTorch Fast Style Transfer

This script provides a baseline implementation of Fast Neural Style Transfer
using pure PyTorch. It can load pre-trained weights and process images.

Usage:
    python baseline_pytorch.py --input photo.jpg --style candy --output result.jpg

    # List available styles
    python baseline_pytorch.py --list-styles

    # Download pre-trained weights
    python baseline_pytorch.py --download-style candy
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.transformer_net import TransformerNet, AVAILABLE_STYLES
from utils.image_utils import load_image, preprocess_image, save_image, tensor_to_numpy
from utils.benchmark import benchmark_model, print_benchmark_results


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def list_styles():
    """Print all available pre-trained styles."""
    print("Available Pre-Trained Styles:")
    print("-" * 40)
    for style in AVAILABLE_STYLES:
        print(f"  - {style}")
    print("-" * 40)
    print("Use --download-style <name> to download weights")


def download_style(style_name: str, output_dir: str = "models/pretrained"):
    """Download pre-trained weights for a style."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if style_name not in AVAILABLE_STYLES:
        print(f"Error: Unknown style '{style_name}'")
        print(f"Available styles: {', '.join(AVAILABLE_STYLES)}")
        return False

    output_path = output_dir / f"{style_name}.pth"

    if output_path.exists():
        print(f"✅ Weights already exist: {output_path}")
        return True

    from models.transformer_net import get_style_url
    url = get_style_url(style_name)

    print(f"Downloading {style_name} weights from {url}...")

    try:
        import urllib.request
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def process_image(
    input_path: str,
    style_name: str,
    output_path: str,
    checkpoint_dir: str = "models/pretrained",
    image_size: int = 512,
    benchmark: bool = False,
):
    """Process an image with style transfer."""
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = Path(checkpoint_dir) / f"{style_name}.pth"

    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print(f"Downloading {style_name} weights...")
        if not download_style(style_name, checkpoint_dir):
            return False

    print(f"Loading model: {style_name}")
    model = TransformerNet().to(device)
    model.load_checkpoint(str(checkpoint_path))
    model.eval()

    print(f"Model size: {model.get_model_size():.2f} MB")

    # Load and preprocess image
    print(f"Loading image: {input_path}")
    img = load_image(input_path, size=image_size, square=False)
    original_size = img.size
    print(f"  Original size: {original_size}")

    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    print(f"  Processing size: {input_tensor.shape[2:]}")

    # Process
    if benchmark:
        # Run benchmark
        result = benchmark_model(
            model,
            input_tensor,
            warmup_iters=10,
            test_iters=50,
            name=f"baseline_{style_name}"
        )
        print_benchmark_results(result)
    else:
        # Single inference
        print("Applying style transfer...")
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Convert back to image
        output_img = transforms.ToPILImage()(output_tensor.squeeze(0).clamp(0, 1))

        # Resize back to original dimensions
        output_img = output_img.resize(original_size, Image.Resampling.LANCZOS)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_img.save(output_path, quality=95)
        print(f"✅ Saved to: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Baseline PyTorch Fast Style Transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process an image with candy style
  python baseline_pytorch.py --input photo.jpg --style candy --output result.jpg

  # Use different image size
  python baseline_pytorch.py --input photo.jpg --style starry --output result.jpg --size 1024

  # Run benchmark
  python baseline_pytorch.py --input photo.jpg --style candy --benchmark

  # List available styles
  python baseline_pytorch.py --list-styles

  # Download weights for a style
  python baseline_pytorch.py --download-style candy
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input image path'
    )
    parser.add_argument(
        '--style', '-s',
        type=str,
        choices=AVAILABLE_STYLES,
        help='Style name (use --list-styles to see all)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output.jpg',
        help='Output image path (default: output.jpg)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models/pretrained',
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=512,
        choices=[256, 512, 1024],
        help='Processing size (default: 512)'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark instead of single inference'
    )
    parser.add_argument(
        '--list-styles',
        action='store_true',
        help='List available pre-trained styles'
    )
    parser.add_argument(
        '--download-style',
        type=str,
        metavar='NAME',
        help='Download pre-trained weights for a style'
    )

    args = parser.parse_args()

    # Handle special commands
    if args.list_styles:
        list_styles()
        return 0

    if args.download_style:
        return 0 if download_style(args.download_style, args.checkpoint_dir) else 1

    # Validate required args
    if not args.input:
        parser.error("--input is required (unless using --list-styles or --download-style)")
    if not args.style:
        parser.error("--style is required")

    # Process image
    return 0 if process_image(
        args.input,
        args.style,
        args.output,
        args.checkpoint_dir,
        args.size,
        args.benchmark,
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
