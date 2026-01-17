#!/usr/bin/env python3
"""
Download pre-trained Fast Style Transfer model weights.

This script downloads pre-trained weights from the fast-neural-style repository.
Each style file is ~5-10 MB.

Usage:
    # Download all styles
    python download_models.py --all

    # Download specific style
    python download_models.py --style candy

    # List available styles
    python download_models.py --list
"""

import argparse
import hashlib
import sys
from pathlib import Path

import requests


# Pre-trained style models from community repositories
# Original jcjohnson repository uses .t7 (Torch) format, not .pth (PyTorch)
# These are working URLs from active PyTorch implementations
STYLES = {
    "candy": {
        "url": "https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/candy.pth",
        "size_mb": 6.4,
        "description": "Candy style - colorful and vibrant",
        "source": "yakhyo/fast-neural-style-transfer"
    },
    "mosaic": {
        "url": "https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/mosaic.pth",
        "size_mb": 6.4,
        "description": "Mosaic - tile mosaic style",
        "source": "yakhyo/fast-neural-style-transfer"
    },
    "udnie": {
        "url": "https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/udnie.pth",
        "size_mb": 6.4,
        "description": "Udnie - abstract expressionist style",
        "source": "yakhyo/fast-neural-style-transfer"
    },
    "rain_princess": {
        "url": "https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/rain-princess.pth",
        "size_mb": 6.4,
        "description": "Rain Princess - impressionist style",
        "source": "yakhyo/fast-neural-style-transfer"
    },
    "starry": {
        "url": "https://raw.githubusercontent.com/rrmina/fast-neural-style-pytorch/master/transforms/starry.pth",
        "size_mb": 6.5,
        "description": "Starry Night - Van Gogh style",
        "source": "rrmina/fast-neural-style-pytorch"
    },
    "wave": {
        "url": "https://raw.githubusercontent.com/rrmina/fast-neural-style-pytorch/master/transforms/wave.pth",
        "size_mb": 6.5,
        "description": "The Great Wave - Japanese art style",
        "source": "rrmina/fast-neural-style-pytorch"
    },
}


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file with progress bar."""
    print(f"\n{'='*60}")
    print(f"Downloading: {description or output_path.name}")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 8192

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)

                # Print progress
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / 1024 / 1024
                    mb_total = total_size / 1024 / 1024
                    print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

        print()  # New line after progress

        # Verify file was downloaded
        if output_path.stat().st_size > 0:
            actual_size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"✅ Downloaded successfully ({actual_size_mb:.1f} MB)")
            return True
        else:
            print("❌ Download failed - empty file")
            return False

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        return False


def verify_file(output_path: Path, expected_size_mb: float) -> bool:
    """Verify downloaded file size is approximately correct."""
    if not output_path.exists():
        return False

    actual_size_mb = output_path.stat().st_size / 1024 / 1024
    tolerance = 1.0  # MB tolerance

    return abs(actual_size_mb - expected_size_mb) < tolerance


def list_styles():
    """List all available styles."""
    print("\n" + "="*60)
    print("Available Pre-Trained Styles")
    print("="*60)
    print(f"{'Style':<15} {'Size':<10} {'Description'}")
    print("-"*60)

    for name, info in STYLES.items():
        print(f"{name:<15} {info['size_mb']:<10.1f} MB {info['description']}")

    print("="*60)
    print(f"Total: {len(STYLES)} styles available")
    print()


def download_style(style_name: str, output_dir: str = "models/pretrained") -> bool:
    """Download a specific style."""
    if style_name not in STYLES:
        print(f"❌ Unknown style: {style_name}")
        print(f"Available styles: {', '.join(STYLES.keys())}")
        return False

    output_dir = Path(output_dir)
    output_path = output_dir / f"{style_name}.pth"

    # Check if already exists
    if output_path.exists():
        info = STYLES[style_name]
        if verify_file(output_path, info['size_mb']):
            print(f"✅ Already downloaded: {output_path}")
            return True
        else:
            print(f"⚠️  File exists but size mismatch, re-downloading...")

    # Download
    info = STYLES[style_name]
    success = download_file(info['url'], output_path, info['description'])

    if success and verify_file(output_path, info['size_mb']):
        return True
    elif success:
        print("⚠️  Downloaded but size verification failed")
        return False
    return False


def download_all(output_dir: str = "models/pretrained") -> bool:
    """Download all available styles."""
    print("Downloading all pre-trained styles...")

    results = {}
    for style_name in STYLES.keys():
        results[style_name] = download_style(style_name, output_dir)

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)

    success_count = sum(1 for v in results.values() if v)
    print(f"Successfully downloaded: {success_count}/{len(results)}")

    for style, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {style}")

    return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained Fast Style Transfer model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available styles
  python download_models.py --list

  # Download a specific style
  python download_models.py --style candy

  # Download all styles
  python download_models.py --all

  # Download to custom directory
  python download_models.py --all --output-dir ./weights
        """
    )

    parser.add_argument(
        '--style', '-s',
        type=str,
        help='Specific style to download'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Download all available styles'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available styles'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='models/pretrained',
        help='Output directory for downloaded weights'
    )

    args = parser.parse_args()

    if args.list:
        list_styles()
        return 0

    if args.all:
        return 0 if download_all(args.output_dir) else 1

    if args.style:
        return 0 if download_style(args.style, args.output_dir) else 1

    # No arguments, show list
    list_styles()
    print("Use --style <name> to download a specific style")
    print("Use --all to download all styles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
