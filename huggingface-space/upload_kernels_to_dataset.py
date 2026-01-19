#!/usr/bin/env python3
"""
Upload pre-compiled CUDA kernels to Hugging Face Dataset.

This avoids git push issues with binary files on Hugging Face Spaces.

Usage:
    python upload_kernels_to_dataset.py

The kernels will be uploaded to: huggingface.co/datasets/oliau/styleforge-kernels
"""

import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("ERROR: huggingface_hub not installed.")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


def upload_kernels():
    """Upload pre-compiled kernels to Hugging Face dataset."""

    # Configuration
    DATASET_ID = "oliau/styleforge-kernels"
    PREBUILT_DIR = Path("kernels/prebuilt")

    print("=" * 60)
    print("StyleForge Kernel Uploader")
    print("=" * 60)
    print()

    # Check if prebuilt directory exists
    if not PREBUILT_DIR.exists():
        print(f"ERROR: Prebuilt directory not found: {PREBUILT_DIR}")
        print("Run compile_kernels.py first to generate the kernels.")
        sys.exit(1)

    # Find all kernel files
    kernel_files = list(PREBUILT_DIR.glob("*.so")) + list(PREBUILT_DIR.glob("*.pyd"))

    if not kernel_files:
        print(f"ERROR: No kernel files found in {PREBUILT_DIR}")
        print("Expected .so or .pyd files.")
        sys.exit(1)

    print(f"Found {len(kernel_files)} kernel file(s):")
    for f in kernel_files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    print()

    # Initialize HF API
    api = HfApi()

    # Check if user is logged in
    try:
        whoami = api.whoami()
        print(f"Logged in as: {whoami.get('name', whoami.get('user', 'unknown'))}")
    except Exception:
        print("Not logged in to Hugging Face.")
        print("Please run: huggingface-cli login")
        print("Or set HF_TOKEN environment variable.")
        sys.exit(1)

    print()
    print(f"Uploading to dataset: {DATASET_ID}")
    print()

    # Create dataset if it doesn't exist
    try:
        repo_info = api.repo_info(DATASET_ID, repo_type="dataset")
        print(f"Dataset exists: {DATASET_ID}")
    except Exception:
        print(f"Creating new dataset: {DATASET_ID}")
        api.create_repo(
            repo_id=DATASET_ID.split("/")[1],
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"Dataset created: {DATASET_ID}")

    # Create README for the dataset
    readme_content = """---
title: StyleForge CUDA Kernels
license: mit
tags:
  - cuda
  - neural-style-transfer
  - styleforge
---

# StyleForge Pre-compiled CUDA Kernels

This repository contains pre-compiled CUDA kernels for the StyleForge neural style transfer project.

## Files

"""
    for f in kernel_files:
        readme_content += f"- `{f.name}`\n"

    readme_content += """
## Usage

These kernels are automatically downloaded by StyleForge when running on Hugging Face Spaces.

## Compilation

Kernels are compiled for multiple GPU architectures:
- sm_70 (V100)
- sm_75 (T4)
- sm_80 (A100)

For local compilation, see `compile_kernels.py` in the main repository.
"""

    # Upload files
    print("Uploading files...")

    # Upload README
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=DATASET_ID,
        repo_type="dataset",
        commit_message="Add dataset README"
    )
    print("  Uploaded: README.md")

    # Upload kernel files
    for kernel_file in kernel_files:
        print(f"  Uploading {kernel_file.name}...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(kernel_file),
            path_in_repo=kernel_file.name,
            repo_id=DATASET_ID,
            repo_type="dataset",
            commit_message=f"Add {kernel_file.name}"
        )
        print("✓")

    print()
    print("=" * 60)
    print("Upload complete!")
    print("=" * 60)
    print()
    print(f"Dataset URL: https://huggingface.co/datasets/{DATASET_ID}")
    print()
    print("The kernels will be automatically downloaded by StyleForge")
    print("when running on Hugging Face Spaces.")


if __name__ == "__main__":
    upload_kernels()
