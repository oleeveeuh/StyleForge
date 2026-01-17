"""
Test configuration for StyleForge verification tests.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = PROJECT_ROOT / "tests"
TEST_OUTPUTS_DIR = PROJECT_ROOT / "test_outputs"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
MODELS_DIR = PROJECT_ROOT / "models" / "pretrained"

# Test image URLs (for downloading test data if not present)
TEST_IMAGE_URLS = {
    "portrait": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=512",
    "landscape": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=512",
    "building": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=512",
}

# Test configurations
TEST_IMAGE_SIZES = [(256, 256), (512, 512), (1024, 1024)]
AVAILABLE_STYLES = ["candy", "mosaic", "udnie", "rain_princess", "starry", "wave"]

# Numerical tolerance thresholds
NUMERICAL_TOLERANCE = {
    "max_abs_diff": 1e-3,
    "mean_abs_diff": 1e-4,
    "output_min": -1.0,
    "output_max": 256.0,
}

# Benchmark settings
BENCHMARK_WARMUP_ITERS = 10
BENCHMARK_TEST_ITERS = 50

# Device settings
def get_device():
    """Get the best available device for testing."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
IS_CUDA_AVAILABLE = DEVICE.type == "cuda"

# Memory test settings (defined after device detection)
MEMORY_TEST_ITERATIONS = 1000
MEMORY_TEST_WARMUP = 10
# Use larger threshold for CPU/MPS since we measure RSS which can fluctuate more
MEMORY_ALLOWABLE_GROWTH_MB = 10 if IS_CUDA_AVAILABLE else 30
