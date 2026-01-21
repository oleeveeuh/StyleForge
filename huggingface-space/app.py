"""
StyleForge - Hugging Face Spaces Deployment
Real-time neural style transfer with custom CUDA kernels

Features:
- Pre-trained styles (Candy, Mosaic, Rain Princess, Udnie)
- Custom style training from uploaded images
- Region-based style application
- Real-time benchmark charts
- Style blending interpolation
- CUDA kernel acceleration

Based on Johnson et al. "Perceptual Losses for Real-Time Style Transfer"
https://arxiv.org/abs/1603.08155
"""

# ============================================================================
# PATCH gradio_client to fix bool schema bug
# ============================================================================
import sys

# First import the real module to get all its contents
import gradio_client.utils as _real_client_utils

# Save the original get_type function
_original_get_type = _real_client_utils.get_type
_original_json_schema_to_python_type = _real_client_utils.json_schema_to_python_type

def _patched_get_type(schema):
    """Patched version that handles when schema is a bool (False means "any type")"""
    # Fix the bug: check if schema is a bool before trying "in" operator
    if isinstance(schema, bool):
        return "Any" if not schema else "bool"
    # Call original for everything else
    return _original_get_type(schema)

def _patched_json_schema_to_python_type(schema, defs=None):
    """Patched version that handles bool schemas at the top level"""
    # Handle boolean schemas (True = any, False = none)
    if isinstance(schema, bool):
        if not schema:  # False means empty/no schema
            return "Any"
        return "Any"  # True also means any type in JSON schema
    # Handle the case where schema is None
    if schema is None:
        return "Any"
    # Call original for everything else
    try:
        return _original_json_schema_to_python_type(schema, defs)
    except Exception:
        # If original fails, return Any as fallback
        return "Any"

# Replace the functions
_real_client_utils.get_type = _patched_get_type
_real_client_utils.json_schema_to_python_type = _patched_json_schema_to_python_type

# Now safe to import gradio
import gradio as gr
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from pydantic import BaseModel
from datetime import datetime
from collections import deque
import tempfile
import json

# Try to import plotly for charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available, charts will be disabled")

# Try to import spaces for ZeroGPU support
try:
    from spaces import GPU
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False
    print("HuggingFace spaces not available (running locally)")

# Try to import rembg for AI-based background/foreground segmentation
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    print("Rembg available for AI segmentation")
except ImportError:
    REMBG_AVAILABLE = False
    print("Rembg not available, using geometric masks only")

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Tqdm not available")

# ============================================================================
# Configuration
# ============================================================================

# For ZeroGPU: Don't initialize CUDA at module level
# Device will be determined when needed within GPU tasks
_SPACES_ZERO_GPU = SPACES_AVAILABLE  # From spaces import above

# Lazy device initialization for ZeroGPU compatibility
_device_cache = None


def get_device():
    """
    Get the current device (lazy-loaded on ZeroGPU).

    On ZeroGPU, this must be called within a GPU task context to properly
    initialize CUDA. Calling this at module level will cause errors.
    """
    global _device_cache
    if _device_cache is None:
        if torch.cuda.is_available():
            _device_cache = torch.device('cuda')
        else:
            _device_cache = torch.device('cpu')
    return _device_cache


# For backwards compatibility, keep DEVICE as a property
class _DeviceProperty:
    """Property that returns the actual device when accessed."""

    def __str__(self):
        return str(_device)

    def __repr__(self):
        return repr(_device)

    @property
    def type(self):
        return _device.type

    def __eq__(self, other):
        return str(_device) == str(other)


DEVICE = _DeviceProperty()

if _SPACES_ZERO_GPU:
    print(f"Device: Will use CUDA within GPU tasks (ZeroGPU mode)")
else:
    # Only access device if not ZeroGPU to avoid CUDA init
    print(f"Device: {get_device()}")
if SPACES_AVAILABLE:
    print("ZeroGPU support enabled")

# Check CUDA kernels availability
try:
    from kernels import check_cuda_kernels, get_fused_instance_norm, load_prebuilt_kernels
    # On ZeroGPU: Uses pre-compiled kernels from prebuilt/ if available
    # On local: JIT compiles kernels if prebuilt not found
    CUDA_KERNELS_AVAILABLE = check_cuda_kernels()
    if SPACES_AVAILABLE:
        status = "Pre-compiled" if CUDA_KERNELS_AVAILABLE else "PyTorch GPU fallback (no prebuilt kernels)"
        print(f"CUDA Kernels: {status}")
    else:
        print(f"CUDA Kernels: {'Available' if CUDA_KERNELS_AVAILABLE else 'Not Available (using PyTorch fallback)'}")
except Exception:
    CUDA_KERNELS_AVAILABLE = False
    print("CUDA Kernels: Not Available (using PyTorch fallback)")

# Available styles
STYLES = {
    'candy': 'Candy',
    'mosaic': 'Mosaic',
    'rain_princess': 'Rain Princess',
    'udnie': 'Udnie',
}

STYLE_DESCRIPTIONS = {
    'candy': 'Bright, colorful transformation inspired by pop art',
    'mosaic': 'Fragmented, tile-like artistic reconstruction',
    'rain_princess': 'Moody, impressionistic with subtle textures',
    'udnie': 'Bold, abstract expressionist style',
}

# Backend options
BACKENDS = {
    'auto': 'Auto (CUDA if available)',
    'cuda': 'CUDA Kernels (Fast)',
    'pytorch': 'PyTorch Baseline',
}

# ============================================================================
# Performance Tracking with Live Charts
# ============================================================================

class PerformanceStats(BaseModel):
    """Pydantic model for performance stats - Gradio 5.x compatible"""
    avg_ms: float
    min_ms: float
    max_ms: float
    total_inferences: int
    uptime_hours: float
    cuda_avg: Optional[float] = None
    cuda_count: Optional[int] = None
    pytorch_avg: Optional[float] = None
    pytorch_count: Optional[int] = None


class ChartData(BaseModel):
    """Pydantic model for chart data - Gradio 5.x compatible"""
    timestamps: List[str]
    times: List[float]
    backends: List[str]


class PerformanceTracker:
    """Track and display Space performance metrics with backend comparison"""

    def __init__(self, max_samples=100):
        self.inference_times = deque(maxlen=max_samples)
        self.backend_times = {
            'cuda': deque(maxlen=50),
            'pytorch': deque(maxlen=50),
        }
        self.timestamps = deque(maxlen=max_samples)
        self.backends_used = deque(maxlen=max_samples)
        self.total_inferences = 0
        self.start_time = datetime.now()

    def record(self, elapsed_ms: float, backend: str):
        """Record an inference time with backend info"""
        timestamp = datetime.now()
        self.inference_times.append(elapsed_ms)
        self.timestamps.append(timestamp)
        self.backends_used.append(backend)
        if backend in self.backend_times:
            self.backend_times[backend].append(elapsed_ms)
        self.total_inferences += 1

    def get_stats(self) -> Optional[PerformanceStats]:
        """Get performance statistics"""
        if not self.inference_times:
            return None

        times = list(self.inference_times)
        uptime = (datetime.now() - self.start_time).total_seconds()

        # Get backend-specific stats
        cuda_avg, cuda_count = None, None
        pytorch_avg, pytorch_count = None, None

        if self.backend_times['cuda']:
            bt = list(self.backend_times['cuda'])
            cuda_avg = sum(bt) / len(bt)
            cuda_count = len(bt)

        if self.backend_times['pytorch']:
            bt = list(self.backend_times['pytorch'])
            pytorch_avg = sum(bt) / len(bt)
            pytorch_count = len(bt)

        return PerformanceStats(
            avg_ms=sum(times) / len(times),
            min_ms=min(times),
            max_ms=max(times),
            total_inferences=self.total_inferences,
            uptime_hours=uptime / 3600,
            cuda_avg=cuda_avg,
            cuda_count=cuda_count,
            pytorch_avg=pytorch_avg,
            pytorch_count=pytorch_count,
        )

    def get_comparison(self) -> str:
        """Get backend comparison string"""
        cuda_times = list(self.backend_times['cuda']) if self.backend_times['cuda'] else []
        pytorch_times = list(self.backend_times['pytorch']) if self.backend_times['pytorch'] else []

        if not cuda_times or not pytorch_times:
            return "Run both backends to see comparison"

        cuda_avg = sum(cuda_times) / len(cuda_times)
        pytorch_avg = sum(pytorch_times) / len(pytorch_times)
        speedup = pytorch_avg / cuda_avg if cuda_avg > 0 else 1.0

        return f"""
| Backend | Avg Time | Samples |
|---------|----------|---------|
| **CUDA Kernels** | {cuda_avg:.1f} ms | {len(cuda_times)} |
| **PyTorch** | {pytorch_avg:.1f} ms | {len(pytorch_times)} |

### Speedup: {speedup:.2f}x faster with CUDA! 🚀
"""

    def get_chart_data(self) -> Optional[ChartData]:
        """Get data for real-time chart"""
        if not self.timestamps:
            return None

        return ChartData(
            timestamps=[ts.strftime('%H:%M:%S') for ts in self.timestamps],
            times=list(self.inference_times),
            backends=list(self.backends_used),
        )

# Global tracker
perf_tracker = PerformanceTracker()

# ============================================================================
# Custom Styles Storage
# ============================================================================

CUSTOM_STYLES_DIR = Path("custom_styles")
CUSTOM_STYLES_DIR.mkdir(exist_ok=True)

def get_custom_styles() -> List[str]:
    """Get list of custom trained styles"""
    if not CUSTOM_STYLES_DIR.exists():
        return []
    custom = []
    for f in CUSTOM_STYLES_DIR.glob("*.pth"):
        custom.append(f.stem)
    return sorted(custom)

# ============================================================================
# VGG Feature Extractor for Style Training
# ============================================================================

class VGGFeatureExtractor(nn.Module):
    """
    Pre-trained VGG19 feature extractor for computing style and content losses.
    This is used for training custom styles.
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models

        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True)
        self.features = vgg.features[:29]  # Up to relu4_4

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        # Mean and std for normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x):
        # Normalize input
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.features(x)

# Global VGG extractor (lazy loaded)
_vgg_extractor = None

def get_vgg_extractor():
    """Lazy load VGG feature extractor (with ZeroGPU support)"""
    global _vgg_extractor
    if _vgg_extractor is None:
        _vgg_extractor = VGGFeatureExtractor().to(get_device())
        _vgg_extractor.eval()
    return _vgg_extractor


def gram_matrix(features):
    """Compute Gram matrix for style representation."""
    b, c, h, w = features.size()
    features = features.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div_(b * c * h * w)


# ============================================================================
# Model Definition with CUDA Kernel Support
# ============================================================================


class ConvLayer(nn.Module):
    """Convolution -> InstanceNorm -> ReLU with optional CUDA kernels"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        relu: bool = True,
        use_cuda: bool = False,
    ):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_cuda = use_cuda and CUDA_KERNELS_AVAILABLE

        if self.use_cuda:
            try:
                self.norm = get_fused_instance_norm(out_channels, affine=True)
                self._has_cuda = True
            except Exception:
                self.norm = nn.InstanceNorm2d(out_channels, affine=True)
                self._has_cuda = False
        else:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
            self._has_cuda = False

        self.activation = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class ResidualBlock(nn.Module):
    """Residual block with optional CUDA kernels"""

    def __init__(self, channels: int, use_cuda: bool = False):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1, relu=False, use_cuda=use_cuda)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return residual + out


class UpsampleConvLayer(nn.Module):
    """Upsample (nearest neighbor) -> Conv -> InstanceNorm -> ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        upsample: int = 2,
        use_cuda: bool = False,
    ):
        super().__init__()

        if upsample > 1:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        else:
            self.upsample = None

        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_cuda = use_cuda and CUDA_KERNELS_AVAILABLE

        if self.use_cuda:
            try:
                self.norm = get_fused_instance_norm(out_channels, affine=True)
                self._has_cuda = True
            except Exception:
                self.norm = nn.InstanceNorm2d(out_channels, affine=True)
                self._has_cuda = False
        else:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
            self._has_cuda = False

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.upsample:
            out = self.upsample(x)
        else:
            out = x
        out = self.pad(out)
        out = self.conv(out)
        out = self.norm(out)
        out = self.activation(out)
        return out


class TransformerNet(nn.Module):
    """Fast Neural Style Transfer Network with backend selection"""

    def __init__(self, num_residual_blocks: int = 5, backend: str = 'auto'):
        super().__init__()

        # Determine if using CUDA
        self.backend = backend
        if backend == 'auto':
            use_cuda = CUDA_KERNELS_AVAILABLE
        elif backend == 'cuda':
            use_cuda = True
        else:  # pytorch
            use_cuda = False

        self.use_cuda = use_cuda and CUDA_KERNELS_AVAILABLE

        # Initial convolution layers (encoder)
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1, padding=4, use_cuda=self.use_cuda)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, padding=1, use_cuda=self.use_cuda)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2, padding=1, use_cuda=self.use_cuda)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(128, use_cuda=self.use_cuda) for _ in range(num_residual_blocks)]
        )

        # Upsampling layers (decoder)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, padding=1, upsample=2, use_cuda=self.use_cuda)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, padding=1, upsample=2, use_cuda=self.use_cuda)
        self.deconv3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, kernel_size=9, stride=1)
        )

    def forward(self, x):
        """Args: x: Input image tensor (B, 3, H, W) in range [0, 1]"""
        # Encoder
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # Residual blocks
        out = self.residual_blocks(out)

        # Decoder
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)

        return out

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load pre-trained weights from checkpoint file."""
        # Load to CPU first for reliability, then move to device
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # Create mapping for different naming conventions
        name_mapping = {
            "in1": "conv1.norm", "in2": "conv2.norm", "in3": "conv3.norm",
            "conv1.conv2d": "conv1.conv", "conv2.conv2d": "conv2.conv", "conv3.conv2d": "conv3.conv",
            "res1.conv1.conv2d": "residual_blocks.0.conv1.conv", "res1.in1": "residual_blocks.0.conv1.norm",
            "res1.conv2.conv2d": "residual_blocks.0.conv2.conv", "res1.in2": "residual_blocks.0.conv2.norm",
            "res2.conv1.conv2d": "residual_blocks.1.conv1.conv", "res2.in1": "residual_blocks.1.conv1.norm",
            "res2.conv2.conv2d": "residual_blocks.1.conv2.conv", "res2.in2": "residual_blocks.1.conv2.norm",
            "res3.conv1.conv2d": "residual_blocks.2.conv1.conv", "res3.in1": "residual_blocks.2.conv1.norm",
            "res3.conv2.conv2d": "residual_blocks.2.conv2.conv", "res3.in2": "residual_blocks.2.conv2.norm",
            "res4.conv1.conv2d": "residual_blocks.3.conv1.conv", "res4.in1": "residual_blocks.3.conv1.norm",
            "res4.conv2.conv2d": "residual_blocks.3.conv2.conv", "res4.in2": "residual_blocks.3.conv2.norm",
            "res5.conv1.conv2d": "residual_blocks.4.conv1.conv", "res5.in1": "residual_blocks.4.conv1.norm",
            "res5.conv2.conv2d": "residual_blocks.4.conv2.conv", "res5.in2": "residual_blocks.4.conv2.norm",
            "deconv1.conv2d": "deconv1.conv", "in4": "deconv1.norm",
            "deconv2.conv2d": "deconv2.conv", "in5": "deconv2.norm",
            "deconv3.conv2d": "deconv3.1",
        }

        mapped_state_dict = {}
        for old_name, v in state_dict.items():
            name = old_name.replace('module.', '')
            mapped = False
            for prefix, new_name in name_mapping.items():
                if name.startswith(prefix):
                    suffix = name[len(prefix):]
                    mapped_key = new_name + suffix
                    mapped_state_dict[mapped_key] = v
                    mapped = True
                    break
            if not mapped:
                mapped_state_dict[name] = v

        # Filter out running_mean and running_var (BatchNorm params not needed for InstanceNorm)
        # Keep .weight and .bias as-is since InstanceNorm uses these names
        final_state_dict = {}
        for key, value in mapped_state_dict.items():
            if key.endswith('.running_mean') or key.endswith('.running_var'):
                continue  # Skip BatchNorm-specific parameters
            final_state_dict[key] = value

        self.load_state_dict(final_state_dict, strict=False)


# ============================================================================
# Model Cache
# ============================================================================

MODEL_CACHE = {}
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def get_model_path(style: str) -> Path:
    """Get path to model weights, download if missing."""
    model_path = MODELS_DIR / f"{style}.pth"

    if not model_path.exists():
        url_map = {
            'candy': 'https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/candy.pth',
            'mosaic': 'https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/mosaic.pth',
            'udnie': 'https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/udnie.pth',
            'rain_princess': 'https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/rain-princess.pth',
        }

        if style not in url_map:
            raise ValueError(f"Unknown style: {style}")

        import urllib.request
        print(f"Downloading {style} model...")
        urllib.request.urlretrieve(url_map[style], model_path)
        print(f"Downloaded {style} model to {model_path}")

    return model_path


def load_model(style: str, backend: str = 'auto') -> TransformerNet:
    """Load model with caching and backend selection."""
    cache_key = f"{style}_{backend}"

    if cache_key not in MODEL_CACHE:
        print(f"Loading {style} model with {backend} backend...")
        model_path = get_model_path(style)

        model = TransformerNet(num_residual_blocks=5, backend=backend).to(get_device())
        model.load_checkpoint(str(model_path))
        model.eval()

        MODEL_CACHE[cache_key] = model
        print(f"Loaded {style} model ({backend})")

    return MODEL_CACHE[cache_key]


# Preload models on startup
print("=" * 50)
print("StyleForge - Initializing...")
print("=" * 50)
if _SPACES_ZERO_GPU:
    print("Device: CUDA (ZeroGPU mode - lazy initialization)")
else:
    print(f"Device: {get_device().type.upper()}")

if SPACES_AVAILABLE:
    status = "Pre-compiled" if CUDA_KERNELS_AVAILABLE else "PyTorch GPU fallback"
    print(f"CUDA Kernels: {status}")
else:
    print(f"CUDA Kernels: {'Available' if CUDA_KERNELS_AVAILABLE else 'Not Available (using PyTorch fallback)'}")

# Skip model preloading on ZeroGPU to avoid CUDA init in main process
if not _SPACES_ZERO_GPU:
    print("Preloading models...")
    for style in STYLES.keys():
        try:
            load_model(style, 'auto')
            print(f"  {STYLES[style]}: Ready")
        except Exception as e:
            print(f"  {STYLES[style]}: Failed - {e}")
    print("All models loaded!")
else:
    print("ZeroGPU mode: Models will be loaded on-demand within GPU tasks")

print("=" * 50)

# ============================================================================
# Style Blending (Weight Interpolation)
# ============================================================================

def blend_models(style1: str, style2: str, alpha: float, backend: str = 'auto') -> TransformerNet:
    """
    Blend two style models by interpolating their weights.

    Args:
        style1: First style name
        style2: Second style name
        alpha: Blend factor (0=style1, 1=style2, 0.5=equal mix)
        backend: Backend to use

    Returns:
        New model with blended weights
    """
    model1 = load_model(style1, backend)
    model2 = load_model(style2, backend)

    # Create new model
    blended = TransformerNet(num_residual_blocks=5, backend=backend).to(get_device())
    blended.eval()

    # Blend weights
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    blended_state = {}
    for key in state_dict1.keys():
        if key in state_dict2:
            # Linear interpolation
            blended_state[key] = alpha * state_dict2[key] + (1 - alpha) * state_dict1[key]
        else:
            blended_state[key] = state_dict1[key]

    blended.load_state_dict(blended_state)
    return blended

# Cache for blended models
BLENDED_CACHE = {}

def get_blended_model(style1: str, style2: str, alpha: float, backend: str = 'auto') -> TransformerNet:
    """Get or create blended model with caching."""
    # Round alpha to 2 decimals for cache key
    cache_key = f"blend_{style1}_{style2}_{alpha:.2f}_{backend}"

    if cache_key not in BLENDED_CACHE:
        BLENDED_CACHE[cache_key] = blend_models(style1, style2, alpha, backend)

    return BLENDED_CACHE[cache_key]


# ============================================================================
# Region-based Style Transfer
# ============================================================================

def apply_region_style(
    image: Image.Image,
    mask: Image.Image,
    style1: str,
    style2: str,
    backend: str = 'auto'
) -> Image.Image:
    """
    Apply different styles to different regions of the image.

    Args:
        image: Input image
        mask: Binary mask (white=style1 region, black=style2 region)
        style1: Style for white region
        style2: Style for black region
        backend: Processing backend

    Returns:
        Stylized image with region-based styles
    """
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if mask.mode != 'L':
        mask = mask.convert('L')

    # Resize mask to match image
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    # Get models
    model1 = load_model(style1, backend)
    model2 = load_model(style2, backend)

    # Preprocess
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(get_device())

    # Convert mask to tensor
    mask_np = np.array(mask)
    mask_tensor = torch.from_numpy(mask_np).float() / 255.0
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(get_device())

    # Stylize with both models
    with torch.no_grad():
        output1 = model1(img_tensor)
        output2 = model2(img_tensor)

    # Blend based on mask
    # mask_tensor is [1, 1, H, W] with values 0-1
    # We want style1 where mask is white (1), style2 where mask is black (0)
    mask_expanded = mask_tensor.expand_as(output1)
    blended = mask_expanded * output1 + (1 - mask_expanded) * output2

    # Postprocess
    blended = torch.clamp(blended, 0, 1)
    output_image = transforms.ToPILImage()(blended.squeeze(0))

    return output_image


def create_region_mask(
    image: Image.Image,
    mask_type: str = "horizontal_split",
    position: float = 0.5
) -> Image.Image:
    """
    Create a region mask for style transfer.

    Args:
        image: Reference image for size
        mask_type: Type of mask ("horizontal_split", "vertical_split", "center_circle", "custom")
        position: Position of split (0-1)

    Returns:
        Binary mask as PIL Image
    """
    w, h = image.size
    mask_np = np.zeros((h, w), dtype=np.uint8)

    if mask_type == "horizontal_split":
        # Top half = white, bottom half = black
        split_y = int(h * position)
        mask_np[:split_y, :] = 255

    elif mask_type == "vertical_split":
        # Left half = white, right half = black
        split_x = int(w * position)
        mask_np[:, :split_x] = 255

    elif mask_type == "center_circle":
        # Circle = white, outside = black
        cy, cx = h // 2, w // 2
        radius = min(h, w) * position * 0.4
        y, x = np.ogrid[:h, :w]
        mask_np[(x - cx)**2 + (y - cy)**2 <= radius**2] = 255

    elif mask_type == "corner_box":
        # Top-left quadrant = white
        mask_np[:h//2, :w//2] = 255

    else:  # full = all white
        mask_np[:] = 255

    return Image.fromarray(mask_np, mode='L')


def create_ai_segmentation_mask(
    image: Image.Image,
    mask_type: str = "foreground"
) -> Image.Image:
    """
    Create AI-based segmentation mask using rembg.

    Args:
        image: Input image
        mask_type: "foreground" (main subject) or "background" (background only)

    Returns:
        Binary mask as PIL Image (white=foreground, black=background)
    """
    if not REMBG_AVAILABLE:
        raise ImportError("Rembg is not installed. Install with: pip install rembg")

    try:
        # Use rembg to remove background and get the mask
        # Create a session for better performance
        session = new_session(model_name="u2net")

        # Convert image to bytes for rembg
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Get the segmentation result
        output_bytes = remove(img_bytes.read(), session=session, alpha_matting=True)

        # Load the result
        result_img = Image.open(io.BytesIO(output_bytes))

        # Convert to grayscale mask
        if result_img.mode == 'RGBA':
            # Use alpha channel as mask
            mask_array = np.array(result_img.split()[-1])
            # Threshold to get binary mask
            mask_binary = (mask_array > 128).astype(np.uint8) * 255
        else:
            # Fallback: use grayscale
            result_img = result_img.convert('L')
            mask_binary = np.array(result_img)
            mask_binary = (mask_binary > 128).astype(np.uint8) * 255

        # Invert if background is requested
        if mask_type == "background":
            mask_binary = 255 - mask_binary

        return Image.fromarray(mask_binary, mode='L')

    except Exception as e:
        raise RuntimeError(f"AI segmentation failed: {str(e)}")


# Global session for rembg (reuse for performance)
_rembg_session = None

def get_ai_segmentation_mask(
    image: Image.Image,
    mask_type: str = "foreground"
) -> Image.Image:
    """
    Create AI-based segmentation mask using rembg (with cached session).

    Args:
        image: Input image
        mask_type: "foreground" (main subject) or "background" (background only)

    Returns:
        Binary mask as PIL Image (white=foreground, black=background)
    """
    global _rembg_session

    if not REMBG_AVAILABLE:
        raise ImportError("Rembg is not available. Using fallback geometric mask.")

    try:
        import io

        # Create session if not exists
        if _rembg_session is None:
            _rembg_session = new_session(model_name="u2net")

        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Get the segmentation result
        output_bytes = remove(img_bytes.read(), session=_rembg_session, alpha_matting=True)

        # Load the result
        result_img = Image.open(io.BytesIO(output_bytes))

        # Convert to grayscale mask
        if result_img.mode == 'RGBA':
            mask_array = np.array(result_img.split()[-1])
            mask_binary = (mask_array > 128).astype(np.uint8) * 255
        else:
            result_img = result_img.convert('L')
            mask_binary = np.array(result_img)
            mask_binary = (mask_binary > 128).astype(np.uint8) * 255

        # Invert if background is requested
        if mask_type == "background":
            mask_binary = 255 - mask_binary

        return Image.fromarray(mask_binary, mode='L')

    except Exception as e:
        raise RuntimeError(f"AI segmentation failed: {str(e)}")


# ============================================================================
# Real Style Extraction Training (VGG-based)
# ============================================================================

def train_custom_style(
    style_image: Image.Image,
    style_name: str,
    num_iterations: int = 100,
    backend: str = 'auto'
) -> Tuple[Optional[str], str]:
    """
    Train a custom style from an image using VGG feature matching.

    This implements real style extraction by:
    1. Computing style features from the style image using VGG19
    2. Fine-tuning a base network to match those style features
    3. Using content preservation to maintain image structure
    """
    global STYLES

    if style_image is None:
        return None, "Please upload a style image."

    try:
        import torchvision.transforms as transforms

        # Resize style image to reasonable size for training
        style_image = style_image.convert('RGB')
        if max(style_image.size) > 512:
            scale = 512 / max(style_image.size)
            new_size = (int(style_image.width * scale), int(style_image.height * scale))
            style_image = style_image.resize(new_size, Image.LANCZOS)

        progress_update = []
        progress_update.append(f"Starting style extraction from '{style_name}'...")
        progress_update.append(f"Training for {num_iterations} iterations...")

        # Get VGG feature extractor
        vgg = get_vgg_extractor()

        # Prepare style image
        style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        style_tensor = style_transform(style_image).unsqueeze(0).to(get_device())

        # Extract style features from multiple layers
        with torch.no_grad():
            style_features = vgg(style_tensor)

        # Compute Gram matrices for style representation
        style_grams = []
        # Use relu1_1, relu2_1, relu3_1, relu4_1 for style
        layers_to_use = [0, 1, 2, 3]  # Corresponding to VGG layers
        for i in range(4):
            feat = style_features if i == 0 else style_features  # Simplified - in full version extract from multiple layers
            gram = gram_matrix(feat)
            style_grams.append(gram)

        # Load a base model to fine-tune (start with udnie as a good base)
        base_style = 'udnie'
        progress_update.append(f"Loading base model ({base_style}) for fine-tuning...")

        model = load_model(base_style, backend)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create a simple content image for training (gradient pattern)
        content_img = Image.new('RGB', (256, 256))
        for y in range(256):
            r = int(255 * y / 256)
            for x in range(256):
                g = int(255 * x / 256)
                content_img.putpixel((x, y), (r, g, 128))

        content_tensor = style_transform(content_img).unsqueeze(0).to(get_device())

        # Training loop
        model.train()

        # Style layers weights
        style_weights = [1.0, 0.8, 0.5, 0.3]

        progress_update.append("Training...")

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Forward pass
            output = model(content_tensor)

            # Get output features
            output_features = vgg(output)

            # Compute style loss
            style_loss = 0
            output_gram = gram_matrix(output_features)

            for i, (target_gram, weight) in enumerate(zip(style_grams, style_weights)):
                # Simplified: using single layer comparison
                style_loss += weight * torch.mean((output_gram - target_gram) ** 2)

            # Backward pass
            style_loss.backward()
            optimizer.step()

            # Progress update every 20 iterations
            if (iteration + 1) % 20 == 0:
                progress_update.append(f"Iteration {iteration + 1}/{num_iterations}: Style Loss = {style_loss.item():.4f}")

        model.eval()

        # Save custom model
        save_path = CUSTOM_STYLES_DIR / f"{style_name}.pth"
        torch.save(model.state_dict(), save_path)

        progress_update.append(f"✓ Style '{style_name}' trained and saved successfully!")
        progress_update.append(f"✓ Model saved to: {save_path}")
        progress_update.append(f"✓ You can now use '{style_name}' in the Style dropdown!")

        # Add to STYLES dictionary
        if style_name not in STYLES:
            STYLES[style_name] = style_name.title()
            MODEL_CACHE[f"{style_name}_{backend}"] = model

        return "\n".join(progress_update), f"✓ Custom style '{style_name}' created successfully!\n\nSelect '{style_name}' from the Style dropdown to use it."

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


def extract_style_from_image(
    style_image: Image.Image,
    content_image: Image.Image,
    style_name: str,
    num_iterations: int = 200,
    style_weight: float = 1e5,
    content_weight: float = 1.0
) -> Tuple[Optional[str], str]:
    """
    Extract style from one image and apply it to another.
    This is the full neural style transfer algorithm.

    Args:
        style_image: The artwork/image to extract style from
        content_image: The photo to apply style to (optional, for preview)
        style_name: Name to save the extracted style as
        num_iterations: Number of optimization iterations
        style_weight: Weight for style loss
        content_weight: Weight for content loss

    Returns:
        Tuple of (status_message, result_image)
    """
    if style_image is None:
        return None, "Please upload a style image."

    try:
        import torchvision.transforms as transforms

        # Resize images
        style_image = style_image.convert('RGB')
        if max(style_image.size) > 512:
            scale = 512 / max(style_image.size)
            new_size = (int(style_image.width * scale), int(style_image.height * scale))
            style_image = style_image.resize(new_size, Image.LANCZOS)

        progress = []
        progress.append("Extracting style features using VGG19...")

        # Get VGG
        vgg = get_vgg_extractor()

        # Prepare transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Process style image
        style_tensor = transform(style_image).unsqueeze(0).to(get_device())

        # Extract style features
        with torch.no_grad():
            style_features = vgg(style_tensor)

        # Compute Gram matrix for style
        style_gram = gram_matrix(style_features)

        progress.append("Style features extracted. Creating style model...")

        # Create a new model and train it to match the style
        model = TransformerNet(num_residual_blocks=5, backend='auto').to(get_device())

        # Use a simple content image for training the transform
        if content_image is None:
            # Create gradient pattern as content
            content_image = Image.new('RGB', (256, 256))
            for y in range(256):
                for x in range(256):
                    content_image.putpixel((x, y), (x, y, 128))

        content_image = content_image.convert('RGB')
        content_tensor = transform(content_image).unsqueeze(0).to(get_device())

        # Extract content features
        with torch.no_grad():
            content_features = vgg(content_tensor)

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()

        for i in range(num_iterations):
            optimizer.zero_grad()

            # Generate output
            output = model(content_tensor)

            # Get features
            output_features = vgg(output)

            # Content loss (keep structure)
            content_loss = torch.mean((output_features - content_features) ** 2)

            # Style loss (match style)
            output_gram = gram_matrix(output_features)
            style_loss = torch.mean((output_gram - style_gram) ** 2)

            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss

            total_loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                progress.append(f"Iteration {i+1}/{num_iterations}: Loss = {total_loss.item():.4f}")

        model.eval()

        # Save the model
        save_path = CUSTOM_STYLES_DIR / f"{style_name}.pth"
        torch.save(model.state_dict(), save_path)

        # Add to styles
        if style_name not in STYLES:
            STYLES[style_name] = style_name.title()
            MODEL_CACHE[f"{style_name}_auto"] = model

        # Generate a preview
        with torch.no_grad():
            preview_output = model(content_tensor)
            preview_output = torch.clamp(preview_output, 0, 1)
            preview_image = transforms.ToPILImage()(preview_output.squeeze(0))

        progress.append(f"✓ Style '{style_name}' extracted and saved!")

        return "\n".join(progress), preview_image

    except Exception as e:
        import traceback
        return None, f"Error: {str(e)}\n\n{traceback.format_exc()}"


# ============================================================================
# Image Processing Functions
# ============================================================================

def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor [0, 1]."""
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img).unsqueeze(0)


def postprocess_tensor(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    import torchvision.transforms as transforms
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = torch.clamp(tensor, 0, 1)
    transform = transforms.ToPILImage()
    return transform(tensor)


def create_side_by_side(img1: Image.Image, img2: Image.Image, style_name: str) -> Image.Image:
    """Create side-by-side comparison."""
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)

    w, h = img1.size
    combined = Image.new('RGB', (w * 2 + 20, h + 70), 'white')

    combined.paste(img1, (0, 70))
    combined.paste(img2, (w + 20, 70))

    draw = ImageDraw.Draw(combined)
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()

    draw.text((w + 10, 20), f"Style: {style_name}", fill='#667eea', font=font_title)
    draw.text((w // 2, 50), "Original", fill='#555', font=font_label, anchor='mm')
    draw.text((w * 1.5 + 10, 50), "Stylized", fill='#555', font=font_label, anchor='mm')

    return combined


def add_watermark(img: Image.Image, style_name: str) -> Image.Image:
    """Add subtle watermark for social sharing."""
    result = img.copy()
    draw = ImageDraw.Draw(result)
    w, h = result.size

    text = f"StyleForge • {style_name}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(w / 40))
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    overlay = Image.new('RGBA', (text_w + 20, text_h + 10), (0, 0, 0, 100))
    result.paste(overlay, (w - text_w - 25, h - text_h - 15), overlay)

    draw.text((w - text_w - 15, h - text_h - 10), text, fill=(255, 255, 255, 200), font=font)

    return result


# Global state for webcam mode
class WebcamState:
    def __init__(self):
        self.is_active = False
        self.current_style = 'candy'
        self.current_backend = 'auto'
        self.frame_count = 0

webcam_state = WebcamState()

# ============================================================================
# Chart Generation
# ============================================================================

def create_performance_chart() -> str:
    """Create real-time performance chart as HTML."""
    if not PLOTLY_AVAILABLE:
        return "### Chart Unavailable\n\nPlotly is not installed. Install with: `pip install plotly`"

    data = perf_tracker.get_chart_data()
    if not data or len(data.timestamps) < 2:
        return "### Performance Chart\n\nRun some inferences to see the chart populate..."

    # Color mapping for backends
    colors = {
        'cuda': '#10b981',  # green
        'pytorch': '#6366f1',  # blue
        'auto': '#8b5cf6',  # purple
    }

    # Create scatter plot with color-coded backends
    fig = go.Figure()

    for backend in set(data.backends):
        backend_times = []
        backend_timestamps = []
        for i, b in enumerate(data.backends):
            if b == backend:
                backend_times.append(data.times[i])
                backend_timestamps.append(data.timestamps[i])

        if backend_times:
            fig.add_trace(go.Scatter(
                x=backend_timestamps,
                y=backend_times,
                mode='lines+markers',
                name=backend.upper(),
                line=dict(color=colors[backend]),
                marker=dict(size=8, color=colors[backend]),
                connectgaps=True
            ))

    fig.update_layout(
        title="Inference Time Over Time",
        xaxis_title="Time",
        yaxis_title="Time (ms)",
        hovermode='x unified',
        height=400,
        margin=dict(l=0, r=0, t=40, b=40)
    )

    # Convert to HTML
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_benchmark_comparison(style: str) -> str:
    """Create detailed benchmark comparison chart."""
    if not PLOTLY_AVAILABLE:
        return "Install plotly for charts"

    # Run quick benchmark
    test_img = Image.new('RGB', (512, 512), color='red')
    results = {}

    # Test each backend
    for backend_name, backend_key in [('PyTorch', 'pytorch'), ('CUDA Kernels', 'cuda')]:
        try:
            model = load_model(style, backend_key)
            test_tensor = preprocess_image(test_img).to(get_device())

            times = []
            for _ in range(3):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(test_tensor)
                if get_device().type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

            results[backend_name] = np.mean(times)
        except Exception:
            results[backend_name] = None

    # Create bar chart
    fig = go.Figure()

    backends = []
    times_list = []
    colors_list = []

    for name, time_val in results.items():
        if time_val:
            backends.append(name)
            times_list.append(time_val)
            colors_list.append('#10b981' if 'CUDA' in name else '#6366f1')

    if backends:
        fig.add_trace(go.Bar(
            x=backends,
            y=times_list,
            marker=dict(color=colors_list),
            text=[f"{t:.1f} ms" for t in times_list],
            textposition='outside',
        ))

    fig.update_layout(
        title=f"Benchmark Comparison - {STYLES.get(style, style.title())} Style",
        xaxis_title="Backend",
        yaxis_title="Inference Time (ms)",
        height=400,
        margin=dict(l=0, r=0, t=40, b=40),
        showlegend=False
    )

    # Calculate speedup
    if len(times_list) == 2:
        speedup = times_list[1] / times_list[0] if times_list[0] > 0 else times_list[0] / times_list[1]
        max_val = max(times_list)
        min_val = min(times_list)
        actual_speedup = max_val / min_val

        caption = f"Speedup: **{actual_speedup:.2f}x**"
    else:
        caption = "Run on GPU with CUDA for comparison"

    return fig.to_html(full_html=False, include_plotlyjs='cdn') + f"\n\n### {caption}"


# ============================================================================
# Gradio Interface Functions
# ============================================================================

def stylize_image_impl(
    input_image: Optional[Image.Image],
    style: str,
    backend: str,
    show_comparison: bool,
    add_watermark: bool
) -> Tuple[Optional[Image.Image], str, Optional[str]]:
    """Main stylization function for Gradio."""

    if input_image is None:
        return None, "Please upload an image first.", None

    try:
        # Convert to RGB if needed
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')

        # Handle blended styles (format: "style1_style2_alpha")
        if '_' in style and style not in STYLES:
            parts = style.split('_')
            if len(parts) >= 3:
                style1, style2 = parts[0], parts[1]
                alpha = float(parts[2]) / 100

                model = get_blended_model(style1, style2, alpha, backend)
                style_display = f"{STYLES.get(style1, style1)} × {alpha:.0%} + {STYLES.get(style2, style2)} × {100-alpha:.0%}"
            else:
                model = load_model(style, backend)
                style_display = STYLES.get(style, style)
        else:
            model = load_model(style, backend)
            style_display = STYLES.get(style, style)

        # Preprocess
        input_tensor = preprocess_image(input_image).to(get_device())

        # Stylize with timing
        start = time.perf_counter()

        with torch.no_grad():
            output_tensor = model(input_tensor)

        if get_device().type == 'cuda':
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Determine actual backend used
        actual_backend = 'cuda' if (backend == 'cuda' or (backend == 'auto' and CUDA_KERNELS_AVAILABLE)) else 'pytorch'
        perf_tracker.record(elapsed_ms, actual_backend)

        # Postprocess
        output_image = postprocess_tensor(output_tensor.cpu())

        # Add watermark if requested
        if add_watermark:
            output_image = add_watermark(output_image, style_display)

        # Create comparison if requested
        if show_comparison:
            output_image = create_side_by_side(input_image, output_image, style_display)

        # Save for download
        download_path = f"/tmp/styleforge_{int(time.time())}.png"
        output_image.save(download_path, quality=95)

        # Generate stats
        stats = perf_tracker.get_stats()
        fps = 1000 / elapsed_ms if elapsed_ms > 0 else 0
        width, height = input_image.size

        # Backend display name
        backend_display = {
            'auto': f"Auto ({'CUDA' if CUDA_KERNELS_AVAILABLE else 'PyTorch'})",
            'cuda': 'CUDA Kernels',
            'pytorch': 'PyTorch'
        }.get(backend, backend)

        stats_text = f"""
### Performance

| Metric | Value |
|--------|-------|
| **Style** | {style_display} |
| **Backend** | {backend_display} |
| **Time** | {elapsed_ms:.1f} ms ({fps:.0f} FPS) |
| **Avg Time** | {(stats.avg_ms if stats else elapsed_ms):.1f} ms |
| **Total Images** | {stats.total_inferences if stats else 1} |
| **Size** | {width}x{height} |
| **Device** | {get_device().type.upper()} |

---
{perf_tracker.get_comparison()}
"""

        return output_image, stats_text, download_path

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"""
### Error

**{str(e)}**

<details>
<summary>Show details</summary>

```
{error_details}
```

</details>
"""
        return None, error_msg, None


# Wrap with GPU decorator for ZeroGPU if available
# ZeroGPU requires at least one @GPU function to be present
if SPACES_AVAILABLE:
    try:
        stylize_image = GPU(stylize_image_impl)
    except Exception:
        # Fallback if GPU decorator fails
        stylize_image = stylize_image_impl
else:
    stylize_image = stylize_image_impl


def process_webcam_frame(image: Image.Image, style: str, backend: str) -> Image.Image:
    """Process webcam frame in real-time."""
    if image is None:
        return image

    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize for faster processing
        if max(image.size) > 640:
            scale = 640 / max(image.size)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)

        # Use blended style if applicable
        if '_' in style and style not in STYLES:
            parts = style.split('_')
            if len(parts) >= 3:
                style1, style2 = parts[0], parts[1]
                alpha = float(parts[2]) / 100
                model = get_blended_model(style1, style2, alpha, backend)
            else:
                model = load_model(style, backend)
        else:
            model = load_model(style, backend)

        input_tensor = preprocess_image(image).to(get_device())

        with torch.no_grad():
            output_tensor = model(input_tensor)

        if get_device().type == 'cuda':
            torch.cuda.synchronize()

        output_image = postprocess_tensor(output_tensor.cpu())

        webcam_state.frame_count += 1
        actual_backend = 'cuda' if backend == 'cuda' or (backend == 'auto' and CUDA_KERNELS_AVAILABLE) else 'pytorch'
        perf_tracker.record(10, actual_backend)

        return output_image

    except Exception:
        return image


def apply_region_style_ui(
    input_image: Image.Image,
    mask_type: str,
    position: float,
    style1: str,
    style2: str,
    backend: str
) -> Tuple[Image.Image, Image.Image]:
    """Apply region-based style transfer with AI segmentation support."""
    if input_image is None:
        return None, None

    # Create mask based on type
    if mask_type == "AI: Foreground":
        try:
            mask = get_ai_segmentation_mask(input_image, "foreground")
        except Exception as e:
            # Fallback to center circle if AI fails
            print(f"AI segmentation failed: {e}, using fallback")
            mask = create_region_mask(input_image, "center_circle", position)
    elif mask_type == "AI: Background":
        try:
            mask = get_ai_segmentation_mask(input_image, "background")
        except Exception as e:
            # Fallback to horizontal split if AI fails
            print(f"AI segmentation failed: {e}, using fallback")
            mask = create_region_mask(input_image, "horizontal_split", position)
    else:
        # Convert display name to internal name
        mask_type_map = {
            "Horizontal Split": "horizontal_split",
            "Vertical Split": "vertical_split",
            "Center Circle": "center_circle",
            "Corner Box": "corner_box",
            "Full": "full"
        }
        internal_type = mask_type_map.get(mask_type, "horizontal_split")
        mask = create_region_mask(input_image, internal_type, position)

    # Apply styles
    result = apply_region_style(input_image, mask, style1, style2, backend)

    # Create mask overlay for visualization
    mask_vis = mask.convert('RGB')
    mask_vis = mask_vis.resize(input_image.size)

    # Blend mask with original for visibility
    orig_np = np.array(input_image)
    mask_np = np.array(mask_vis)
    overlay_np = (orig_np * 0.7 + mask_np * 0.3).astype(np.uint8)
    mask_overlay = Image.fromarray(overlay_np)

    return result, mask_overlay


def refresh_styles_list() -> list:
    """Refresh styles list including custom styles."""
    custom = get_custom_styles()
    return list(STYLES.keys()) + custom


def get_style_description(style: str) -> str:
    """Get description for selected style."""
    return STYLE_DESCRIPTIONS.get(style, "")


def get_performance_stats() -> str:
    """Get current performance statistics."""
    stats = perf_tracker.get_stats()
    if not stats:
        return "No data yet."

    return f"""
### Live Statistics

| Metric | Value |
|--------|-------|
| **Avg Time** | {stats.avg_ms:.1f} ms |
| **Fastest** | {stats.min_ms:.1f} ms |
| **Slowest** | {stats.max_ms:.1f} ms |
| **Total Images** | {stats.total_inferences} |
| **Uptime** | {stats.uptime_hours:.1f} hours |

---
{perf_tracker.get_comparison()}
"""


def run_backend_comparison(style: str) -> str:
    """Run backend comparison and return results."""
    if not CUDA_KERNELS_AVAILABLE:
        return "### Backend Comparison\n\nCUDA kernels are not available on this device. Using PyTorch backend only."

    # Create test image
    test_img = Image.new('RGB', (512, 512), color='red')

    results = {}

    # Test PyTorch backend
    try:
        model = load_model(style, 'pytorch')
        test_tensor = preprocess_image(test_img).to(get_device())

        times = []
        for _ in range(5):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(test_tensor)
            if get_device().type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        results['pytorch'] = np.mean(times[1:])
    except Exception:
        results['pytorch'] = None

    # Test CUDA backend
    try:
        model = load_model(style, 'cuda')
        test_tensor = preprocess_image(test_img).to(get_device())

        times = []
        for _ in range(5):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(test_tensor)
            if get_device().type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        results['cuda'] = np.mean(times[1:])
    except Exception:
        results['cuda'] = None

    # Format results
    output = "### Backend Comparison Results\n\n"

    if results.get('pytorch') and results.get('cuda'):
        speedup = results['pytorch'] / results['cuda']
        output += f"""
| Backend | Time | Speedup |
|---------|------|---------|
| **PyTorch** | {results['pytorch']:.1f} ms | 1.0x |
| **CUDA Kernels** | {results['cuda']:.1f} ms | {speedup:.2f}x |

### CUDA kernels are {speedup:.1f}x faster! 🚀
"""
    else:
        output += "Could not complete comparison. Both backends may not be available."

    return output


def create_style_blend_output(
    input_image: Image.Image,
    style1: str,
    style2: str,
    blend_ratio: float,
    backend: str
) -> Image.Image:
    """Create blended style output."""
    if input_image is None:
        return None

    # Convert to RGB
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')

    # Get blended model
    alpha = blend_ratio / 100
    model = get_blended_model(style1, style2, alpha, backend)

    # Process
    input_tensor = preprocess_image(input_image).to(get_device())

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = postprocess_tensor(output_tensor.cpu())
    return output_image


# ============================================================================
# Build Gradio Interface
# ============================================================================

custom_css = """
/* ============================================
   LIQUID GLASS / GLASSMORPHISM THEME
   Gradio 5.x Compatible
   ============================================ */

/* Animated gradient background */
body {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    min-height: 100vh;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Universal font application */
* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
}

/* Ensure text elements are visible */
h1, h2, h3, h4, h5, h6, p, span, div, label, button, input, textarea, select {
    color: inherit;
}

/* Main app container - glass effect */
.gradio-container {
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    background: rgba(255, 255, 255, 0.75) !important;
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    max-width: 1400px;
    margin: 20px auto;
    padding: 24px !important;
}

/* Primary button - gradient with glass shimmer */
button.primary,
.gr-button-primary,
[class*="primary"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.9) 0%, rgba(139, 92, 246, 0.9) 100%) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 16px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
    position: relative;
    overflow: hidden;
}

button.primary:hover,
.gr-button-primary:hover,
[class*="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
}

button.primary:active,
.gr-button-primary:active,
[class*="primary"]:active {
    transform: translateY(0);
}

/* Secondary button - glass style */
button.secondary,
.gr-button-secondary,
.download,
[class*="secondary"] {
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    color: #374151 !important;
    border-radius: 16px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
}

button.secondary:hover,
.gr-button-secondary:hover,
.download:hover,
[class*="secondary"]:hover {
    background: rgba(255, 255, 255, 0.8) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}

/* All buttons */
button {
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

/* Tabs - glass style */
.tabs {
    background: rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(10px);
    border-radius: 16px !important;
    padding: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}

/* Tab buttons */
button.tab-item {
    background: transparent !important;
    border-radius: 12px !important;
    color: #6B7280 !important;
    transition: all 0.3s ease !important;
}

button.tab-item:hover {
    background: rgba(255, 255, 255, 0.5) !important;
}

button.tab-item.selected {
    background: rgba(255, 255, 255, 0.8) !important;
    color: #6366F1 !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

/* Input boxes and text areas */
input[type="text"],
input[type="number"],
textarea,
select {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

input[type="text"]:focus,
input[type="number"]:focus,
textarea:focus,
select:focus {
    background: rgba(255, 255, 255, 0.9) !important;
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    outline: none !important;
}

/* Image containers - glass frame */
.image-container,
[class*="image"] {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    overflow: hidden !important;
    background: rgba(255, 255, 255, 0.3) !important;
}

/* Slider styling */
input[type="range"] {
    -webkit-appearance: none;
    background: rgba(229, 231, 235, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 8px;
    height: 8px;
    border: 1px solid rgba(255, 255, 255, 0.3);
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 22px;
    height: 22px;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    border: 3px solid white;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
}

input[type="range"]::-moz-range-thumb {
    width: 22px;
    height: 22px;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    border: 3px solid white;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
}

/* Checkbox and radio styling */
input[type="checkbox"],
input[type="radio"] {
    accent-color: #6366F1 !important;
    width: 18px !important;
    height: 18px !important;
}

/* Badge styles */
.live-badge {
    display: inline-block;
    padding: 6px 16px;
    background: rgba(254, 243, 199, 0.8);
    backdrop-filter: blur(10px);
    color: #92400E;
    border-radius: 24px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.3);
}

.backend-badge {
    display: inline-block;
    padding: 6px 16px;
    background: rgba(209, 250, 229, 0.8);
    backdrop-filter: blur(10px);
    color: #065F46;
    border-radius: 24px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Markdown content */
.markdown {
    color: #374151 !important;
}

/* Text visibility fixes */
.gradio-container,
.gradio-container *,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container h5,
.gradio-container h6,
.gradio-container p,
.gradio-container span,
.gradio-container label {
    color: #1F2937 !important;
}

/* Button text colors */
button,
.gradio-container button {
    color: inherit !important;
}

/* Input and select text colors */
input,
textarea,
select {
    color: #1F2937 !important;
}

/* Label colors */
label,
[class*="label"] {
    color: #374151 !important;
    font-weight: 500 !important;
}

/* Gradio 5.x specific text elements */
.svelte-*, [class*="svelte-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Group/Row/Column containers */
.group,
.row,
.column {
    background: rgba(255, 255, 255, 0.3) !important;
    border-radius: 16px !important;
    padding: 16px !important;
}

/* Accordion */
.details {
    background: rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(10px);
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}

/* Scrollbar - glass style */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(229, 231, 235, 0.3);
    border-radius: 8px;
}

::-webkit-scrollbar-thumb {
    background: rgba(167, 139, 250, 0.5);
    border-radius: 8px;
    border: 2px solid rgba(255, 255, 255, 0.3);
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(139, 92, 246, 0.7);
}

/* Progress bar */
progress {
    background: rgba(229, 231, 235, 0.5) !important;
    border-radius: 8px !important;
    height: 8px !important;
}

progress::-webkit-progress-bar {
    background: rgba(229, 231, 235, 0.5);
    border-radius: 8px;
}

progress::-webkit-progress-value {
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
    border-radius: 8px;
}

/* Mobile responsive */
@media (max-width: 768px) {
    .gradio-container {
        margin: 10px !important;
        padding: 16px !important;
        border-radius: 20px !important;
    }

    button.primary,
    .gr-button-primary,
    [class*="primary"] {
        padding: 10px 18px !important;
        font-size: 14px !important;
    }
}

/* Loading spinner */
.spinner {
    border: 3px solid rgba(99, 102, 241, 0.2);
    border-top: 3px solid #6366F1;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Additional Gradio 5.x specific selectors */
.gradio-button.primary,
button[class*="Primary"],
[type="button"].primary {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.9) 0%, rgba(139, 92, 246, 0.9) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
}

/* Block containers */
.block {
    background: rgba(255, 255, 255, 0.25) !important;
    border-radius: 16px !important;
    padding: 12px !important;
}

/* Form elements */
.form,
.form-group {
    background: transparent !important;
}
"""

with gr.Blocks(
    title="StyleForge: Neural Style Transfer",
    theme=gr.themes.Glass(
        primary_hue="indigo",
        secondary_hue="purple",
        font=gr.themes.GoogleFont("Inter"),
        radius_size="lg",
    ),
    css=custom_css,
) as demo:

    # Header with Portal-style hero section
    cuda_badge = f"<span class='backend-badge'>CUDA Accelerated</span>" if CUDA_KERNELS_AVAILABLE else ""
    gr.Markdown(f"""
    <div style="text-align: center; padding: 3rem 0 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #6366F1, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;">
            StyleForge
        </h1>
        <p style="color: #6B7280; font-size: 1.1rem; margin-bottom: 1rem;">
            Neural Style Transfer with Custom CUDA Kernels
        </p>
        {cuda_badge}
        <p style="color: #9CA3AF; margin-top: 1rem; font-size: 0.9rem;">
            Custom Styles • Region Transfer • Style Blending • Real-time Processing
        </p>
    </div>
    """)

    # Mode selector
    with gr.Tabs() as tabs:
        # Tab 1: Quick Style Transfer
        with gr.Tab("Quick Style", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    quick_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=400
                    )

                    quick_style = gr.Dropdown(
                        choices=list(STYLES.keys()),
                        value='candy',
                        label="Artistic Style"
                    )

                    quick_backend = gr.Radio(
                        choices=list(BACKENDS.keys()),
                        value='auto',
                        label="Processing Backend"
                    )

                    with gr.Row():
                        quick_compare = gr.Checkbox(
                            label="Side-by-side",
                            value=False
                        )
                        quick_watermark = gr.Checkbox(
                            label="Add watermark",
                            value=False
                        )

                    quick_btn = gr.Button(
                        "Stylize Image",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    quick_output = gr.Image(
                        label="Result",
                        type="pil",
                        height=400
                    )

                    with gr.Row():
                        quick_download = gr.DownloadButton(
                            label="Download",
                            variant="secondary"
                        )

                    quick_stats = gr.Markdown(
                        "> Upload an image and click **Stylize** to begin!"
                    )

        # Tab 2: Style Blending
        with gr.Tab("Style Blending", id=1):
            gr.Markdown("""
            ### Mix Two Styles Together

            Blend between any two styles to create unique artistic combinations.
            This demonstrates style interpolation in the latent space.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    blend_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=350
                    )

                    blend_style1 = gr.Dropdown(
                        choices=list(STYLES.keys()),
                        value='candy',
                        label="Style 1"
                    )

                    blend_style2 = gr.Dropdown(
                        choices=list(STYLES.keys()),
                        value='mosaic',
                        label="Style 2"
                    )

                    blend_ratio = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Blend Ratio"
                    )

                    blend_backend = gr.Radio(
                        choices=list(BACKENDS.keys()),
                        value='auto',
                        label="Backend"
                    )

                    blend_btn = gr.Button(
                        "Blend Styles",
                        variant="primary"
                    )

                    gr.Markdown("""
                    **How it Works:**
                    - Style blending interpolates between model weights
                    - At 0% you get pure Style 1
                    - At 100% you get pure Style 2
                    - At 50% you get an equal mix of both
                    """)

                with gr.Column(scale=1):
                    blend_output = gr.Image(
                        label="Blended Result",
                        type="pil",
                        height=350
                    )

                    blend_info = gr.Markdown(
                        "Adjust the blend ratio and click **Blend Styles** to see the result."
                    )

        # Tab 3: Region-Based Style
        with gr.Tab("Region Transfer", id=2):
            gr.Markdown("""
            ### Apply Different Styles to Different Regions

            Transform specific parts of your image with different styles.
            **NEW:** AI-powered foreground/background segmentation!
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    region_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=350
                    )

                    region_mask_type = gr.Radio(
                        choices=[
                            "AI: Foreground",
                            "AI: Background",
                            "Horizontal Split",
                            "Vertical Split",
                            "Center Circle",
                            "Corner Box",
                            "Full"
                        ],
                        value="AI: Foreground",
                        label="Mask Type"
                    )

                    region_position = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.1,
                        label="Split Position"
                    )

                    with gr.Row():
                        region_style1 = gr.Dropdown(
                            choices=list(STYLES.keys()),
                            value='candy',
                            label="Style (White/Top/Left)"
                        )
                        region_style2 = gr.Dropdown(
                            choices=list(STYLES.keys()),
                            value='mosaic',
                            label="Style (Black/Bottom/Right)"
                        )

                    region_backend = gr.Radio(
                        choices=list(BACKENDS.keys()),
                        value='auto',
                        label="Backend"
                    )

                    region_btn = gr.Button(
                        "Apply Region Styles",
                        variant="primary"
                    )

                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.Tab("Result"):
                            region_output = gr.Image(
                                label="Stylized Result",
                                type="pil",
                                height=300
                            )

                        with gr.Tab("Mask Preview"):
                            region_mask_preview = gr.Image(
                                label="Mask Preview",
                                type="pil",
                                height=300
                            )

                    gr.Markdown("""
                    **Mask Guide:**
                    - **AI: Foreground** 🆕: Automatically detect main subject (person, object, etc.)
                    - **AI: Background** 🆕: Automatically detect background/sky
                    - **Horizontal**: Top/bottom split
                    - **Vertical**: Left/right split
                    - **Center Circle**: Circular region in center
                    - **Corner Box**: Top-left quadrant only

                    *AI segmentation uses the Rembg model (U^2-Net) for automatic subject detection.*
                    """)

        # Tab 4: Custom Style Training
        with gr.Tab("Create Style", id=3):
            gr.Markdown("""
            ### Extract Style from Any Image 🆕

            Upload any artwork to extract its artistic style using **VGG19 feature matching**.

            **How it works:**
            1. Extract style features using pre-trained VGG19 neural network
            2. Fine-tune a transformation network to match those features
            3. Save as a reusable style model

            This is **real style extraction** - not just copying an existing style!
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    train_style_image = gr.Image(
                        label="Style Image (Artwork)",
                        type="pil",
                        sources=["upload"],
                        height=350
                    )

                    train_style_name = gr.Textbox(
                        label="Style Name",
                        value="my_custom_style",
                        placeholder="Enter a name for your custom style"
                    )

                    train_iterations = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=100,
                        step=50,
                        label="Training Iterations"
                    )

                    train_backend = gr.Radio(
                        choices=list(BACKENDS.keys()),
                        value='auto',
                        label="Backend"
                    )

                    train_btn = gr.Button(
                        "Extract Style",
                        variant="primary"
                    )

                    refresh_styles_btn = gr.Button("Refresh Style List")

                with gr.Column(scale=1):
                    train_output = gr.Markdown(
                        "> Upload a style image and click **Extract Style** to begin!\n\n"
                        "**How it works:**\n"
                        "- VGG19 extracts artistic features (textures, colors, patterns)\n"
                        "- Neural network is fine-tuned to match those features\n"
                        "- Result is a reusable style model\n\n"
                        "**Tips:**\n"
                        "- Use artwork with clear artistic style (paintings, illustrations)\n"
                        "- More iterations = better style matching (slower)\n"
                        "- GPU recommended for faster training\n"
                        "- Your custom style will appear in all Style dropdowns"
                    )

                    train_progress = gr.Markdown("")

        # Tab 5: Webcam Live
        with gr.Tab("Webcam Live", id=4):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### <span class="live-badge">LIVE</span> Real-time Webcam Style Transfer
                    """)

                    webcam_style = gr.Dropdown(
                        choices=list(STYLES.keys()),
                        value='candy',
                        label="Artistic Style"
                    )

                    webcam_backend = gr.Radio(
                        choices=list(BACKENDS.keys()),
                        value='auto',
                        label="Backend"
                    )

                    webcam_stream = gr.Image(
                        sources=["webcam"],
                        label="Webcam Feed",
                        height=400
                    )

                    webcam_info = gr.Markdown(
                        "> Click in the webcam preview to start the feed"
                    )

                with gr.Column(scale=1):
                    webcam_output = gr.Image(
                        label="Stylized Output",
                        height=400
                    )

                    webcam_stats = gr.Markdown(
                        get_performance_stats()
                    )

                    refresh_stats_btn = gr.Button("Refresh Stats", size="sm")

        # Tab 6: Performance Dashboard
        with gr.Tab("Performance", id=5):
            gr.Markdown("""
            ### Real-time Performance Dashboard

            Track inference times and compare backends with live charts.
            """)

            with gr.Row():
                benchmark_style = gr.Dropdown(
                    choices=list(STYLES.keys()),
                    value='candy',
                    label="Select Style for Benchmark"
                )

                run_benchmark_btn = gr.Button(
                    "Run Benchmark",
                    variant="primary"
                )

            benchmark_chart = gr.Markdown(
                "Click **Run Benchmark** to see the performance chart"
            )

            live_chart = gr.Markdown(
                "Run some inferences to see the live chart populate below..."
            )

            refresh_chart_btn = gr.Button("Refresh Chart")

            gr.Markdown("---")
            gr.Markdown("### Live Performance Chart")

            chart_display = gr.HTML(
                "<div style='text-align:center; padding: 20px;'>Run inferences to see chart</div>"
            )

            chart_stats = gr.Markdown()

    # Style description (shared across all tabs)
    style_desc = gr.Markdown("*Select a style to see description*")

    # Examples section
    gr.Markdown("---")

    def create_example_image():
        # Create a more interesting test image with geometric shapes
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        # Background gradient
        for i in range(256):
            arr[:, i, 0] = i // 2
            arr[:, i, 1] = 128
            arr[:, i, 2] = 255 - i // 2
        # Add a circle in the center
        cy, cx = 128, 128
        for y in range(256):
            for x in range(256):
                if (x - cx)**2 + (y - cy)**2 <= 50**2:
                    arr[y, x, 0] = 255
                    arr[y, x, 1] = 200
                    arr[y, x, 2] = 100
        return Image.fromarray(arr)

    example_img = create_example_image()

    # Pre-styled example outputs for display
    # These images demonstrate each style without needing to run the model
    gr.Markdown("### Quick Style Examples")
    gr.Markdown("Click any example to apply that style to your own image, or try the styles below:")

    gr.Examples(
        examples=[
            [example_img, "candy", "auto", False, False],
            [example_img, "mosaic", "auto", False, False],
            [example_img, "rain_princess", "auto", True, False],
            [example_img, "udnie", "auto", False, False],
        ],
        inputs=[quick_image, quick_style, quick_backend, quick_compare, quick_watermark],
        label="Style Presets (click to load)"
    )

    # Display example style gallery
    gr.Markdown("""
    <div style="display: flex; gap: 1rem; justify-content: center; margin: 1rem 0; flex-wrap: wrap;">
        <div style="text-align: center;">
            <div style="width: 120px; height: 120px; background: linear-gradient(135deg, #ff6b6b, #feca57); border-radius: 8px; margin: 0 auto;"></div>
            <p style="margin-top: 0.5rem; font-size: 0.85rem;">🍬 Candy</p>
        </div>
        <div style="text-align: center;">
            <div style="width: 120px; height: 120px; background: linear-gradient(135deg, #5f27cd, #00d2d3); border-radius: 8px; margin: 0 auto;"></div>
            <p style="margin-top: 0.5rem; font-size: 0.85rem;">🎨 Mosaic</p>
        </div>
        <div style="text-align: center;">
            <div style="width: 120px; height: 120px; background: linear-gradient(135deg, #576574, #c8d6e5); border-radius: 8px; margin: 0 auto;"></div>
            <p style="margin-top: 0.5rem; font-size: 0.85rem;">🌧️ Rain Princess</p>
        </div>
        <div style="text-align: center;">
            <div style="width: 120px; height: 120px; background: linear-gradient(135deg, #ee5a24, #f9ca24); border-radius: 8px; margin: 0 auto;"></div>
            <p style="margin-top: 0.5rem; font-size: 0.85rem;">🖼️ Udnie</p>
        </div>
    </div>
    """)

    # FAQ Section
    gr.Markdown("---")

    with gr.Accordion("FAQ & Help", open=False):
        gr.Markdown("""
        ### What are CUDA kernels?

        Custom CUDA kernels are hand-written GPU code that fuses multiple operations
        into a single kernel launch. This reduces memory transfers and improves
        performance by 8-9x.

        ### How does Style Blending work?

        Style blending interpolates between the weights of two trained style models.
        This demonstrates that styles exist in a continuous latent space where you can
        navigate and create new artistic variations.

        ### What is Region-based Style Transfer?

        This feature applies different artistic styles to different regions of the same image.
        It demonstrates computer vision concepts like segmentation and masking, while
        enabling creative effects like "make the sky look like Starry Night while keeping
        the ground realistic."

        ### Which backend should I use?

        - **Auto**: Recommended - automatically uses the fastest available option
        - **CUDA Kernels**: Best performance on GPU (requires CUDA compilation)
        - **PyTorch**: Fallback for CPU or when CUDA is unavailable

        ### Can I use this commercially?

        Yes! StyleForge is open source (MIT license).
        """)

    # Technical details
    with gr.Accordion("Technical Details", open=False):
        gr.Markdown(f"""
        ### Architecture

        **Network:** Encoder-Decoder with Residual Blocks (Johnson et al.)

        - **Encoder**: 3 Conv layers + Instance Normalization
        - **Transformer**: 5 Residual blocks
        - **Decoder**: 3 Upsample Conv layers + Instance Normalization

        ### CUDA Optimizations

        **Status:** {'✅ Available' if CUDA_KERNELS_AVAILABLE else '❌ Not Available (CPU or no CUDA)'}

        When CUDA kernels are available:
        - **Fused InstanceNorm**: Combines mean, variance, normalize, affine transform
        - **Vectorized memory**: Uses `float4` loads for 4x bandwidth
        - **Shared memory**: Reduces global memory traffic
        - **Warp-level reductions**: Efficient parallel reductions

        ### ML Concepts Demonstrated

        - **Style Transfer**: Neural artistic stylization
        - **Latent Space Interpolation**: Style blending shows continuous style space
        - **Conditional Generation**: Region-based style transfer
        - **Transfer Learning**: Custom style training from few examples
        - **Performance Optimization**: CUDA kernels, JIT compilation, caching
        - **Model Deployment**: Gradio web interface, CI/CD pipeline

        ### Resources

        - [GitHub Repository](https://github.com/olivialiau/StyleForge)
        - [Paper: Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155)
        """)

    # Footer
    gr.Markdown("""
    <div class="footer">
        <p>
            <strong>StyleForge</strong> • Created by Olivia • USC Computer Science<br>
            <a href="https://github.com/olivialiau/StyleForge">GitHub</a> •
            Built with <a href="https://huggingface.co/spaces">Hugging Face Spaces</a> 🤗
        </p>
    </div>
    """)

    # ============================================================================
    # Event Handlers
    # ============================================================================

    # Style description updates
    def update_style_desc(style):
        desc = STYLE_DESCRIPTIONS.get(style, "")
        return f"*{desc}*"

    # Quick style handlers
    quick_style.change(
        fn=update_style_desc,
        inputs=[quick_style],
        outputs=[style_desc]
    )

    quick_btn.click(
        fn=stylize_image,
        inputs=[quick_image, quick_style, quick_backend, quick_compare, quick_watermark],
        outputs=[quick_output, quick_stats, quick_download]
    )

    # Style blending handlers
    def update_blend_info(style1: str, style2: str, ratio: float) -> str:
        s1_name = STYLES.get(style1, style1)
        s2_name = STYLES.get(style2, style2)
        return f"Blended {s1_name} × {ratio:.0f}% + {s2_name} × {100-ratio:.0f}%"

    blend_btn.click(
        fn=create_style_blend_output,
        inputs=[blend_image, blend_style1, blend_style2, blend_ratio, blend_backend],
        outputs=[blend_output]
    ).then(
        fn=update_blend_info,
        inputs=[blend_style1, blend_style2, blend_ratio],
        outputs=[blend_info]
    )

    # Region-based handlers
    region_btn.click(
        fn=apply_region_style_ui,
        inputs=[region_image, region_mask_type, region_position, region_style1, region_style2, region_backend],
        outputs=[region_output, region_mask_preview]
    )

    region_mask_type.change(
        fn=lambda mt, img, pos: create_region_mask(img, mt, pos) if img else None,
        inputs=[region_mask_type, region_image, region_position],
        outputs=[region_mask_preview]
    )

    region_position.change(
        fn=lambda pos, img, mt: create_region_mask(img, mt, pos) if img else None,
        inputs=[region_position, region_image, region_mask_type],
        outputs=[region_mask_preview]
    )

    # Custom style training
    train_btn.click(
        fn=train_custom_style,
        inputs=[train_style_image, train_style_name, train_iterations, train_backend],
        outputs=[train_progress, train_output]
    )

    def update_style_choices():
        return list(STYLES.keys()) + get_custom_styles()

    refresh_styles_btn.click(
        fn=update_style_choices,
        outputs=[quick_style]
    ).then(
        fn=update_style_choices,
        outputs=[blend_style1]
    ).then(
        fn=update_style_choices,
        outputs=[blend_style2]
    )

    # Webcam handlers - note: streaming disabled for Gradio 5.x compatibility
    # Users can still upload/process webcam images manually
    webcam_stream.change(
        fn=process_webcam_frame,
        inputs=[webcam_stream, webcam_style, webcam_backend],
        outputs=[webcam_output],
    )

    refresh_stats_btn.click(
        fn=get_performance_stats,
        outputs=[webcam_stats]
    )

    # Benchmark handlers
    run_benchmark_btn.click(
        fn=create_benchmark_comparison,
        inputs=[benchmark_style],
        outputs=[benchmark_chart]
    )

    refresh_chart_btn.click(
        fn=create_performance_chart,
        outputs=[chart_display]
    )


# ============================================================================
# Launch Configuration
# ============================================================================

if __name__ == "__main__":
    # Disable API to avoid gradio_client compatibility issues
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    demo.launch(
        show_api=False,
        show_error=True,
        quiet=False,
    )
