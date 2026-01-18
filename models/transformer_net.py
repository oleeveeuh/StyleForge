"""
Fast Neural Style Transfer Transformer Network

Based on Johnson et al. "Perceptual Losses for Real-Time Style Transfer"
https://arxiv.org/abs/1603.08155

Architecture:
    Input (3, H, W)
        ↓
    Encoder: 3 conv layers with InstanceNorm
        ↓
    Residual blocks: 5-10 residual blocks
        ↓
    Decoder: 3 upsampling conv layers with InstanceNorm
        ↓
    Output (3, H, W)

This network is designed to be trained for each specific style.
Pre-trained weights can be loaded from the fast-neural-style repository.

CUDA Kernels:
    When CUDA is available, uses FusedInstanceNorm2d for 3-5x speedup.
    Falls back to nn.InstanceNorm2d on CPU/MPS.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

# Try to import CUDA kernels for accelerated InstanceNorm
# Only available on CUDA devices (not MPS/CPU)
try:
    import torch
    if torch.cuda.is_available():
        from kernels.instance_norm_wrapper import FusedInstanceNorm2d
        from kernels.conv_fusion_wrapper import FusedConvInstanceNormReLU
        CUDA_KERNELS_AVAILABLE = True
    else:
        CUDA_KERNELS_AVAILABLE = False
        FusedInstanceNorm2d = None
        FusedConvInstanceNormReLU = None
except (ImportError, RuntimeError):
    CUDA_KERNELS_AVAILABLE = False
    FusedInstanceNorm2d = None
    FusedConvInstanceNormReLU = None


class ConvLayer(nn.Module):
    """Convolution -> InstanceNorm -> ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        relu: bool = True,
        norm: bool = True,
    ):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if norm:
            # Use CUDA kernel if available, otherwise PyTorch InstanceNorm
            # track_running_stats=True to match pre-trained checkpoints
            if CUDA_KERNELS_AVAILABLE and FusedInstanceNorm2d is not None:
                self.norm = FusedInstanceNorm2d(out_channels, affine=True)
                self._use_cuda_norm = True
            else:
                self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
                self._use_cuda_norm = False
        else:
            self.norm = None
            self._use_cuda_norm = False

        if relu:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class ResidualBlock(nn.Module):
    """Residual block with two ConvLayers and skip connection"""

    def __init__(self, channels: int):
        super().__init__()
        # padding=1 = kernel_size // 2, maintains spatial dimensions for residual connection
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1, relu=False)

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
    ):
        super().__init__()

        if upsample > 1:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        else:
            self.upsample = None

        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Use CUDA kernel if available, otherwise PyTorch InstanceNorm
        if CUDA_KERNELS_AVAILABLE and FusedInstanceNorm2d is not None:
            self.norm = FusedInstanceNorm2d(out_channels, affine=True)
        else:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)

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
    """
    Fast Neural Style Transfer Network

    Args:
        num_residual_blocks: Number of residual blocks (default: 5)
            - Original paper uses 5 for faster training
            - Use 10 for better quality results
        checkpoint_path: Optional path to load pre-trained weights

    Example:
        >>> model = TransformerNet()
        >>> model.load_state_dict(torch.load('candy.pth'))
        >>> output = model(input_image)
    """

    def __init__(self, num_residual_blocks: int = 5):
        super().__init__()

        # Initial convolution layers (encoder)
        # Input: (3, H, W) -> (32, H, W)
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1, padding=4)
        # (32, H, W) -> (64, H/2, W/2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, padding=1)
        # (64, H/2, W/2) -> (128, H/4, W/4)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_residual_blocks)]
        )

        # Upsampling layers (decoder)
        # (128, H/4, W/4) -> (64, H/2, W/2)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, padding=1, upsample=2)
        # (64, H/2, W/2) -> (32, H, W)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, padding=1, upsample=2)
        # (32, H, W) -> (3, H, W)
        self.deconv3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, kernel_size=9, stride=1)
        )

    def forward(self, x):
        """
        Args:
            x: Input image tensor (B, 3, H, W) in range [-1, 1] or [0, 1]

        Returns:
            Stylized image (B, 3, H, W) in range [-1, 1]
        """
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

    def load_checkpoint(self, checkpoint_path: str, device: Optional[torch.device] = None) -> None:
        """
        Load pre-trained weights from checkpoint file.

        Handles multiple checkpoint formats:
        - PyTorch examples format (conv1, in1-3, res1-5, upsample_conv, etc.)
        - Original pytorch format with .net naming (conv1.norm, etc.)
        - DataParallel wrapped (module. prefix)

        Args:
            checkpoint_path: Path to .pth file containing state_dict
            device: Device to load weights onto (defaults to current device of model)

        Example:
            >>> model = TransformerNet()
            >>> model.load_checkpoint('models/pretrained/candy.pth')
        """
        if device is None:
            device = next(self.parameters()).device

        state_dict = torch.load(checkpoint_path, map_location=device)

        # Handle different state dict formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # Create mapping for different naming conventions
        mapped_state_dict = {}

        # Naming convention mapping from pytorch/examples to our structure
        name_mapping = {
            # Encoder
            "in1": "conv1.norm",      # InstanceNorm
            "in2": "conv2.norm",
            "in3": "conv3.norm",
            "conv1.conv2d": "conv1.conv",
            "conv2.conv2d": "conv2.conv",
            "conv3.conv2d": "conv3.conv",

            # Residual blocks
            "res1.conv1.conv2d": "residual_blocks.0.conv1.conv",
            "res1.in1": "residual_blocks.0.conv1.norm",
            "res1.conv2.conv2d": "residual_blocks.0.conv2.conv",
            "res1.in2": "residual_blocks.0.conv2.norm",

            "res2.conv1.conv2d": "residual_blocks.1.conv1.conv",
            "res2.in1": "residual_blocks.1.conv1.norm",
            "res2.conv2.conv2d": "residual_blocks.1.conv2.conv",
            "res2.in2": "residual_blocks.1.conv2.norm",

            "res3.conv1.conv2d": "residual_blocks.2.conv1.conv",
            "res3.in1": "residual_blocks.2.conv1.norm",
            "res3.conv2.conv2d": "residual_blocks.2.conv2.conv",
            "res3.in2": "residual_blocks.2.conv2.norm",

            "res4.conv1.conv2d": "residual_blocks.3.conv1.conv",
            "res4.in1": "residual_blocks.3.conv1.norm",
            "res4.conv2.conv2d": "residual_blocks.3.conv2.conv",
            "res4.in2": "residual_blocks.3.conv2.norm",

            "res5.conv1.conv2d": "residual_blocks.4.conv1.conv",
            "res5.in1": "residual_blocks.4.conv1.norm",
            "res5.conv2.conv2d": "residual_blocks.4.conv2.conv",
            "res5.in2": "residual_blocks.4.conv2.norm",

            # Decoder
            "deconv1.conv2d": "deconv1.conv",
            "in4": "deconv1.norm",
            "deconv2.conv2d": "deconv2.conv",
            "in5": "deconv2.norm",
            # deconv3 is a Sequential wrapper, need special handling
            "deconv3.conv2d": "deconv3.1",  # Maps to nn.Conv2d at index 1 in Sequential
        }

        for old_name, v in state_dict.items():
            # Remove 'module.' prefix if present (from DataParallel)
            name = old_name.replace('module.', '')

            # Try name mapping
            mapped = False
            for prefix, new_name in name_mapping.items():
                if name.startswith(prefix):
                    suffix = name[len(prefix):]
                    mapped_key = new_name + suffix
                    mapped_state_dict[mapped_key] = v
                    mapped = True
                    break

            if not mapped:
                # Use as-is if no mapping found
                mapped_state_dict[name] = v

        # Post-process: map .weight/.bias to .gamma/.beta for InstanceNorm parameters
        final_state_dict = {}
        for key, value in mapped_state_dict.items():
            # For InstanceNorm layers (keys ending in .norm), map weight/bias to gamma/beta
            if key.endswith('.norm.weight'):
                final_state_dict[key[:-6] + 'gamma'] = value
            elif key.endswith('.norm.bias'):
                # Remove '.bias' (5 chars) and add '.beta'
                final_state_dict[key[:-5] + '.beta'] = value
            else:
                final_state_dict[key] = value

        # Try loading with strict=False to see what's missing/unexpected
        try:
            missing_keys, unexpected_keys = self.load_state_dict(final_state_dict, strict=False)

            if missing_keys:
                print(f"⚠️  Missing keys (will use random init): {len(missing_keys)}")
                for key in list(missing_keys)[:5]:  # Show first 5
                    print(f"    - {key}")
                if len(missing_keys) > 5:
                    print(f"    ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"⚠️  Unexpected keys (will be ignored): {len(unexpected_keys)}")

        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            raise

        print(f"✅ Loaded checkpoint from {checkpoint_path}")

    def get_model_size(self) -> float:
        """Return model size in MB (FP32)"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / 1e6

    def get_parameter_count(self) -> tuple[int, int]:
        """Return (total_params, trainable_params)"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# =============================================================================
# Variant: Pure PyTorch Baseline (no CUDA kernels)
# =============================================================================

class ConvLayerBaseline(nn.Module):
    """Convolution -> InstanceNorm -> ReLU (Pure PyTorch, no CUDA kernels)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        relu: bool = True,
        norm: bool = True,
    ):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if norm:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        else:
            self.norm = None

        if relu:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class ResidualBlockBaseline(nn.Module):
    """Residual block with pure PyTorch layers"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvLayerBaseline(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayerBaseline(channels, channels, kernel_size=3, stride=1, padding=1, relu=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return residual + out


class UpsampleConvLayerBaseline(nn.Module):
    """Upsample (nearest neighbor) -> Conv -> InstanceNorm -> ReLU (Pure PyTorch)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        upsample: int = 2,
    ):
        super().__init__()

        if upsample > 1:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        else:
            self.upsample = None

        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
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


class TransformerNetBaseline(nn.Module):
    """
    Pure PyTorch baseline - No CUDA kernels

    This is the baseline for comparison. Uses standard PyTorch operations.
    """

    def __init__(self, num_residual_blocks: int = 5):
        super().__init__()

        self.conv1 = ConvLayerBaseline(3, 32, kernel_size=9, stride=1, padding=4)
        self.conv2 = ConvLayerBaseline(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvLayerBaseline(64, 128, kernel_size=3, stride=2, padding=1)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlockBaseline(128) for _ in range(num_residual_blocks)]
        )

        self.deconv1 = UpsampleConvLayerBaseline(128, 64, kernel_size=3, stride=1, padding=1, upsample=2)
        self.deconv2 = UpsampleConvLayerBaseline(64, 32, kernel_size=3, stride=1, padding=1, upsample=2)

        self.deconv3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, kernel_size=9, stride=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.residual_blocks(out)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        return out


# =============================================================================
# Variant: Fully Fused (Conv+IN+ReLU)
# =============================================================================

class FusedConvLayer(nn.Module):
    """Fused Convolution -> InstanceNorm -> ReLU in a single CUDA kernel"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        relu: bool = True,
    ):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)

        # Use FusedConvInstanceNormReLU if available
        if CUDA_KERNELS_AVAILABLE and FusedConvInstanceNormReLU is not None:
            self.fused = FusedConvInstanceNormReLU(
                in_channels, out_channels, kernel_size, stride
            )
            self._use_fused = True
        else:
            # Fall back to separate layers
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
            self.activation = nn.ReLU(inplace=True)
            self._use_fused = False

    def forward(self, x):
        out = self.pad(x)
        if self._use_fused:
            out = self.fused(out)
        else:
            out = self.conv(out)
            out = self.norm(out)
            out = self.activation(out)
        return out


class FusedResidualBlock(nn.Module):
    """Residual block with fused Conv+IN+ReLU layers"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = FusedConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = FusedConvLayer(channels, channels, kernel_size=3, stride=1, padding=1, relu=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return residual + out


class TransformerNetFused(nn.Module):
    """
    Fully fused variant using FusedConvInstanceNormReLU

    This version fuses Conv+InstanceNorm+ReLU into a single kernel
    for maximum performance.
    """

    def __init__(self, num_residual_blocks: int = 5):
        super().__init__()

        self.conv1 = FusedConvLayer(3, 32, kernel_size=9, stride=1, padding=4)
        self.conv2 = FusedConvLayer(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = FusedConvLayer(64, 128, kernel_size=3, stride=2, padding=1)

        self.residual_blocks = nn.Sequential(
            *[FusedResidualBlock(128) for _ in range(num_residual_blocks)]
        )

        self.deconv1 = FusedConvLayer(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = FusedConvLayer(64, 32, kernel_size=3, stride=1, padding=1)

        self.deconv3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, kernel_size=9, stride=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.residual_blocks(out)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        return out


# =============================================================================
# Variant Factory
# =============================================================================

def create_transformer_net(
    variant: Literal["auto", "baseline", "fused"] = "auto",
    num_residual_blocks: int = 5
) -> nn.Module:
    """
    Create a TransformerNet with the specified variant.

    Args:
        variant:
            - "auto": Use CUDA kernels if available (default TransformerNet)
            - "baseline": Pure PyTorch, no CUDA kernels
            - "fused": Fully fused Conv+IN+ReLU kernels
        num_residual_blocks: Number of residual blocks

    Returns:
        TransformerNet model
    """
    if variant == "auto":
        return TransformerNet(num_residual_blocks)
    elif variant == "baseline":
        return TransformerNetBaseline(num_residual_blocks)
    elif variant == "fused":
        return TransformerNetFused(num_residual_blocks)
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose from: auto, baseline, fused")


def get_available_variants() -> list[str]:
    """Return list of available variants based on CUDA availability"""
    variants = ["baseline"]
    if CUDA_KERNELS_AVAILABLE:
        variants.extend(["auto", "fused"])
    return variants


# Pre-trained style names from active community repositories
# Note: Original jcjohnson repository uses .t7 (Torch) format, not .pth (PyTorch)
AVAILABLE_STYLES = [
    "candy",         # Candy style - from yakhyo
    "mosaic",        # Mosaic style - from yakhyo
    "udnie",         # Udnie style - from yakhyo
    "rain_princess", # Rain Princess style - from yakhyo
    "starry",        # Starry Night style - from rrmina
    "wave",          # The Great Wave style - from rrmina
]

# Mapping of style names to their download URLs
STYLE_URLS = {
    "candy": "https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/candy.pth",
    "mosaic": "https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/mosaic.pth",
    "udnie": "https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/udnie.pth",
    "rain_princess": "https://github.com/yakhyo/fast-neural-style-transfer/releases/download/v1.0/rain-princess.pth",
    "starry": "https://raw.githubusercontent.com/rrmina/fast-neural-style-pytorch/master/transforms/starry.pth",
    "wave": "https://raw.githubusercontent.com/rrmina/fast-neural-style-pytorch/master/transforms/wave.pth",
}


def get_style_url(style_name: str) -> str:
    """Get download URL for pre-trained style weights"""
    if style_name not in STYLE_URLS:
        raise ValueError(f"Unknown style: {style_name}. Available: {', '.join(AVAILABLE_STYLES)}")
    return STYLE_URLS[style_name]
