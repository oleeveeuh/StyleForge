"""
StyleForge - Model Initialization

Exports main model classes for easy importing.
Optimized models require CUDA kernels and are imported conditionally.
"""

from .conv_blocks import ConvBlock, DeconvBlock, ResidualBlock
from .style_transfer_net import StyleTransferNetwork
from .transformer import TransformerBlock

# Try to import optimized models (require CUDA kernels)
try:
    from .optimized_transformer import OptimizedTransformerBlock
    from .optimized_blocks import OptimizedConvBlock, OptimizedDeconvBlock
    from .optimized_style_transfer_net import OptimizedStyleTransferNetwork, create_optimized_model
    _has_optimized = True
except ImportError:
    # CUDA kernels not available
    OptimizedTransformerBlock = None
    OptimizedConvBlock = None
    OptimizedDeconvBlock = None
    OptimizedStyleTransferNetwork = None
    create_optimized_model = None
    _has_optimized = False

__all__ = [
    "StyleTransferNetwork",
    "TransformerBlock",
    "ConvBlock",
    "DeconvBlock",
    "ResidualBlock",
]

if _has_optimized:
    __all__.extend([
        "OptimizedTransformerBlock",
        "OptimizedConvBlock",
        "OptimizedDeconvBlock",
        "OptimizedStyleTransferNetwork",
        "create_optimized_model",
    ])

__version__ = "0.1.0"
