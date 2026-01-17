"""StyleForge kernels module"""

from .conv_fusion_wrapper import (
    FusedConvInstanceNormReLU,
    ResidualBlock,
    benchmark_conv_fusion_vs_pytorch,
    run_comprehensive_benchmark as run_conv_fusion_benchmark,
)
from .instance_norm_wrapper import FusedInstanceNorm2d, benchmark_instance_norm_vs_pytorch
from .attention_v3_wrapper import FusedAttentionV3

__all__ = [
    "FusedConvInstanceNormReLU",
    "ResidualBlock",
    "benchmark_conv_fusion_vs_pytorch",
    "run_conv_fusion_benchmark",
    "FusedInstanceNorm2d",
    "benchmark_instance_norm_vs_pytorch",
    "FusedAttentionV3",
]

__version__ = "0.1.0"
