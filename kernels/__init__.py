"""StyleForge kernels module"""

from .attention_wrapper import FusedAttention, FusedQKVProjection, test_fused_attention
from .attention_v2_wrapper import FusedAttentionV2, benchmark_v2_vs_others
from .ffn_wrapper import FusedFFN, benchmark_ffn_vs_pytorch
from .instance_norm_wrapper import FusedInstanceNorm2d, benchmark_instance_norm_vs_pytorch
from .conv_fusion_wrapper import (
    FusedConvInstanceNormReLU,
    ResidualBlock,
    benchmark_conv_fusion_vs_pytorch,
    run_comprehensive_benchmark as run_conv_fusion_benchmark,
)

__all__ = [
    "FusedAttention",
    "FusedQKVProjection",
    "test_fused_attention",
    "FusedAttentionV2",
    "benchmark_v2_vs_others",
    "FusedFFN",
    "benchmark_ffn_vs_pytorch",
    "FusedInstanceNorm2d",
    "benchmark_instance_norm_vs_pytorch",
    "FusedConvInstanceNormReLU",
    "ResidualBlock",
    "benchmark_conv_fusion_vs_pytorch",
    "run_conv_fusion_benchmark",
]

__version__ = "0.1.0"
