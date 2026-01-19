"""
StyleForge - Fused Attention V3 Python Wrapper

V3 uses register-based accumulation (no shared memory for V).
Educational kernel - still slower than Flash Attention 2 due to
fundamental limitations (element-wise matmul vs tensor cores).
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from utils import compile_inline

_attention_v3_module = None

def get_attention_v3_module():
    global _attention_v3_module

    if _attention_v3_module is not None:
        return _attention_v3_module

    kernel_path = Path(__file__).parent / "attention_v3.cu"

    if not kernel_path.exists():
        raise FileNotFoundError(f"V3 kernel not found at {kernel_path}")

    cuda_source = kernel_path.read_text()

    print("Compiling fused attention V3 kernel (register-based)...")
    _attention_v3_module = compile_inline(
        name='fused_attention_v3',
        cuda_source=cuda_source,
        functions=['fused_attention_v3'],
        build_directory=Path('build_v3'),
        verbose=False
    )
    print("V3 Compilation complete!")

    return _attention_v3_module

class FusedAttentionV3Function(torch.autograd.Function):
    MAX_SEQ_LEN = 4096  # Conservative limit
    MAX_HEAD_DIM = 128

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        w_out: torch.Tensor,
        bias_qkv: Optional[torch.Tensor],
        bias_out: Optional[torch.Tensor],
        num_heads: int,
        scale: float
    ) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        batch_size = x.size(0)
        seq_len = x.size(1)
        embed_dim = x.size(2)
        head_dim = embed_dim // num_heads

        if seq_len > FusedAttentionV3Function.MAX_SEQ_LEN:
            raise ValueError(f"seq_len {seq_len} exceeds MAX_SEQ_LEN {FusedAttentionV3Function.MAX_SEQ_LEN}")

        module = get_attention_v3_module()

        ctx.save_for_backward(x, w_qkv, w_out, bias_qkv, bias_out)
        ctx.num_heads = num_heads
        ctx.scale = scale
        ctx.embed_dim = embed_dim

        output = module.fused_attention_v3(
            x.contiguous(),
            w_qkv.contiguous(),
            w_out.contiguous(),
            bias_qkv,
            bias_out,
            scale,
            num_heads
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # No autograd support
        return None, None, None, None, None, None, None

class FusedAttentionV3(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.w_qkv = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.bias_qkv = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None

        self.w_out = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.bias_out = nn.Parameter(torch.empty(embed_dim)) if bias else None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w_qkv)
        nn.init.xavier_uniform_(self.w_out)
        if self.bias_qkv is not None:
            nn.init.zeros_(self.bias_qkv)
        if self.bias_out is not None:
            nn.init.zeros_(self.bias_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FusedAttentionV3Function.apply(
            x,
            self.w_qkv,
            self.w_out,
            self.bias_qkv,
            self.bias_out,
            self.num_heads,
            self.scale
        )
