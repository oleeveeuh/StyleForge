"""
Wrapper for custom FFN kernel compatible with LLM workloads

Adapts your existing ffn.cu kernel to work with Llama-2 style inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for kernel imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import your existing kernel
try:
    from kernels.ffn_wrapper import FusedFFN
    CUSTOM_KERNEL_AVAILABLE = True
except ImportError:
    print("Warning: FusedFFN not found.")
    CUSTOM_KERNEL_AVAILABLE = False


class CustomFFN(nn.Module):
    """
    Custom feed-forward network using fused CUDA kernel

    Architecture: x -> FC1 -> GELU -> FC2 -> output

    Features:
    - Single kernel launch (vs 3 in PyTorch)
    - Inline GELU using PTX assembly
    - Eliminates intermediate tensor allocations
    - Shared memory tiling for large matrices

    Compatible with Llama-2 architecture.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        if CUSTOM_KERNEL_AVAILABLE:
            # Use the existing fused FFN kernel
            self.ffn = FusedFFN(
                embed_dim=hidden_size,
                ffn_dim=intermediate_size,
                dropout=dropout,
                bias=bias,
            )
        else:
            # Fallback: create standard components
            self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
            self.ffn = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.use_custom_kernel = CUSTOM_KERNEL_AVAILABLE

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using custom CUDA kernel

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        if self.use_custom_kernel and self.ffn is not None:
            # Use custom fused kernel
            output = self.ffn(hidden_states)
        else:
            # Fallback to PyTorch implementation
            output = self._pytorch_ffn(hidden_states)

        if self.dropout is not None:
            output = self.dropout(output)

        return output

    def _pytorch_ffn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fallback PyTorch implementation"""
        if not hasattr(self, 'fc1'):
            # Initialize fallback components if not already done
            self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
            self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
            self.to(hidden_states.device)

        # FC1: [batch, seq, hidden] @ [hidden, intermediate]^T -> [batch, seq, intermediate]
        intermediate = self.fc1(hidden_states)

        # GELU activation
        intermediate = F.gelu(intermediate)

        # FC2: [batch, seq, intermediate] @ [intermediate, hidden]^T -> [batch, seq, hidden]
        output = self.fc2(intermediate)

        return output


class PyTorchFFN(nn.Module):
    """
    Standard PyTorch FFN for baseline comparison

    Identical architecture to CustomFFN but using PyTorch ops.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass"""
        intermediate = F.gelu(self.fc1(hidden_states))
        output = self.fc2(intermediate)

        if self.dropout is not None:
            output = self.dropout(output)

        return output


def count_ffn_operations(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int
) -> dict:
    """
    Count FLOPs for FFN layer

    Useful for estimating theoretical throughput.
    """
    # FC1: [B, S, H] @ [H, I] = B*S*H*I MACs = 2*B*S*H*I FLOPs
    fc1_flops = 2 * batch_size * seq_len * hidden_size * intermediate_size

    # GELU: ~5 ops per element (approx)
    gelu_flops = 5 * batch_size * seq_len * intermediate_size

    # FC2: [B, S, I] @ [I, H] = B*S*I*H MACs = 2*B*S*I*H FLOPs
    fc2_flops = 2 * batch_size * seq_len * intermediate_size * hidden_size

    total_flops = fc1_flops + gelu_flops + fc2_flops

    return {
        'fc1_flops': fc1_flops,
        'gelu_flops': gelu_flops,
        'fc2_flops': fc2_flops,
        'total_flops': total_flops,
        'total_gflops': total_flops / 1e9,
    }
