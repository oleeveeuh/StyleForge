"""
Utility functions for LLM model handling and testing
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path for kernel imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def create_dummy_attention_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create dummy Q, K, V tensors for attention benchmarking

    Args:
        batch_size: Number of sequences in batch
        seq_len: Sequence length (number of tokens)
        num_heads: Number of attention heads
        head_dim: Dimension per head
        device: Device to create tensors on

    Returns:
        Tuple of (Q, K, V) tensors with shape [batch, heads, seq_len, head_dim]
    """
    shape = (batch_size, num_heads, seq_len, head_dim)

    Q = torch.randn(shape, dtype=torch.float32, device=device)
    K = torch.randn(shape, dtype=torch.float32, device=device)
    V = torch.randn(shape, dtype=torch.float32, device=device)

    # Normalize to realistic ranges (prevents overflow in softmax)
    Q = Q / (head_dim ** 0.5)

    return Q, K, V


def create_dummy_ffn_inputs(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create dummy inputs for FFN benchmarking

    Args:
        batch_size: Number of sequences
        seq_len: Sequence length
        hidden_dim: Model hidden dimension
        intermediate_dim: FFN intermediate dimension
        device: Device to create tensors on

    Returns:
        Tuple of (input, w1, w2) where:
            input: [batch, seq_len, hidden_dim]
            w1: [intermediate_dim, hidden_dim] (up-projection)
            w2: [hidden_dim, intermediate_dim] (down-projection)
    """
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_dim,
        dtype=torch.float32, device=device
    )

    # Xavier initialization for weights (more realistic)
    w1 = torch.randn(intermediate_dim, hidden_dim, device=device)
    w1 *= (2.0 / (hidden_dim + intermediate_dim)) ** 0.5

    w2 = torch.randn(hidden_dim, intermediate_dim, device=device)
    w2 *= (2.0 / (hidden_dim + intermediate_dim)) ** 0.5

    return input_tensor, w1, w2


def estimate_memory_usage(
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    intermediate_dim: int,
    batch_size: int = 1,
    dtype_bytes: int = 4  # float32
) -> dict:
    """
    Estimate memory usage for transformer layer

    Returns dictionary with memory estimates in MB
    """
    head_dim = hidden_dim // num_heads

    # Attention memory
    qkv_memory = 3 * batch_size * seq_len * hidden_dim * dtype_bytes
    attention_scores = batch_size * num_heads * seq_len * seq_len * dtype_bytes
    attention_output = batch_size * seq_len * hidden_dim * dtype_bytes

    # FFN memory
    ffn_intermediate = batch_size * seq_len * intermediate_dim * dtype_bytes
    ffn_output = batch_size * seq_len * hidden_dim * dtype_bytes

    total_attention = qkv_memory + attention_scores + attention_output
    total_ffn = ffn_intermediate + ffn_output

    return {
        'attention_mb': total_attention / (1024 ** 2),
        'attention_scores_mb': attention_scores / (1024 ** 2),
        'ffn_mb': total_ffn / (1024 ** 2),
        'total_mb': (total_attention + total_ffn) / (1024 ** 2),
    }


def validate_attention_output(
    output1: torch.Tensor,
    output2: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-6
) -> Tuple[bool, float, float]:
    """
    Validate that two attention outputs match within tolerance

    Args:
        output1: First output tensor
        output2: Second output tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (is_close, max_error, mean_error)
    """
    is_close = torch.allclose(output1, output2, rtol=rtol, atol=atol)

    diff = (output1 - output2).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()

    return is_close, max_error, mean_error


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage statistics"""
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated,
    }


def print_gpu_info():
    """Print GPU information for debugging"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU Information:")
    print(f"  Device: {props.name}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
    print(f"  Multi-Processors: {props.multi_processor_count}")

    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"  Allocated: {mem_info['allocated_gb']:.2f} GB")
        print(f"  Reserved: {mem_info['reserved_gb']:.2f} GB")
