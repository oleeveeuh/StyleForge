"""
StyleForge - Performance Metrics Module

Calculate and track performance metrics including:
- Latency (ms)
- Throughput (FPS/images per second)
- Memory bandwidth (GB/s)
- GPU utilization (estimated)
- Arithmetic intensity
- FLOPs calculation
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GPUSpecs:
    """GPU specifications for performance calculations."""
    name: str
    compute_capability: str
    total_memory_gb: float
    multi_processor_count: int
    peak_tflops_fp32: float
    memory_bandwidth_gbps: float

    @classmethod
    def from_device(cls, device: torch.device = None) -> 'GPUSpecs':
        """Create GPUSpecs from current device."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type != 'cuda':
            raise ValueError("GPU specs only available for CUDA devices")

        props = torch.cuda.get_device_properties(device)

        # Estimate peak FLOP/s for FP32
        # These are rough estimates based on architecture
        cc = f"{props.major}.{props.minor}"

        # Peak TFLOPS estimates (very approximate)
        peak_tflops = props.multi_processor_count * 128 * 1.5e-3  # Conservative

        # Memory bandwidth estimate (GB/s)
        # These are rough estimates
        bandwidth_map = {
            '8.0': 900,   # Volta (V100)
            '7.5': 660,   # Turing (RTX 2080)
            '8.6': 1000,  # Ampere (RTX 3080/3090)
            '8.9': 1008,  # Ada (RTX 4090)
            '9.0': 1000,  # Hopper (H100)
        }
        bandwidth = bandwidth_map.get(cc, 500)

        return cls(
            name=props.name,
            compute_capability=cc,
            total_memory_gb=props.total_memory / 1e9,
            multi_processor_count=props.multi_processor_count,
            peak_tflops_fp32=peak_tflops,
            memory_bandwidth_gbps=bandwidth,
        )


class PerformanceMetrics:
    """
    Calculate and track performance metrics.

    Metrics:
    - Latency (ms)
    - Throughput (FPS/images per second)
    - Memory bandwidth (GB/s)
    - GPU utilization (estimated)
    - Arithmetic intensity (FLOPs/byte)
    """

    @staticmethod
    def calculate_memory_bandwidth(
        input_bytes: int,
        output_bytes: int,
        time_ms: float,
        read_write_ratio: float = 1.0
    ) -> float:
        """
        Calculate memory bandwidth in GB/s.

        Args:
            input_bytes: Total input data size in bytes
            output_bytes: Total output data size in bytes
            time_ms: Execution time in milliseconds
            read_write_ratio: Ratio of reads to writes (for estimation)

        Returns:
            Memory bandwidth in GB/s
        """
        total_bytes = input_bytes + output_bytes
        time_s = time_ms / 1000.0

        if time_s <= 0:
            return 0.0

        bandwidth_gbps = (total_bytes / 1e9) / time_s
        return bandwidth_gbps

    @staticmethod
    def calculate_arithmetic_intensity(
        flops: int,
        memory_bytes: int
    ) -> float:
        """
        Calculate arithmetic intensity (FLOPs per byte).

        High intensity (>10) = compute-bound
        Low intensity (<1) = memory-bound

        Args:
            flops: Number of floating point operations
            memory_bytes: Total bytes transferred

        Returns:
            Arithmetic intensity (FLOPs/byte)
        """
        if memory_bytes <= 0:
            return float('inf')

        return flops / memory_bytes

    @staticmethod
    def estimate_gpu_utilization(
        achieved_tflops: float,
        peak_tflops: float
    ) -> float:
        """
        Estimate GPU compute utilization.

        Args:
            achieved_tflops: Achieved TFLOP/s
            peak_tflops: Peak theoretical TFLOP/s

        Returns:
            Utilization percentage (0-100)
        """
        if peak_tflops <= 0:
            return 0.0

        return min(100.0, (achieved_tflops / peak_tflops) * 100.0)

    @staticmethod
    def estimate_memory_utilization(
        achieved_bandwidth_gbps: float,
        peak_bandwidth_gbps: float
    ) -> float:
        """
        Estimate memory bandwidth utilization.

        Args:
            achieved_bandwidth_gbps: Achieved bandwidth
            peak_bandwidth_gbps: Peak theoretical bandwidth

        Returns:
            Utilization percentage (0-100)
        """
        if peak_bandwidth_gbps <= 0:
            return 0.0

        return min(100.0, (achieved_bandwidth_gbps / peak_bandwidth_gbps) * 100.0)

    @staticmethod
    def calculate_conv_flops(
        batch_size: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ) -> int:
        """
        Calculate FLOPs for 2D convolution.

        Args:
            batch_size: Batch size
            in_channels: Input channels
            out_channels: Output channels
            height: Input height
            width: Input width
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding

        Returns:
            Number of floating point operations
        """
        # Output dimensions
        out_height = (height + 2 * padding - kernel_size) // stride + 1
        out_width = (width + 2 * padding - kernel_size) // stride + 1

        # FLOPs per output element
        # Each output: in_channels * kernel_size * kernel_size multiplications
        #              + in_channels * kernel_size * kernel_size - 1 additions
        #              + 1 bias addition
        flops_per_output = in_channels * kernel_size * kernel_size * 2 + 1

        total_outputs = batch_size * out_channels * out_height * out_width

        return flops_per_output * total_outputs

    @staticmethod
    def calculate_conv_memory_bytes(
        batch_size: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        kernel_size: int,
        dtype_bytes: int = 4
    ) -> Tuple[int, int]:
        """
        Calculate memory bytes for convolution.

        Returns:
            (input_bytes, output_bytes)
        """
        # Input: [batch, in_channels, height, width]
        input_bytes = batch_size * in_channels * height * width * dtype_bytes

        # Output (assuming same padding, stride=1)
        output_bytes = batch_size * out_channels * height * width * dtype_bytes

        return input_bytes, output_bytes

    @staticmethod
    def calculate_instance_norm_flops(
        batch_size: int,
        channels: int,
        height: int,
        width: int
    ) -> int:
        """
        Calculate FLOPs for instance normalization.

        Includes: mean, variance, normalization, affine transform

        Args:
            batch_size: Batch size
            channels: Number of channels
            height: Feature map height
            width: Feature map width

        Returns:
            Number of floating point operations
        """
        spatial_size = height * width

        # Per channel operations (each channel independent in instance norm)
        # Mean: spatial_size adds
        # Variance: spatial_size multiplies + spatial_size adds
        # Normalize: spatial_size subtracts + spatial_size divides
        # Affine: spatial_size multiplies + spatial_size adds

        flops_per_channel = spatial_size * 6
        total_flops = batch_size * channels * flops_per_channel

        return total_flops

    @staticmethod
    def calculate_instance_norm_memory_bytes(
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        dtype_bytes: int = 4
    ) -> Tuple[int, int]:
        """
        Calculate memory bytes for instance normalization.

        Returns:
            (input_bytes, output_bytes)
        """
        # Input and output are same size
        bytes_per_channel = height * width * dtype_bytes
        total_bytes = batch_size * channels * bytes_per_channel

        return total_bytes, total_bytes

    @staticmethod
    def calculate_attention_flops(
        batch_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None
    ) -> int:
        """
        Calculate FLOPs for multi-head attention.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            embed_dim: Input embedding dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head (default: embed_dim // num_heads)

        Returns:
            Number of floating point operations
        """
        if head_dim is None:
            head_dim = embed_dim // num_heads

        # QKV projection: 3 * embed_dim * embed_dim * seq_len
        qkv_flops = 3 * embed_dim * embed_dim * seq_len * 2

        # Attention scores: batch * heads * seq_len * seq_len * head_dim
        attn_flops = batch_size * num_heads * seq_len * seq_len * head_dim * 2

        # Output projection: embed_dim * embed_dim * seq_len * 2
        out_flops = embed_dim * embed_dim * seq_len * 2

        total_flops = (qkv_flops + attn_flops + out_flops) * batch_size

        return total_flops

    @staticmethod
    def calculate_roofline_model(
        flops: int,
        memory_bytes: int,
        peak_tflops: float,
        peak_bandwidth_gbps: float
    ) -> Dict[str, float]:
        """
        Calculate roofline model predictions.

        The roofline model bounds performance by either:
        - Compute roof (peak FLOPs)
        - Memory roof (peak bandwidth * arithmetic intensity)

        Args:
            flops: Total FLOPs for operation
            memory_bytes: Total bytes transferred
            peak_tflops: Peak compute in TFLOP/s
            peak_bandwidth_gbps: Peak memory bandwidth in GB/s

        Returns:
            Dictionary with roofline predictions
        """
        if memory_bytes <= 0:
            return {
                'arithmetic_intensity': float('inf'),
                'compute_bound_achieved_tflops': peak_tflops,
                'memory_bound_achieved_tflops': 0.0,
                'roof_tflops': peak_tflops,
            }

        arithmetic_intensity = flops / memory_bytes

        # Ridge point where compute and memory bounds intersect
        ridge_point = peak_tflops / peak_bandwidth_gbps

        # Determine which roof we're bounded by
        if arithmetic_intensity >= ridge_point:
            # Compute-bound
            roof_tflops = peak_tflops
        else:
            # Memory-bound
            roof_tflops = peak_bandwidth_gbps * arithmetic_intensity

        return {
            'arithmetic_intensity': arithmetic_intensity,
            'ridge_point': ridge_point,
            'roof_tflops': roof_tflops,
            'is_compute_bound': arithmetic_intensity >= ridge_point,
        }

    @staticmethod
    def calculate_efficiency(
        achieved_tflops: float,
        roof_tflops: float
    ) -> float:
        """
        Calculate efficiency relative to roofline.

        Args:
            achieved_tflops: Achieved TFLOP/s
            roof_tflops: Roofline TFLOP/s

        Returns:
            Efficiency percentage (0-100)
        """
        if roof_tflops <= 0:
            return 0.0

        return min(100.0, (achieved_tflops / roof_tflops) * 100.0)


def analyze_kernel_performance(
    time_ms: float,
    flops: int,
    input_bytes: int,
    output_bytes: int,
    gpu_specs: GPUSpecs
) -> Dict[str, float]:
    """
    Comprehensive analysis of kernel performance.

    Args:
        time_ms: Execution time in milliseconds
        flops: Total FLOPs
        input_bytes: Input bytes
        output_bytes: Output bytes
        gpu_specs: GPU specifications

    Returns:
        Dictionary with performance metrics
    """
    time_s = time_ms / 1000.0

    # Compute metrics
    achieved_tflops = (flops / 1e12) / time_s
    bandwidth_gbps = PerformanceMetrics.calculate_memory_bandwidth(
        input_bytes, output_bytes, time_ms
    )
    arithmetic_intensity = PerformanceMetrics.calculate_arithmetic_intensity(
        flops, input_bytes + output_bytes
    )

    # Roofline analysis
    roofline = PerformanceMetrics.calculate_roofline_model(
        flops=flops,
        memory_bytes=input_bytes + output_bytes,
        peak_tflops=gpu_specs.peak_tflops_fp32,
        peak_bandwidth_gbps=gpu_specs.memory_bandwidth_gbps
    )

    # Utilization
    compute_util = PerformanceMetrics.estimate_gpu_utilization(
        achieved_tflops, gpu_specs.peak_tflops_fp32
    )
    memory_util = PerformanceMetrics.estimate_memory_utilization(
        bandwidth_gbps, gpu_specs.memory_bandwidth_gbps
    )

    # Efficiency
    efficiency = PerformanceMetrics.calculate_efficiency(
        achieved_tflops, roofline['roof_tflops']
    )

    return {
        'time_ms': time_ms,
        'throughput_fps': 1000.0 / time_ms,
        'achieved_tflops': achieved_tflops,
        'bandwidth_gbps': bandwidth_gbps,
        'arithmetic_intensity': arithmetic_intensity,
        'compute_utilization_percent': compute_util,
        'memory_utilization_percent': memory_util,
        'roofline_efficiency_percent': efficiency,
        'is_compute_bound': roofline['is_compute_bound'],
        'ridge_point': roofline['ridge_point'],
    }
