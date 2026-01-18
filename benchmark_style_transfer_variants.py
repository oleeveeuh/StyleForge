#!/usr/bin/env python3
"""
StyleForge - Style Transfer Variant Benchmark

Compares three TransformerNet implementations:
1. Baseline (PyTorch) - No CUDA kernels
2. Auto - FusedInstanceNorm2d when available
3. Fused - Fully fused Conv+IN+ReLU kernels

Usage:
    python benchmark_style_transfer_variants.py
    python benchmark_style_transfer_variants.py --sizes 256 512 1024
"""

import argparse
import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np

from models.transformer_net import (
    TransformerNet,
    TransformerNetBaseline,
    TransformerNetFused,
    create_transformer_net,
    get_available_variants,
)


def benchmark_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    warmup: int = 10,
    iterations: int = 50,
) -> Dict[str, float]:
    """
    Benchmark a single model.

    Returns:
        Dict with: avg_ms, std_ms, min_ms, max_ms, fps
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(input_tensor)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(input_tensor)
                times.append((time.perf_counter() - t0) * 1000)

    return {
        "avg_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "fps": 1000 / np.mean(times),
    }


def compare_variants(
    size: int,
    checkpoint_path: str = None,
    warmup: int = 10,
    iterations: int = 50,
) -> Dict:
    """
    Compare all variants at a given resolution.

    Args:
        size: Image size (square)
        checkpoint_path: Optional checkpoint to load
        warmup: Warmup iterations
        iterations: Benchmark iterations

    Returns:
        Dict with results for each variant
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Resolution: {size}x{size} on {device.type.upper()}")
    print(f"{'='*70}")

    input_tensor = torch.randn(1, 3, size, size, device=device)
    results = {}

    # Variants to test
    variants = [
        ("baseline", TransformerNetBaseline),
        ("auto", TransformerNet),
        ("fused", TransformerNetFused),
    ]

    baseline_time = None

    for variant_name, model_class in variants:
        try:
            # Create model
            model = model_class(num_residual_blocks=5).to(device)
            model.eval()

            # Load checkpoint if provided (for first variant only, then copy weights)
            if checkpoint_path and variant_name == "baseline":
                model.load_checkpoint(checkpoint_path, device)
                reference_state = model.state_dict()
            elif checkpoint_path and baseline_time is not None:
                # Copy weights from baseline for fair comparison
                model.load_state_dict(reference_state)

            # Benchmark
            print(f"\n{variant_name.upper():10} - JIT compiling...", end="", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            metrics = benchmark_model(model, input_tensor, warmup, iterations)

            # Store baseline time for speedup calculation
            if variant_name == "baseline":
                baseline_time = metrics["avg_ms"]

            results[variant_name] = metrics

            # Calculate speedup
            if baseline_time and variant_name != "baseline":
                speedup = baseline_time / metrics["avg_ms"]
                print(f"\r{variant_name.upper():10} {metrics['avg_ms']:6.2f} ms  ({metrics['fps']:5.1f} FPS)  {speedup:+.2f}x")
            else:
                print(f"\r{variant_name.upper():10} {metrics['avg_ms']:6.2f} ms  ({metrics['fps']:5.1f} FPS)  (baseline)")

        except Exception as e:
            print(f"\r{variant_name.upper():10} ERROR: {e}")
            results[variant_name] = None

    return results, baseline_time


def print_summary(all_results: Dict[int, Dict]):
    """Print summary table of all results."""

    print(f"\n{'='*70}")
    print("SUMMARY - Style Transfer Performance")
    print(f"{'='*70}")

    # Header
    print(f"\n{'Variant':<12} {'256':^12} {'512':^12} {'1024':^12} {'2048':^12}")
    print("-" * 60)

    # Data rows
    for variant in ["baseline", "auto", "fused"]:
        if variant not in list(all_results.values())[0]:
            continue

        row = f"{variant.upper():<12}"
        for size, results in all_results.items():
            if variant in results and results[variant]:
                fps = results[variant]["fps"]
                row += f"{fps:6.1f} FPS  "
            else:
                row += " " * 12
        print(row)

    # Speedup rows
    print("\nSpeedup vs Baseline:")
    print(f"{'Variant':<12} {'256':^12} {'512':^12} {'1024':^12} {'2048':^12}")
    print("-" * 60)

    for variant in ["auto", "fused"]:
        row = f"{variant.upper():<12}"
        for size, results in all_results.items():
            baseline = results.get("baseline", {})
            variant_result = results.get(variant, {})

            if baseline and variant_result:
                baseline_ms = baseline["avg_ms"]
                variant_ms = variant_result["avg_ms"]
                speedup = baseline_ms / variant_ms
                row += f"{speedup:6.2f}x     "
            else:
                row += " " * 12
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Benchmark StyleTransfer variants")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
        help="Image sizes to benchmark",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file for realistic weights",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Benchmark iterations",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("StyleForge - Style Transfer Variant Benchmark")
    print("=" * 70)

    # Check device
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️  CUDA not available - running on CPU")

    # Check available variants
    available = get_available_variants()
    print(f"\nAvailable variants: {', '.join(available)}")

    all_results = {}

    for size in args.sizes:
        results, _ = compare_variants(
            size=size,
            checkpoint_path=args.checkpoint,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        all_results[size] = results

    print_summary(all_results)

    # Print conclusions
    print(f"\n{'='*70}")
    print("CONCLUSIONS")
    print(f"{'='*70}")

    if len(all_results) > 0 and 512 in all_results:
        r = all_results[512]
        if "baseline" in r and r["baseline"] and "auto" in r and r["auto"]:
            speedup_auto = r["baseline"]["avg_ms"] / r["auto"]["avg_ms"]
            print(f"\n• FusedInstanceNorm2d: {speedup_auto:.2f}x speedup vs PyTorch baseline")

        if "baseline" in r and r["baseline"] and "fused" in r and r["fused"]:
            speedup_fused = r["baseline"]["avg_ms"] / r["fused"]["avg_ms"]
            print(f"• Fully Fused (Conv+IN+ReLU): {speedup_fused:.2f}x speedup vs PyTorch baseline")

        if "fused" in r and r["fused"] and "auto" in r and r["auto"]:
            speedup = r["auto"]["avg_ms"] / r["fused"]["avg_ms"]
            print(f"• Fully Fused vs FusedInstanceNorm2d: {speedup:.2f}x additional speedup")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
