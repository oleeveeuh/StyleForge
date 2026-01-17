#!/usr/bin/env python3
"""
StyleForge End-to-End Style Transfer Pipeline

This module provides a complete style transfer system that maximizes use of
custom CUDA kernels:
- FusedInstanceNorm2d for Fast Style Transfer
- fused_attention_v1 for ViT-based Style Transfer

Features:
- Automatic backend selection (PyTorch/CUDA)
- Kernel usage verification
- Comprehensive benchmarking
- Quality metrics (SSIM, PSNR)
"""

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
from PIL import Image

# Local imports
from models.transformer_net import TransformerNet, AVAILABLE_STYLES
from models.vit_style_transfer import (
    StyleForgeTransformer,
    create_model,
    STYLEFORGE_MODELS
)
from models.custom_attention_wrapper import get_attention_kernel_stats, print_attention_stats
from utils.image_utils import load_image, preprocess_image, postprocess_image


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    backend: str
    model_name: str
    image_size: Tuple[int, int]
    iterations: int
    total_time: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    fps: float
    memory_mb: float
    cuda_kernel_calls: int = 0
    pytorch_fallback_calls: int = 0
    cuda_percentage: float = 0.0


@dataclass
class QualityMetrics:
    """Image quality comparison metrics."""
    ssim: float
    psnr: float
    mse: float
    mae: float


@dataclass
class PipelineConfig:
    """Configuration for StyleForgePipeline."""
    backend: Literal['auto', 'pytorch', 'cuda', 'hybrid'] = 'auto'
    model_type: Literal['fast', 'vit', 'hybrid'] = 'hybrid'
    fast_style: str = 'candy'
    vit_variant: str = 'small'
    device: Optional[str] = None
    verbose: bool = True


class StyleForgePipeline:
    """
    End-to-end style transfer pipeline with automatic kernel usage.

    Args:
        config: PipelineConfig instance or None for defaults

    Example:
        >>> pipeline = StyleForgePipeline()
        >>> output = pipeline.stylize("content.jpg", style="candy")
        >>> pipeline.save(output, "output.jpg")

        >>> # Benchmark
        >>> results = pipeline.benchmark(image_size=512, iterations=50)
        >>> print(results.summary())
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = self._get_device()
        self.model = None
        self.model_name = None
        self._kernel_debug_mode = False

        # Load model based on configuration
        self._load_model()

        if self.config.verbose:
            self._print_pipeline_info()

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device:
            return torch.device(self.config.device)

        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _load_model(self):
        """Load the appropriate model based on configuration."""
        backend = self.config.backend
        if backend == 'auto':
            backend = 'cuda' if torch.cuda.is_available() else 'pytorch'

        model_type = self.config.model_type

        if model_type == 'fast':
            self._load_fast_style_transfer(backend)
        elif model_type == 'vit':
            self._load_vit_style_transfer(backend)
        elif model_type == 'hybrid':
            # Use ViT if CUDA is available, else Fast Style Transfer
            if backend == 'cuda' and torch.cuda.is_available():
                self._load_vit_style_transfer(backend)
            else:
                self._load_fast_style_transfer(backend)

    def _load_fast_style_transfer(self, backend: str):
        """Load Fast Style Transfer model."""
        self.model_name = f"Fast Style Transfer ({self.config.fast_style})"
        self.model = TransformerNet(num_residual_blocks=5)

        # Try to load pre-trained weights
        checkpoint_path = Path('saved_models') / f'{self.config.fast_style}.pth'
        if checkpoint_path.exists():
            self.model.load_checkpoint(str(checkpoint_path))
            if self.config.verbose:
                print(f"✅ Loaded pre-trained weights: {checkpoint_path}")
        else:
            if self.config.verbose:
                print(f"⚠️  No pre-trained weights found, using random initialization")

        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_vit_style_transfer(self, backend: str):
        """Load ViT-based Style Transfer model."""
        self.model_name = f"ViT Style Transfer ({self.config.vit_variant})"
        use_cuda = backend == 'cuda'

        self.model = create_model(
            variant=self.config.vit_variant,
            use_cuda_kernels=use_cuda
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.config.verbose and use_cuda:
            self._print_kernel_stats()

    def _print_pipeline_info(self):
        """Print pipeline configuration information."""
        print("=" * 60)
        print("STYLEFORGE PIPELINE")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Backend: {self.config.backend}")

        if torch.cuda.is_available():
            print(f"CUDA Available: Yes")
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("CUDA Available: No")
        print("=" * 60)

    def _print_kernel_stats(self):
        """Print CUDA kernel statistics."""
        if hasattr(self.model, 'get_kernel_stats'):
            stats = self.model.get_kernel_stats()
            print(f"\nCUDA Kernel Statistics:")
            print(f"  Attention modules: {stats['attention_modules']}")
            print(f"  Max calls per forward: {stats['attention_modules'] * 2}")  # attn + ffn

    def enable_kernel_debug(self, enabled: bool = True):
        """Enable debug output for kernel usage."""
        self._kernel_debug_mode = enabled

    def reset_stats(self):
        """Reset kernel usage statistics."""
        if hasattr(self.model, 'get_kernel_stats'):
            for name, module in self.model.named_modules():
                if hasattr(module, 'reset_stats'):
                    module.reset_stats()

    def get_kernel_usage(self) -> Dict:
        """Get current kernel usage statistics."""
        if hasattr(self.model, 'get_kernel_stats'):
            return self.model.get_kernel_stats()

        if hasattr(self.model, 'print_kernel_stats'):
            # For Fast Style Transfer, check InstanceNorm usage
            has_cuda_norm = False
            for module in self.model.modules():
                if type(module).__name__ == 'FusedInstanceNorm2d':
                    has_cuda_norm = True
                    break

            return {
                'cuda_instance_norm': has_cuda_norm,
                'model_type': 'fast_style_transfer'
            }

        return {'cuda_kernels': False}

    def preprocess(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Preprocess input image for model.

        Args:
            image: Input image (path, PIL, or tensor)
            target_size: Optional target size (width, height)

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        if isinstance(image, (str, Path)):
            image = load_image(str(image))
        elif isinstance(image, torch.Tensor):
            return image

        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        return preprocess_image(image, self.model_name.startswith('Fast'))

    def postprocess(
        self,
        tensor: torch.Tensor,
        output_format: Literal['pil', 'tensor'] = 'pil'
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Postprocess model output.

        Args:
            tensor: Output tensor from model
            output_format: 'pil' for PIL Image, 'tensor' for tensor

        Returns:
            Processed output
        """
        result = postprocess_image(tensor)

        if output_format == 'tensor':
            return result

        if isinstance(result, Image.Image):
            return result

        # Convert tensor to PIL if needed
        if isinstance(result, torch.Tensor):
            result = result.squeeze(0).permute(1, 2, 0).cpu()
            result = (result * 255).clamp(0, 255).to(torch.uint8)
            return Image.fromarray(result.numpy())

        return result

    def stylize(
        self,
        content_image: Union[str, Path, Image.Image, torch.Tensor],
        style_image: Optional[Union[str, Path, Image.Image, torch.Tensor]] = None,
        return_tensor: bool = False
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Apply style transfer to an image.

        Args:
            content_image: Content image (what to preserve)
            style_image: Style image (artistic style) - required for ViT model
            return_tensor: If True, return tensor instead of PIL Image

        Returns:
            Stylized image
        """
        # Preprocess content
        content = self.preprocess(content_image)
        content = content.to(self.device)

        # Preprocess style if needed
        style = None
        if style_image is not None and 'ViT' in self.model_name:
            style = self.preprocess(style_image)
            style = style.to(self.device)

        # Forward pass with kernel debug
        with torch.no_grad():
            if self._kernel_debug_mode:
                print(f"\n[KERNEL DEBUG] Forward pass start")

            if style is not None:
                output = self.model(content, style)
            else:
                output = self.model(content)

            if self._kernel_debug_mode:
                stats = self.get_kernel_usage()
                print(f"[KERNEL DEBUG] Forward pass complete")
                print(f"[KERNEL DEBUG] Stats: {stats}")

        # Postprocess
        return self.postprocess(output, 'tensor' if return_tensor else 'pil')

    def save(
        self,
        image: Union[Image.Image, torch.Tensor],
        path: Union[str, Path]
    ):
        """Save output image to file."""
        if isinstance(image, torch.Tensor):
            image = self.postprocess(image, 'pil')

        image.save(path)
        if self.config.verbose:
            print(f"✅ Saved to {path}")

    def benchmark(
        self,
        image_size: Union[int, Tuple[int, int]] = (512, 512),
        iterations: int = 50,
        warmup: int = 10,
        collect_memory: bool = True
    ) -> BenchmarkResult:
        """
        Benchmark the pipeline performance.

        Args:
            image_size: Size of test image (width, height) or single int for square
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            collect_memory: Whether to collect memory statistics

        Returns:
            BenchmarkResult with detailed timing information
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        print(f"\n{'='*60}")
        print(f"BENCHMARK: {self.model_name}")
        print(f"{'='*60}")
        print(f"Image size: {image_size[0]}x{image_size[1]}")
        print(f"Iterations: {iterations}")
        print(f"Warmup: {warmup}")
        print(f"Device: {self.device}")

        # Create dummy input
        if 'ViT' in self.model_name:
            content = torch.randn(1, 3, image_size[1], image_size[0])
            style = torch.randn(1, 3, image_size[1], image_size[0])
        else:
            content = torch.randn(1, 3, image_size[1], image_size[0])
            style = None

        content = content.to(self.device)
        if style is not None:
            style = style.to(self.device)

        # Reset stats
        self.reset_stats()

        # Warmup
        print(f"\nWarming up ({warmup} iterations)...")
        for _ in range(warmup):
            with torch.no_grad():
                if style is not None:
                    _ = self.model(content, style)
                else:
                    _ = self.model(content)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        print(f"Benchmarking ({iterations} iterations)...")
        times = []

        # Get initial memory
        if collect_memory and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            initial_memory = torch.cuda.memory_allocated() / 1e6

        for i in range(iterations):
            start = time.perf_counter()

            with torch.no_grad():
                if style is not None:
                    output = self.model(content, style)
                else:
                    output = self.model(content)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{iterations}] {times[-1]:.2f} ms")

        # Get final memory
        if collect_memory and self.device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1e6
            memory_mb = peak_memory - initial_memory
        else:
            memory_mb = 0.0

        # Calculate statistics
        times_ms = times
        total_time = sum(times)
        avg_time = sum(times_ms) / len(times_ms)
        min_time = min(times_ms)
        max_time = max(times_ms)
        fps = 1000 / avg_time

        # Get kernel stats
        kernel_stats = self.get_kernel_usage()
        cuda_calls = kernel_stats.get('cuda_kernel_calls', 0)
        pytorch_calls = kernel_stats.get('pytorch_fallback_calls', 0)
        cuda_pct = kernel_stats.get('cuda_percentage', 0.0)

        # Create result
        result = BenchmarkResult(
            backend='cuda' if self.device.type == 'cuda' else 'pytorch',
            model_name=self.model_name,
            image_size=image_size,
            iterations=iterations,
            total_time=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            fps=fps,
            memory_mb=memory_mb,
            cuda_kernel_calls=cuda_calls,
            pytorch_fallback_calls=pytorch_calls,
            cuda_percentage=cuda_pct
        )

        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Average: {avg_time:.2f} ms")
        print(f"Min:     {min_time:.2f} ms")
        print(f"Max:     {max_time:.2f} ms")
        print(f"FPS:     {fps:.2f}")
        if memory_mb > 0:
            print(f"Memory:  {memory_mb:.2f} MB")
        print(f"\nKernel Usage:")
        print(f"  CUDA calls:     {cuda_calls}")
        print(f"  PyTorch calls:  {pytorch_calls}")
        print(f"  CUDA usage:     {cuda_pct:.1f}%")
        print(f"{'='*60}")

        return result

    def compare_quality(
        self,
        output1: Union[Image.Image, torch.Tensor],
        output2: Union[Image.Image, torch.Tensor]
    ) -> QualityMetrics:
        """
        Compare quality between two outputs.

        Args:
            output1: First output (e.g., PyTorch baseline)
            output2: Second output (e.g., CUDA accelerated)

        Returns:
            QualityMetrics with SSIM, PSNR, MSE, MAE
        """
        # Convert to tensors
        if isinstance(output1, Image.Image):
            t1 = torch.from_numpy(
                torch.tensor(list(output1.getdata())).reshape(output1.size[1], output1.size[0], 3).permute(2, 0, 1).numpy()
            ).float() / 255.0
        else:
            t1 = output1

        if isinstance(output2, Image.Image):
            t2 = torch.from_numpy(
                torch.tensor(list(output2.getdata())).reshape(output2.size[1], output2.size[0], 3).permute(2, 0, 1).numpy()
            ).float() / 255.0
        else:
            t2 = output2

        # Resize to match if needed
        if t1.shape != t2.shape:
            t2 = torch.nn.functional.interpolate(
                t2.unsqueeze(0), size=t1.shape[1:], mode='bilinear'
            ).squeeze(0)

        # Calculate metrics
        mse = torch.mean((t1 - t2) ** 2).item()
        mae = torch.mean(torch.abs(t1 - t2)).item()
        psnr = 20 * torch.log10(torch.tensor(255.0)) - 10 * torch.log10(torch.tensor(mse + 1e-10))

        # Simple SSIM approximation
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu1 = torch.mean(t1)
        mu2 = torch.mean(t2)
        sigma1_sq = torch.var(t1)
        sigma2_sq = torch.var(t2)
        sigma12 = torch.mean((t1 - mu1) * (t2 - mu2))
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        ssim = ssim.item()

        return QualityMetrics(ssim=ssim, psnr=psnr.item(), mse=mse, mae=mae)

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1e6,
            'kernel_usage': self.get_kernel_usage()
        }


def create_pipeline(
    backend: str = 'auto',
    model_type: str = 'hybrid',
    style: str = 'candy',
    vit_variant: str = 'small',
    verbose: bool = True
) -> StyleForgePipeline:
    """
    Convenience function to create a StyleForgePipeline.

    Args:
        backend: 'auto', 'pytorch', 'cuda', or 'hybrid'
        model_type: 'fast', 'vit', or 'hybrid'
        style: Style name for Fast Style Transfer
        vit_variant: ViT variant ('small', 'base', 'large')
        verbose: Print verbose output

    Returns:
        Configured StyleForgePipeline
    """
    config = PipelineConfig(
        backend=backend,
        model_type=model_type,
        fast_style=style,
        vit_variant=vit_variant,
        verbose=verbose
    )
    return StyleForgePipeline(config)


# Convenience aliases
FastStylePipeline = lambda: create_pipeline(model_type='fast')
ViTStylePipeline = lambda: create_pipeline(model_type='vit')
HybridStylePipeline = lambda: create_pipeline(model_type='hybrid')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StyleForge Pipeline")
    parser.add_argument('--backend', choices=['auto', 'pytorch', 'cuda', 'hybrid'],
                        default='auto', help='Backend to use')
    parser.add_argument('--model', choices=['fast', 'vit', 'hybrid'],
                        default='hybrid', help='Model type')
    parser.add_argument('--style', default='candy', help='Style for Fast Style Transfer')
    parser.add_argument('--variant', choices=['small', 'base', 'large'],
                        default='small', help='ViT variant')
    parser.add_argument('--content', help='Content image path')
    parser.add_argument('--style-img', help='Style image path (for ViT)')
    parser.add_argument('--output', default='output.jpg', help='Output path')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--size', type=int, default=512, help='Benchmark image size')
    parser.add_argument('--iters', type=int, default=50, help='Benchmark iterations')
    parser.add_argument('--debug', action='store_true', help='Enable kernel debug')

    args = parser.parse_args()

    # Create pipeline
    pipeline = create_pipeline(
        backend=args.backend,
        model_type=args.model,
        style=args.style,
        vit_variant=args.variant
    )

    if args.debug:
        pipeline.enable_kernel_debug(True)

    # Run benchmark or stylize
    if args.benchmark:
        result = pipeline.benchmark(image_size=args.size, iterations=args.iters)

        # Save result to JSON
        results_path = Path('results') / 'benchmark_result.json'
        results_path.parent.mkdir(exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump({
                'backend': result.backend,
                'model': result.model_name,
                'image_size': result.image_size,
                'avg_time_ms': result.avg_time_ms,
                'fps': result.fps,
                'cuda_percentage': result.cuda_percentage,
            }, f, indent=2)

        print(f"\n✅ Results saved to {results_path}")

    elif args.content:
        output = pipeline.stylize(args.content, args.style_img)
        pipeline.save(output, args.output)

    else:
        print("No action specified. Use --benchmark or provide --content")
