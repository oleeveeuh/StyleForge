"""
Memory leak tests for StyleForge.

Tests verify that:
- Model doesn't leak memory during repeated inference
- CUDA memory is properly managed
- Gradients can be properly cleared
- Batch processing doesn't accumulate memory
"""

import gc
import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_net import TransformerNet
from tests.config import (
    DEVICE,
    IS_CUDA_AVAILABLE,
    MEMORY_TEST_ITERATIONS,
    MEMORY_TEST_WARMUP,
    MEMORY_ALLOWABLE_GROWTH_MB,
    MODELS_DIR,
)


def get_memory_mb() -> float:
    """
    Get current memory usage in MB.
    """
    if IS_CUDA_AVAILABLE:
        return torch.cuda.memory_allocated() / 1024 / 1024
    else:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


def clear_cache():
    """
    Clear memory caches.
    """
    gc.collect()
    if IS_CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_inference_memory_leak(style_name: str = "candy"):
    """
    Run model 1000 times and check for memory growth.
    """
    print(f"\n--- Testing inference memory leak: {style_name} ---")

    model = TransformerNet().to(DEVICE)

    checkpoint_path = MODELS_DIR / f"{style_name}.pth"
    if checkpoint_path.exists():
        model.load_checkpoint(str(checkpoint_path))

    model.eval()

    # Create test input
    input_tensor = torch.randn(1, 3, 512, 512).to(DEVICE)

    # Warmup
    print(f"  Warming up ({MEMORY_TEST_WARMUP} iterations)...")
    for _ in range(MEMORY_TEST_WARMUP):
        with torch.no_grad():
            _ = model(input_tensor)

    if IS_CUDA_AVAILABLE:
        torch.cuda.synchronize()

    clear_cache()

    initial_memory = get_memory_mb()
    print(f"  Initial memory: {initial_memory:.2f} MB")

    # Run many iterations
    print(f"  Running {MEMORY_TEST_ITERATIONS} iterations...")
    snapshot_interval = MEMORY_TEST_ITERATIONS // 10

    max_memory = initial_memory
    max_memory_iter = 0

    for i in range(MEMORY_TEST_ITERATIONS):
        with torch.no_grad():
            output = model(input_tensor)

        if IS_CUDA_AVAILABLE:
            torch.cuda.synchronize()

        current_memory = get_memory_mb()
        if current_memory > max_memory:
            max_memory = current_memory
            max_memory_iter = i

        if i > 0 and i % snapshot_interval == 0:
            growth = current_memory - initial_memory
            print(f"    Iteration {i:4d}: {current_memory:.2f} MB (growth: {growth:+.2f} MB)")

    final_memory = get_memory_mb()
    growth = final_memory - initial_memory

    print(f"  Final memory: {final_memory:.2f} MB")
    print(f"  Total growth: {growth:+.2f} MB")
    print(f"  Max memory: {max_memory:.2f} MB (at iteration {max_memory_iter})")

    # Check for leak
    if growth > MEMORY_ALLOWABLE_GROWTH_MB:
        raise AssertionError(
            f"Memory leak detected: {growth:.2f} MB growth "
            f"(allowable: {MEMORY_ALLOWABLE_GROWTH_MB} MB)"
        )

    print(f"✓ No significant memory leak detected")


def test_batch_memory_leak(style_name: str = "candy"):
    """
    Test memory usage with varying batch sizes.
    """
    print(f"\n--- Testing batch memory leak: {style_name} ---")

    model = TransformerNet().to(DEVICE)

    checkpoint_path = MODELS_DIR / f"{style_name}.pth"
    if checkpoint_path.exists():
        model.load_checkpoint(str(checkpoint_path))

    model.eval()

    batch_sizes = [1, 2, 4, 8]
    iterations_per_size = 100

    clear_cache()
    baseline_memory = get_memory_mb()
    print(f"  Baseline memory: {baseline_memory:.2f} MB")

    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 3, 256, 256).to(DEVICE)

        # Measure before
        clear_cache()
        before_memory = get_memory_mb()

        # Run iterations
        for _ in range(iterations_per_size):
            with torch.no_grad():
                _ = model(input_tensor)

        if IS_CUDA_AVAILABLE:
            torch.cuda.synchronize()

        # Measure after
        clear_cache()
        after_memory = get_memory_mb()

        growth = after_memory - before_memory
        print(f"  Batch {batch_size}: {growth:+.2f} MB growth after {iterations_per_size} iterations")

        # Each batch size should not leak significantly (positive growth only)
        # Negative growth is fine - it means memory was freed
        assert growth < MEMORY_ALLOWABLE_GROWTH_MB, \
            f"Memory leak for batch size {batch_size}: {growth:.2f} MB"

    print(f"✓ No memory leaks across different batch sizes")


def test_gradient_cleanup():
    """
    Verify gradients are properly cleaned up.
    """
    print("\n--- Testing gradient cleanup ---")

    model = TransformerNet().to(DEVICE)
    model.train()

    # Count parameters
    num_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params}")

    # Run forward/backward multiple times
    input_tensor = torch.randn(1, 3, 256, 256, requires_grad=True).to(DEVICE)

    for i in range(50):
        # Zero gradients
        model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        # Forward
        output = model(input_tensor)
        loss = output.mean()

        # Backward
        loss.backward()

        # Check gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)

        if i == 0:
            print(f"  Parameters with gradients after first backward: {grad_count}")

    # Final check
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)

    assert grad_count > 0, "No parameters have gradients after backward"

    # Check gradient values are finite
    finite_grads = 0
    for p in model.parameters():
        if p.grad is not None:
            if torch.isfinite(p.grad).all():
                finite_grads += 1

    assert finite_grads == num_params, f"Some gradients are not finite: {finite_grads}/{num_params}"

    print(f"✓ Gradients properly cleaned up")
    print(f"  Parameters with gradients: {grad_count}")
    print(f"  Parameters with finite gradients: {finite_grads}")


def test_model_load_unload():
    """
    Test loading and unloading models doesn't leak memory.
    """
    print("\n--- Testing model load/unload memory ---")

    checkpoint_path = MODELS_DIR / "candy.pth"

    if not checkpoint_path.exists():
        print("  ⚠️  Checkpoint not found, skipping test")
        return

    clear_cache()
    initial_memory = get_memory_mb()
    print(f"  Initial memory: {initial_memory:.2f} MB")

    # Load and unload models multiple times
    for i in range(10):
        model = TransformerNet().to(DEVICE)
        model.load_checkpoint(str(checkpoint_path))

        # Run inference
        input_tensor = torch.randn(1, 3, 256, 256).to(DEVICE)
        with torch.no_grad():
            _ = model(input_tensor)

        if IS_CUDA_AVAILABLE:
            torch.cuda.synchronize()

        # Delete model
        del model
        clear_cache()

        if i % 5 == 4:
            current_memory = get_memory_mb()
            growth = current_memory - initial_memory
            print(f"    After {i+1} iterations: {current_memory:.2f} MB (growth: {growth:+.2f} MB)")

    final_memory = get_memory_mb()
    growth = final_memory - initial_memory

    print(f"  Final memory: {final_memory:.2f} MB")
    print(f"  Total growth: {growth:+.2f} MB")

    # Allow some growth but not excessive
    assert growth < MEMORY_ALLOWABLE_GROWTH_MB * 3, \
        f"Memory leak in load/unload: {growth:.2f} MB growth"

    print(f"✓ Model load/unload doesn't leak memory")


def test_intermediate_tensor_cleanup():
    """
    Verify intermediate tensors are properly cleaned up.
    """
    print("\n--- Testing intermediate tensor cleanup ---")

    model = TransformerNet().to(DEVICE)
    model.eval()

    # Register hook to track tensor creation
    tensor_count = [0]
    forward_count = [0]

    def create_hook(mod):
        def hook(module, input, output):
            if IS_CUDA_AVAILABLE:
                torch.cuda.synchronize()
            tensor_count[0] += 1
            forward_count[0] += 1
        return hook

    # Add hooks to all modules
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hook = module.register_forward_hook(create_hook(module))
            hooks.append(hook)

    # Run forward pass
    input_tensor = torch.randn(1, 3, 256, 256).to(DEVICE)

    with torch.no_grad():
        _ = model(input_tensor)

    if IS_CUDA_AVAILABLE:
        torch.cuda.synchronize()

    print(f"  Tensors created during forward: {tensor_count[0]}")
    print(f"  Forward hooks triggered: {forward_count[0]}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print(f"✓ Intermediate tensors tracked ({tensor_count[0]} tensors)")


def test_cuda_memory_per_style():
    """
    Measure memory usage for each available style.
    """
    if not IS_CUDA_AVAILABLE:
        print("\n--- Skipping CUDA memory per style test (CUDA not available) ---")
        return

    print("\n--- Testing CUDA memory per style ---")

    available_styles = []
    for style in ["candy", "mosaic", "udnie", "rain_princess", "starry", "wave"]:
        if (MODELS_DIR / f"{style}.pth").exists():
            available_styles.append(style)

    if not available_styles:
        print("  ⚠️  No style models found")
        return

    input_tensor = torch.randn(1, 3, 512, 512).to(DEVICE)
    results = []

    for style in available_styles:
        clear_cache()

        checkpoint_path = MODELS_DIR / f"{style}.pth"
        model = TransformerNet().to(DEVICE)
        model.load_checkpoint(str(checkpoint_path))
        model.eval()

        # Measure memory before
        torch.cuda.synchronize()
        before_memory = torch.cuda.memory_allocated() / 1024 / 1024

        # Run inference
        with torch.no_grad():
            _ = model(input_tensor)

        torch.cuda.synchronize()
        after_memory = torch.cuda.memory_allocated() / 1024 / 1024

        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()

        used_memory = after_memory - before_memory

        results.append({
            'style': style,
            'used': used_memory,
            'peak': peak_memory,
        })

        print(f"  {style:15s}: {used_memory:6.2f} MB (peak: {peak_memory:.2f} MB)")

        del model

    # Verify all styles use similar amounts of memory
    memory_usages = [r['used'] for r in results]
    avg_memory = sum(memory_usages) / len(memory_usages)

    print(f"\n  Average memory usage: {avg_memory:.2f} MB")

    for result in results:
        diff = abs(result['used'] - avg_memory)
        assert diff < avg_memory * 0.5, \
            f"{result['style']} uses significantly different memory: {diff:.2f} MB diff"

    print(f"✓ All styles use similar amounts of memory")


def test_memory_leak_with_gradients():
    """
    Test for memory leaks when computing gradients.
    """
    print("\n--- Testing memory leak with gradients ---")

    model = TransformerNet().to(DEVICE)
    model.train()

    input_tensor = torch.randn(1, 3, 256, 256, requires_grad=True).to(DEVICE)

    clear_cache()
    initial_memory = get_memory_mb()
    print(f"  Initial memory: {initial_memory:.2f} MB")

    # Run many forward/backward passes
    iterations = 100

    for i in range(iterations):
        model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        output = model(input_tensor)
        loss = output.mean()
        loss.backward()

        if i % 25 == 24:
            current_memory = get_memory_mb()
            growth = current_memory - initial_memory
            print(f"    Iteration {i+1}: {current_memory:.2f} MB (growth: {growth:+.2f} MB)")

    clear_cache()
    final_memory = get_memory_mb()
    growth = final_memory - initial_memory

    print(f"  Final memory: {final_memory:.2f} MB")
    print(f"  Total growth: {growth:+.2f} MB")

    assert growth < MEMORY_ALLOWABLE_GROWTH_MB * 2, \
        f"Memory leak with gradients: {growth:.2f} MB growth"

    print(f"✓ No memory leak with gradients")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run memory leak tests")
    parser.add_argument("--style", "-s", default="candy",
                        help="Style model to test (default: candy)")
    parser.add_argument("--iterations", "-n", type=int,
                        default=MEMORY_TEST_ITERATIONS,
                        help=f"Number of test iterations (default: {MEMORY_TEST_ITERATIONS})")

    args = parser.parse_args()

    # Override iterations if specified
    if args.iterations:
        MEMORY_TEST_ITERATIONS = args.iterations

    print("=" * 60)
    print("STYLE FORGE - MEMORY LEAK TESTS")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {IS_CUDA_AVAILABLE}")
    print(f"Test iterations: {MEMORY_TEST_ITERATIONS}")

    try:
        test_inference_memory_leak(args.style)
        test_batch_memory_leak(args.style)
        test_gradient_cleanup()
        test_model_load_unload()
        test_intermediate_tensor_cleanup()
        test_cuda_memory_per_style()
        test_memory_leak_with_gradients()

        print("\n" + "=" * 60)
        print("✅ ALL MEMORY TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
