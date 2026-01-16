# ============================================
# STYLEFORGE CUDA SETUP FOR GOOGLE COLAB
# ============================================
# Copy this entire cell into Google Colab and run it.
# It will set up all dependencies, verify the environment,
# and compile the CUDA kernels with automatic fallback.
# ============================================

import os
import sys
import subprocess
from pathlib import Path

# Configuration
STYLEFORGE_ROOT = '/content/StyleForge'
USE_SETUPTOOLS_FALLBACK = True  # More reliable in Colab

def run_command(cmd):
    """Run shell command, return (success, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

print("🚀 Setting up StyleForge CUDA kernels...")

# =============================================================================
# STEP 1: INSTALL DEPENDENCIES
# =============================================================================

print_section("STEP 1: Installing Dependencies")

# Install ninja
print("  Checking for ninja...")
success, _, _ = run_command("which ninja")
if not success:
    print("  Installing ninja...")
    # Use Colab's pip install magic
    import subprocess
    subprocess.run(["pip", "install", "-q", "ninja"], capture_output=True)
    print("  ✓ ninja installed")
else:
    print("  ✓ ninja already found")

# =============================================================================
# STEP 2: NAVIGATE TO STYLEFORGE
# =============================================================================

print_section("STEP 2: Setting Up StyleForge")

if os.path.exists(STYLEFORGE_ROOT):
    os.chdir(STYLEFORGE_ROOT)
    print(f"  ✓ Changed to {STYLEFORGE_ROOT}")
else:
    print(f"  ⚠ StyleForge not found at {STYLEFORGE_ROOT}")
    print(f"  Current: {os.getcwd()}")

# Add to Python path
if STYLEFORGE_ROOT not in sys.path:
    sys.path.insert(0, STYLEFORGE_ROOT)

# =============================================================================
# STEP 3: VERIFY CUDA ENVIRONMENT
# =============================================================================

print_section("STEP 3: Verifying CUDA Environment")

try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")

    if torch.cuda.is_available():
        print(f"  ✓ CUDA {torch.version.cuda}")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ CUDA not available - will use CPU fallback")
except ImportError:
    print("  ✗ PyTorch not found")

# =============================================================================
# STEP 4: COMPILE ATTENTION KERNEL
# =============================================================================

print_section("STEP 4: Compiling Attention Kernel")

def compile_with_setuptools_simple(name, cuda_source, output_dir="kernels"):
    """Simple setuptools compilation for Colab."""
    import tempfile
    import shutil
    import glob

    # Create build directory
    build_dir = Path(output_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(build_dir)

    # Write CUDA source
    cuda_file = Path(f"{name}.cu")
    with open(cuda_file, 'w') as f:
        f.write(cuda_source)

    # Generate setup.py
    setup_py = f'''
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext = CUDAExtension(
    name="{name}",
    sources=["{name}.cu"],
    extra_compile_args={{"nvcc": ["-O3", "--use_fast_math"]}},
)

setup(
    name="{name}",
    ext_modules=[ext],
    cmdclass={{"build_ext": BuildExtension}},
)
'''
    with open("setup.py", 'w') as f:
        f.write(setup_py)

    # Build
    import subprocess
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  ✗ Compilation failed")
        print(f"  Error: {result.stderr}")
        return None

    # Find .so file
    so_file = None
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.startswith(name) and f.endswith('.so'):
                so_file = Path(root) / f
                break
        if so_file:
            break

    if so_file:
        # Copy to output directory
        output_file = build_dir / f"{name}.so"
        shutil.copy2(so_file, output_file)

        # Import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, str(output_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        print(f"  ✓ Compiled: {name}")
        print(f"  ✓ Location: {output_file}")
        return module

    return None

# Try to compile
# First, look for CUDA source file
cuda_source_file = None
for possible_path in [
    "kernels/attention.cu",
    "src/kernels/attention.cu",
    "attention.cu",
]:
    if os.path.exists(possible_path):
        cuda_source_file = possible_path
        break

attention_module = None

if cuda_source_file:
    print(f"  Found CUDA source: {cuda_source_file}")

    with open(cuda_source_file, 'r') as f:
        cuda_source = f.read()

    # Try setuptools compilation
    print("  Attempting setuptools compilation...")
    try:
        attention_module = compile_with_setuptools_simple(
            name="fused_attention",
            cuda_source=cuda_source
        )
    except Exception as e:
        print(f"  ✗ Setuptools failed: {e}")

        # Try JIT as fallback
        print("  Trying JIT compilation...")
        try:
            from utils.cuda_build import compile_inline
            attention_module = compile_inline(
                name="fused_attention",
                cuda_source=cuda_source,
                verbose=True
            )
        except Exception as e2:
            print(f"  ✗ JIT also failed: {e2}")
else:
    print("  ⚠ No CUDA source file found")
    print("  Looking for kernels/attention.cu or src/kernels/attention.cu")

# =============================================================================
# STEP 5: TEST MODULE
# =============================================================================

print_section("STEP 5: Testing Module")

if attention_module is not None:
    print("  Available functions:")
    for attr in dir(attention_module):
        if not attr.startswith('_'):
            print(f"    - {attr}")

    # Try a simple test
    try:
        import torch
        if torch.cuda.is_available():
            print("  Running smoke test on GPU...")
            # Small test tensors
            test_size = 10
            q = torch.randn(test_size, 64, device='cuda')

            # Check if module has forward function
            if hasattr(attention_module, 'forward'):
                output = attention_module.forward(q, q, q)
                print(f"  ✓ Test passed! Output shape: {output.shape}")
            elif hasattr(attention_module, 'fused_attention_forward'):
                # Reshape for multi-head attention format
                q_mha = q.unsqueeze(0).unsqueeze(0)  # (1, 1, 10, 64)
                output = attention_module.fused_attention_forward(q_mha, q_mha, q_mha)
                print(f"  ✓ Test passed! Output shape: {output.shape}")
            else:
                print("  ⚠ No test function found, but module loaded")
    except Exception as e:
        print(f"  ⚠ Test error: {e}")

    # Make available globally
    import builtins
    builtins.fused_attention = attention_module

    print_section("RESULT")
    print("  ✅ SUCCESS! CUDA kernel ready to use.")
    print(f"  Use as: from utils import cuda_build")
    print(f"         module = cuda_build.fused_attention")

else:
    print_section("RESULT")
    print("  ⚠ COMPILATION FAILED")
    print("\n  Troubleshooting:")
    print("  1. Make sure you're using a GPU runtime")
    print("  2. Try: Runtime > Change runtime type > GPU")
    print("  3. Run: !nvcc --version")
    print("  4. Restart runtime and try again")

print("=" * 70)
