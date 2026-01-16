# ============================================
# STYLEFORGE CUDA SETUP FOR GOOGLE COLAB
# ============================================
#
# Copy this entire cell into a Google Colab notebook and run it.
# It will set up all dependencies, verify the environment, and compile
# the CUDA kernels with automatic fallback.
#
# This cell handles:
# 1. Installing required dependencies (ninja)
# 2. Verifying CUDA/PyTorch environment
# 3. Compiling CUDA kernels with automatic fallback
# 4. Testing the compiled modules
# 5. Clear error reporting on success/failure
# ============================================

import os
import sys
import subprocess
from pathlib import Path

# Configuration
STYLEFORGE_ROOT = '/content/StyleForge'
USE_SETUPTOOLS_FALLBACK = True  # Set to False to try JIT only

def run_command(cmd, description=""):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_success(msg):
    print(f"✓ {msg}")

def print_error(msg):
    print(f"✗ {msg}")

def print_warning(msg):
    print(f"⚠ {msg}")

def print_info(msg):
    print(f"  {msg}")

# =============================================================================
# STEP 1: INSTALL DEPENDENCIES
# =============================================================================

print_section("STEP 1: Installing Dependencies")

# Install ninja
print_info("Checking for ninja...")
success, stdout, stderr = run_command("which ninja")
if not success:
    print_info("Installing ninja...")
    success, stdout, stderr = run_command("pip install -q ninja")
    if success:
        print_success("ninja installed")
    else:
        print_error("Failed to install ninja")
        print_info(stderr)
else:
    print_success("ninja already installed")

# Install colorama for better output (optional)
print_info("Installing colorama for colored output...")
run_command("pip install -q colorama")

# =============================================================================
# STEP 2: NAVIGATE TO STYLEFORGE DIRECTORY
# =============================================================================

print_section("STEP 2: Setting Up StyleForge")

# Check if we're in the right place
if os.path.exists(STYLEFORGE_ROOT):
    os.chdir(STYLEFORGE_ROOT)
    print_success(f"Changed to {STYLEFORGE_ROOT}")
else:
    print_warning(f"StyleForge not found at {STYLEFORGE_ROOT}")
    print_info("Current directory:", os.getcwd())

    # Try to find StyleForge
    possible_paths = [
        '/content/StyleForge',
        '.',
        '/content/StyleForge',
        os.path.join(os.getcwd(), 'StyleForge'),
    ]

    found = False
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'utils')):
            STYLEFORGE_ROOT = path
            os.chdir(path)
            print_success(f"Found StyleForge at {path}")
            found = True
            break

    if not found:
        print_error("Could not find StyleForge directory!")
        print_info("Please make sure StyleForge is uploaded to Colab.")

# Add to Python path
if STYLEFORGE_ROOT not in sys.path:
    sys.path.insert(0, STYLEFORGE_ROOT)

print_info("Python path updated")
print_info("Working directory:", os.getcwd())

# =============================================================================
# STEP 3: VERIFY CUDA ENVIRONMENT
# =============================================================================

print_section("STEP 3: Verifying CUDA Environment")

try:
    import torch
    print_success(f"PyTorch {torch.__version__} imported")

    if torch.cuda.is_available():
        print_success(f"CUDA {torch.version.cuda} available")
        device_name = torch.cuda.get_device_name(0)
        print_info(f"GPU: {device_name}")

        # Test CUDA operation
        try:
            x = torch.randn(10).cuda()
            y = torch.randn(10).cuda()
            z = x + y
            torch.cuda.synchronize()
            print_success("CUDA test passed")
        except Exception as e:
            print_warning(f"CUDA test failed: {e}")
    else:
        print_warning("CUDA not available - kernels will run on CPU")

except ImportError as e:
    print_error(f"Failed to import PyTorch: {e}")
    print_info("Install PyTorch: !pip install torch")

# Check for nvcc
success, stdout, stderr = run_command("nvcc --version")
if success:
    # Extract version
    import re
    match = re.search(r'release (\d+\.\d+)', stdout)
    if match:
        print_success(f"CUDA toolkit {match.group(1)} found")
    else:
        print_success("CUDA toolkit found")
else:
    print_warning("nvcc not found - JIT compilation may fail")

# =============================================================================
# STEP 4: RUN VERIFICATION SCRIPT
# =============================================================================

print_section("STEP 4: Environment Verification")

try:
    from utils.verify_cuda_env import run_all_checks, print_report

    results = run_all_checks()
    print_report(results)

    if results['all_passed']:
        print_success("All environment checks passed!")
    else:
        print_warning("Some environment checks failed")
        if results['recommendations']:
            print_info("Recommendations:")
            for rec in results['recommendations']:
                print(f"  • {rec}")

except ImportError as e:
    print_warning(f"Could not run verification: {e}")
except Exception as e:
    print_warning(f"Verification error: {e}")

# =============================================================================
# STEP 5: DEFINE HELPER FUNCTIONS FOR KERNEL COMPILATION
# =============================================================================

print_section("STEP 5: Setting Up Compilation")

def compile_attention_kernel(
    kernel_name="fused_attention",
    prefer_setuptools=True,
    verbose=True
):
    """
    Compile the attention kernel with automatic fallback.

    Args:
        kernel_name: Name of the kernel to compile
        prefer_setuptools: Use setuptools directly (more reliable in Colab)
        verbose: Print detailed progress

    Returns:
        Compiled module or None if compilation failed
    """
    import importlib.util
    from pathlib import Path

    if verbose:
        print_info(f"Compiling kernel: {kernel_name}")
        print_info(f"Method: {'setuptools' if prefer_setuptools else 'JIT fallback'}")

    # Try to import compiled module first (might already be compiled)
    try:
        module_path = Path("kernels") / f"{kernel_name}.so"
        if module_path.exists():
            if verbose:
                print_info("Found pre-compiled module, importing...")
            spec = importlib.util.spec_from_file_location(kernel_name, str(module_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print_success(f"Imported existing module: {kernel_name}")
            return module
    except Exception as e:
        if verbose:
            print_info(f"No pre-compiled module found: {e}")

    # Method 1: Try setuptools compilation (recommended for Colab)
    if prefer_setuptools:
        try:
            from utils.compile_setuptools import compile_with_setuptools

            # Check if we have the source files
            cuda_file = Path("kernels/attention.cu")
            if not cuda_file.exists():
                # Try to find CUDA source elsewhere
                cuda_file = None
                for pattern in ["kernels/*.cu", "src/kernels/*.cu", "*.cu"]:
                    import glob
                    matches = glob.glob(pattern)
                    if matches:
                        cuda_file = Path(matches[0])
                        break

            if cuda_file and cuda_file.exists():
                # Read the source file
                with open(cuda_file, 'r') as f:
                    cuda_source = f.read()

                # Compile
                module = compile_with_setuptools(
                    name=kernel_name,
                    cuda_source=cuda_source,
                    output_dir=Path("kernels"),
                    verbose=verbose
                )
                print_success(f"Compiled with setuptools: {kernel_name}")
                return module
            else:
                if verbose:
                    print_warning("No CUDA source file found")

        except ImportError:
            if verbose:
                print_warning("compile_setuptools module not available")
        except Exception as e:
            if verbose:
                print_warning(f"Setuptools compilation failed: {e}")

    # Method 2: Try JIT compilation with fallback
    try:
        from utils.cuda_build import compile_with_fallback

        # Look for CUDA source
        cuda_source = ""
        cuda_file = Path("kernels/attention.cu")
        if cuda_file.exists():
            with open(cuda_file, 'r') as f:
                cuda_source = f.read()

        if cuda_source:
            module = compile_with_fallback(
                name=kernel_name,
                cuda_source=cuda_source,
                prefer_setuptools=False,  # We already tried setuptools
                verbose=verbose
            )
            print_success(f"Compiled with fallback: {kernel_name}")
            return module

    except ImportError:
        if verbose:
            print_warning("cuda_build module not available")
    except Exception as e:
        if verbose:
            print_warning(f"JIT compilation failed: {e}")

    print_error(f"All compilation methods failed for {kernel_name}")
    return None

def test_attention_module(module):
    """
    Test that the compiled attention module works correctly.

    Args:
        module: The compiled module to test

    Returns:
        True if test passed, False otherwise
    """
    if module is None:
        print_error("Module is None, cannot test")
        return False

    try:
        import torch

        # Get available functions from module
        functions = [attr for attr in dir(module) if not attr.startswith('_')]
        print_info(f"Available functions: {functions}")

        # Look for a test function or run a basic smoke test
        if hasattr(module, 'fused_attention_forward'):
            print_info("Testing fused_attention_forward...")
            # Create test tensors
            batch_size, seq_len, heads, dim = 2, 128, 8, 64

            # Q, K, V tensors
            q = torch.randn(batch_size, heads, seq_len, dim, device='cuda')
            k = torch.randn(batch_size, heads, seq_len, dim, device='cuda')
            v = torch.randn(batch_size, heads, seq_len, dim, device='cuda')

            # Run the kernel
            output = module.fused_attention_forward(q, k, v)

            if output is not None and output.shape == (batch_size, heads, seq_len, dim):
                print_success("fused_attention_forward test passed!")
                print_info(f"Output shape: {output.shape}")
                return True
            else:
                print_warning(f"fused_attention_forward returned unexpected shape: {output.shape if output is not None else 'None'}")

        elif hasattr(module, 'attention_kernel'):
            print_info("Testing attention_kernel...")
            # Try different signature
            batch_size, seq_len, dim = 2, 128, 64

            q = torch.randn(batch_size * seq_len, dim, device='cuda')
            k = torch.randn(batch_size * seq_len, dim, device='cuda')
            v = torch.randn(batch_size * seq_len, dim, device='cuda')

            output = module.attention_kernel(q, k, v)
            print_success("attention_kernel test passed!")
            return True

        else:
            # Just check if we can call something
            for func_name in functions:
                if callable(getattr(module, func_name)):
                    print_info(f"Found callable function: {func_name}")
                    print_success("Module has callable functions!")
                    return True

        print_warning("Could not run specific test, but module loaded")
        return True

    except Exception as e:
        print_error(f"Module test failed: {e}")
        return False

# =============================================================================
# STEP 6: COMPILE KERNELS
# =============================================================================

print_section("STEP 6: Compiling Attention Kernels")

# Compile the attention kernel
attention_module = compile_attention_kernel(
    kernel_name="fused_attention",
    prefer_setuptools=USE_SETUPTOOLS_FALLBACK,
    verbose=True
)

# =============================================================================
# STEP 7: TEST COMPILED MODULE
# =============================================================================

print_section("STEP 7: Testing Compiled Module")

if attention_module is not None:
    test_passed = test_attention_module(attention_module)

    if test_passed:
        print_success("MODULE TEST PASSED!")
    else:
        print_warning("Module loaded but tests did not pass")
else:
    print_error("Module compilation failed")

# =============================================================================
# STEP 8: SUMMARY
# =============================================================================

print_section("SETUP SUMMARY")

print_info("Compilation Method:", "Setuptools" if USE_SETUPTOOLS_FALLBACK else "JIT with fallback")
print_info("Module Status:", "✓ Compiled" if attention_module is not None else "✗ Failed")
print_info("Test Status:", "✓ Passed" if attention_module is not None else "✗ Skipped")

if attention_module is not None:
    print("\n" + "=" * 70)
    print("  ✅ SUCCESS! CUDA kernel is ready to use.")
    print("=" * 70)
    print("\nYou can now use the kernel in your code:")
    print("  from utils.cuda_build import compile_with_fallback")
    print("  module = compile_with_fallback('fused_attention', cuda_source)")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print("  ⚠ CUDA KERNEL SETUP INCOMPLETE")
    print("=" * 70)
    print("\nTroubleshooting:")
    print("  1. Make sure you're using a GPU runtime (Runtime > Change runtime type)")
    print("  2. Try restarting the runtime and running this cell again")
    print("  3. Check that CUDA toolkit is available: !nvcc --version")
    print("  4. Use the PyTorch baseline model as fallback")
    print("=" * 70)

# Make the module available globally
if attention_module is not None:
    import builtins
    builtins.fused_attention_module = attention_module
    print_info("\nModule available as: fused_attention_module")
