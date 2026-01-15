"""
StyleForge - CUDA Build Utilities

Utilities for compiling and testing CUDA kernels with PyTorch.
"""

import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from torch.utils.cpp_extension import load_inline


def get_cuda_info() -> Dict[str, Any]:
    """
    Get CUDA system information and recommended compiler flags.

    Returns:
        Dictionary with CUDA version, compute capability, and build flags
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
    }

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        info['compute_capability'] = f"{major}.{minor}"
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_count'] = torch.cuda.device_count()

        # Determine optimal arch flags
        arch_flags = _get_arch_flags(major, minor)
        info['arch_flags'] = arch_flags

        # Base optimization flags
        info['base_flags'] = [
            '-O3',
            '--use_fast_math',
            '-lineinfo',
            '--expt-relaxed-constexpr',
        ]

        # Combined flags
        info['extra_cuda_cflags'] = info['base_flags'] + arch_flags
        info['extra_cxx_cflags'] = ['-O3']

    return info


def _get_arch_flags(major: int, minor: int) -> List[str]:
    """
    Get architecture-specific compiler flags based on compute capability.

    Args:
        major: Major version of compute capability
        minor: Minor version of compute capability

    Returns:
        List of -gencode flags for nvcc
    """
    arch_flags = []

    # Common architectures (from Volta onwards)
    if major >= 7:
        arch_flags.append('-gencode=arch=compute_70,code=sm_70')  # V100

    # Turing (RTX 20xx, GTX 16xx)
    if major >= 7 or (major == 7 and minor >= 5):
        arch_flags.append('-gencode=arch=compute_75,code=sm_75')

    # Ampere (A100, RTX 30xx)
    if major >= 8:
        arch_flags.append('-gencode=arch=compute_80,code=sm_80')  # A100
        arch_flags.append('-gencode=arch=compute_86,code=sm_86')  # RTX 30xx

    # Ada Lovelace (RTX 40xx)
    if major >= 9 or (major == 8 and minor >= 9):
        arch_flags.append('-gencode=arch=compute_89,code=sm_89')  # RTX 40xx

    # Hopper (H100)
    if major >= 9:
        arch_flags.append('-gencode=arch=compute_90,code=sm_90')  # H100

    return arch_flags


def save_build_config(config: Dict[str, Any], filepath: Optional[Path] = None):
    """
    Save build configuration to JSON file.

    Args:
        config: Configuration dictionary from get_cuda_info()
        filepath: Path to save config (default: build/build_config.json)
    """
    if filepath is None:
        filepath = Path('build/build_config.json')
    else:
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_build_config(filepath: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load build configuration from JSON file.

    Args:
        filepath: Path to config file (default: build/build_config.json)

    Returns:
        Configuration dictionary
    """
    if filepath is None:
        filepath = Path('build/build_config.json')
    else:
        filepath = Path(filepath)

    with open(filepath, 'r') as f:
        return json.load(f)


def compile_inline(
    name: str,
    cuda_source: str,
    cpp_source: str = '',
    functions: Optional[List[str]] = None,
    build_directory: Optional[Path] = None,
    verbose: bool = False,
    with_pybind11: bool = True
):
    """
    Compile CUDA code inline using PyTorch's JIT compilation.

    Args:
        name: Name for the compiled module
        cuda_source: CUDA source code (.cu file contents)
        cpp_source: Optional C++ source code
        functions: List of function names to expose
        build_directory: Directory for build artifacts
        verbose: Whether to print compilation output
        with_pybind11: Whether to use pybind11

    Returns:
        Compiled Python module
    """
    if build_directory is None:
        build_directory = Path('build')
    else:
        build_directory = Path(build_directory)

    build_directory.mkdir(parents=True, exist_ok=True)

    # Get build flags
    cuda_info = get_cuda_info()
    extra_cuda_cflags = cuda_info.get('extra_cuda_cflags', ['-O3'])
    extra_cxx_cflags = cuda_info.get('extra_cxx_cflags', ['-O3'])

    module = load_inline(
        name=name,
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=functions or [],
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=str(build_directory),
        verbose=verbose,
        with_pybind11=with_pybind11
    )

    return module


def verify_cuda_installation() -> tuple[bool, str]:
    """
    Verify CUDA installation and return status.

    Returns:
        Tuple of (is_available, status_message)
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available. Please check your PyTorch installation."

    try:
        # Test basic CUDA operation
        x = torch.randn(10).cuda()
        y = torch.randn(10).cuda()
        z = x + y
        torch.cuda.synchronize()

        major, minor = torch.cuda.get_device_capability(0)
        return True, f"CUDA {torch.version.cuda}, Compute Capability {major}.{minor}, Device: {torch.cuda.get_device_name(0)}"

    except Exception as e:
        return False, f"CUDA test failed: {str(e)}"


def print_cuda_info():
    """Print detailed CUDA system information."""
    print("\n" + "=" * 70)
    print("  CUDA System Information")
    print("=" * 70)

    info = get_cuda_info()

    print(f"  CUDA Available:       {info['cuda_available']}")
    print(f"  CUDA Version:         {info.get('cuda_version', 'N/A')}")
    print(f"  PyTorch Version:      {info.get('pytorch_version', 'N/A')}")

    if info['cuda_available']:
        print(f"  Device Name:          {info.get('device_name', 'N/A')}")
        print(f"  Compute Capability:   {info.get('compute_capability', 'N/A')}")
        print(f"  Device Count:         {info.get('device_count', 'N/A')}")

        print("\n  Recommended CUDA Flags:")
        for flag in info.get('extra_cuda_cflags', []):
            print(f"    {flag}")

    print("=" * 70 + "\n")
