"""
Minimal CUDA build utilities for Hugging Face Spaces
"""

import torch
from pathlib import Path
from typing import List, Optional
from torch.utils.cpp_extension import load_inline

# Global module cache
_COMPILED_MODULES = {}


def compile_inline(
    name: str,
    cuda_source: str,
    cpp_source: str = '',
    functions: Optional[List[str]] = None,
    build_directory: Optional[Path] = None,
    verbose: bool = False,
) -> any:
    """
    Compile CUDA code inline using PyTorch's JIT compilation.
    """
    import time

    if name in _COMPILED_MODULES:
        return _COMPILED_MODULES[name]

    if verbose:
        print(f"Compiling {name}...")

    start_time = time.time()

    # Get CUDA build flags
    cuda_info = get_cuda_info()
    extra_cuda_cflags = cuda_info.get('extra_cuda_cflags', ['-O3'])

    try:
        # Try with with_pybind11 (newer PyTorch)
        try:
            module = load_inline(
                name=name,
                cpp_sources=[cpp_source] if cpp_source else [],
                cuda_sources=[cuda_source] if cuda_source else [],
                extra_cuda_cflags=extra_cuda_cflags,
                verbose=verbose,
                with_pybind11=True
            )
        except TypeError:
            # Fall back to older PyTorch API
            module = load_inline(
                name=name,
                cpp_sources=[cpp_source] if cpp_source else [],
                cuda_sources=[cuda_source] if cuda_source else [],
                extra_cuda_cflags=extra_cuda_cflags,
                verbose=verbose,
            )

        elapsed = time.time() - start_time

        if verbose:
            print(f"{name} compiled successfully in {elapsed:.2f}s")

        _COMPILED_MODULES[name] = module
        return module

    except Exception as e:
        if verbose:
            print(f"Failed to compile {name}: {e}")
        raise


def get_cuda_info() -> dict:
    """Get CUDA system information."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
    }

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        info['compute_capability'] = f"{major}.{minor}"
        info['device_name'] = torch.cuda.get_device_name(0)

        # Architecture-specific flags
        extra_cuda_cflags = ['-O3', '--use_fast_math']

        # Common architectures
        if major >= 7:
            extra_cuda_cflags.append('-gencode=arch=compute_70,code=sm_70')
        if major >= 7 or (major == 7 and minor >= 5):
            extra_cuda_cflags.append('-gencode=arch=compute_75,code=sm_75')
        if major >= 8:
            extra_cuda_cflags.append('-gencode=arch=compute_80,code=sm_80')
            extra_cuda_cflags.append('-gencode=arch=compute_86,code=sm_86')
        if major >= 9 or (major == 8 and minor >= 9):
            extra_cuda_cflags.append('-gencode=arch=compute_89,code=sm_89')

        info['extra_cuda_cflags'] = extra_cuda_cflags

    else:
        info['extra_cuda_cflags'] = ['-O3']

    return info
