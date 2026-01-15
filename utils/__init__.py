"""StyleForge utils module"""

from .cuda_build import (
    compile_inline,
    get_cuda_info,
    load_build_config,
    print_cuda_info,
    save_build_config,
    verify_cuda_installation,
)
from .profiling import KernelProfiler, benchmark_with_profiler, profile_attention_comparison, save_profiling_results

__all__ = [
    "compile_inline",
    "get_cuda_info",
    "load_build_config",
    "print_cuda_info",
    "save_build_config",
    "verify_cuda_installation",
    "KernelProfiler",
    "benchmark_with_profiler",
    "profile_attention_comparison",
    "save_profiling_results",
]

__version__ = "0.1.0"
