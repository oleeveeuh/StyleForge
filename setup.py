"""StyleForge setup script for building CUDA extensions"""

import os
from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, setup

# Root directory
ROOT = Path(__file__).parent

# CUDA source files
CUDA_SOURCES = [
    "kernels/attention.cu",
    "kernels/ffn.cu",
    "kernels/instance_norm.cu",
    "kernels/style_transfer.cu",
]

# Check if CUDA files exist
existing_sources = [str(ROOT / src) for src in CUDA_SOURCES if (ROOT / src).exists()]

ext_modules = []

if existing_sources:
    ext_modules.append(
        CUDAExtension(
            name="styleforge_cuda",
            sources=existing_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_80,code=sm_80",  # A100
                    "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
                    "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
                    "-use_fast_math",
                ],
            },
        )
    )

setup(
    name="styleforge",
    version="0.1.0",
    description="Real-time neural style transfer with CUDA kernels",
    author="Olivia Liau",
    ext_modules=ext_modules,
    cmdclass={"build_ext": "torch.utils.cpp_extension.BuildExtension"},
)
