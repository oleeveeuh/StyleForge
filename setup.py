"""StyleForge setup script for building CUDA extensions"""

import os
import torch
from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, setup, BuildExtension

# Root directory
ROOT = Path(__file__).parent

# CUDA source files
CUDA_SOURCES = [
    "kernels/attention.cu",
    "kernels/attention_v2.cu",
    "kernels/ffn.cu",
    "kernels/instance_norm.cu",
]

# Check if CUDA files exist
existing_sources = [str(ROOT / src) for src in CUDA_SOURCES if (ROOT / src).exists()]

# Get compute capability for current GPU
compute_capabilities = []
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    cc = f"{major}{minor}"
    compute_capabilities.append(f"arch=compute_{cc},code=sm_{cc}")

    # Add common architectures for compatibility
    if major >= 7:
        compute_capabilities.append("arch=compute_70,code=sm_70")  # V100
        compute_capabilities.append("arch=compute_75,code=sm_75")  # T4, RTX 20xx
    if major >= 8:
        compute_capabilities.append("arch=compute_80,code=sm_80")  # A100
        compute_capabilities.append("arch=compute_86,code=sm_86")  # RTX 30xx
else:
    # Defaults for systems without GPU (build will still work, just can't run)
    compute_capabilities = [
        "arch=compute_75,code=sm_75",
        "arch=compute_80,code=sm_80",
        "arch=compute_86,code=sm_86",
    ]

# Build nvcc args
nvcc_args = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]
nvcc_args.extend([f"-gencode={cc}" for cc in compute_capabilities])

ext_modules = []

if existing_sources:
    print(f"Building CUDA extension with {len(existing_sources)} source files...")
    for src in existing_sources:
        print(f"  - {src}")

    ext_modules.append(
        CUDAExtension(
            name="styleforge_cuda",
            sources=existing_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_args,
            },
        )
    )

setup(
    name="styleforge",
    version="0.1.0",
    description="Real-time neural style transfer with CUDA kernels",
    author="Olivia Liau",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
