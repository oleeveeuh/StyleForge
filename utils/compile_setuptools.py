"""
StyleForge - Setuptools-based CUDA Compilation

Fallback compilation method using setuptools when PyTorch's load_inline()
fails in Google Colab or other restricted environments.

This module provides a more reliable compilation method that:
1. Writes a temporary setup.py file
2. Runs setuptools build_ext
3. Locates and imports the compiled .so file
"""

import os
import sys
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager

# Try to import colorama for colored output
try:
    import colorama
    colorama.init()
except ImportError:
    colorama = None


# Color codes (ANSI)
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def blue(s): return f"{Colors.OKBLUE}{s}{Colors.ENDC}"
    @staticmethod
    def cyan(s): return f"{Colors.OKCYAN}{s}{Colors.ENDC}"
    @staticmethod
    def green(s): return f"{Colors.OKGREEN}{s}{Colors.ENDC}"
    @staticmethod
    def warning(s): return f"{Colors.WARNING}{s}{Colors.ENDC}"
    @staticmethod
    def fail(s): return f"{Colors.FAIL}{s}{Colors.ENDC}"
    @staticmethod
    def bold(s): return f"{Colors.BOLD}{s}{Colors.ENDC}"


def _use_colors() -> bool:
    """Check if colors should be used in output."""
    if colorama is not None:
        return True
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def _print_header(text: str):
    """Print a formatted header."""
    if _use_colors():
        print(f"\n{Colors.bold(Colors.blue('═' * 70))}")
        print(f"{Colors.bold(text)}")
        print(f"{Colors.bold(Colors.blue('═' * 70))}")
    else:
        print(f"\n{'=' * 70}")
        print(f"  {text}")
        print(f"{'=' * 70}")


def _print_step(step: str, status: str = "", use_colors: bool = True):
    """Print a step with optional status."""
    icons = {
        'compiling': '🔨',
        'running': '⚙️ ',
        'success': '✓',
        'import': '📦',
        'error': '✗',
        'writing': '📝',
        'building': '🏗️ ',
    }
    icon = icons.get(step.split()[0].lower(), '•')
    if "compiling" in step.lower():
        icon = '🔨'
    elif "running" in step.lower():
        icon = '⚙️ '
    elif "success" in step.lower() or "ready" in step.lower():
        icon = '✓'
    elif "import" in step.lower():
        icon = '📦'
    elif "error" in step.lower() or "failed" in step.lower():
        icon = '✗'
    elif "writing" in step.lower():
        icon = '📝'

    if use_colors and _use_colors():
        if status:
            print(f"{icon} {step}: {Colors.green(status)}")
        else:
            print(f"{icon} {Colors.cyan(step)}")
    else:
        if status:
            print(f"{icon} {step}: {status}")
        else:
            print(f"{icon} {step}")


def _get_cuda_arch_flags() -> List[str]:
    """
    Get CUDA architecture flags based on available GPU.
    Returns list of -gencode flags for nvcc.
    """
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            # Generate appropriate gencode flags
            flags = [
                f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}',
                f'-gencode=arch=compute_{major}{minor},code=compute_{major}{minor}',
            ]
            return flags
    except Exception:
        pass

    # Default flags if CUDA detection fails
    return [
        '-gencode=arch=compute_70,code=sm_70',
        '-gencode=arch=compute_75,code=sm_75',
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_86,code=sm_86',
    ]


def _generate_setup_py(
    name: str,
    cuda_file: Path,
    cpp_file: Optional[Path] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Generate setup.py content for CUDA extension compilation.

    Args:
        name: Name of the extension module
        cuda_file: Path to the .cu source file
        cpp_file: Optional path to the .cpp source file
        extra_cuda_cflags: Additional CUDA compiler flags
        output_dir: Directory for build output

    Returns:
        setup.py content as string
    """
    cuda_file = Path(cuda_file).absolute()
    sources = [f'"{cuda_file}"']
    if cpp_file:
        sources.append(f'"{Path(cpp_file).absolute()}"')

    sources_str = ', '.join(sources)

    # Get architecture flags
    arch_flags = _get_cuda_arch_flags()
    arch_flags_str = ', '.join([f'"{flag}"' for flag in arch_flags])

    # Extra flags
    if extra_cuda_cflags:
        extra_flags_str = ', '.join([f'"{flag}"' for flag in extra_cuda_cflags])
    else:
        extra_flags_str = ''

    # Generate setup.py
    setup_content = f'''
import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# PyTorch include and library paths
try:
    from torch.utils.cpp_extension import include_paths, library_paths
    include_dirs = include_paths("cuda")
    library_dirs = library_paths("cuda")
except ImportError:
    # Fallback for older PyTorch versions
    import torch
    torch_path = os.path.dirname(os.path.dirname(torch.__file__))
    include_dirs = [os.path.join(torch_path, 'include')]
    library_dirs = [os.path.join(torch_path, 'lib')]

# Extra CUDA flags
extra_cuda_cflags = [{arch_flags_str}, "-O3", "--use_fast_math"]
{f'extra_cuda_cflags.extend([{extra_flags_str}])' if extra_flags_str else ''}

# Create the extension
ext = CUDAExtension(
    name="{name}",
    sources=[{sources_str}],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    extra_compile_args={{"cxx": ["-O3"], "nvcc": extra_cuda_cflags}},
)

# Setup
setup(
    name="{name}",
    ext_modules=[ext],
    cmdclass={{"build_ext": BuildExtension}},
)
'''
    return setup_content


@contextmanager
def _temp_build_directory(build_dir: Optional[Path] = None):
    """
    Context manager for temporary build directory.

    Args:
        build_dir: If specified, use this directory instead of temp

    Yields:
        Path to the build directory
    """
    original_dir = os.getcwd()
    temp_dir = None

    try:
        if build_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix='cuda_build_'))
            build_dir = temp_dir
        else:
            build_dir = Path(build_dir)
            build_dir.mkdir(parents=True, exist_ok=True)

        os.chdir(build_dir)
        yield build_dir

    finally:
        os.chdir(original_dir)
        # Don't clean up if a specific build directory was provided
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


def _find_compiled_so(build_dir: Path, module_name: str) -> Optional[Path]:
    """
    Find the compiled .so file in the build directory.

    Args:
        build_dir: Directory to search
        module_name: Name of the module (without .so extension)

    Returns:
        Path to the .so file, or None if not found
    """
    import glob

    # Common locations where setuptools puts the .so file
    search_patterns = [
        build_dir / f"{module_name}*.so",
        build_dir / "build" / "lib.*" / f"{module_name}*.so",
        build_dir / "lib" / f"{module_name}*.so",
    ]

    for pattern_base in search_patterns:
        # Expand glob patterns
        pattern = str(pattern_base).replace("lib.*", "lib")
        for actual_pattern in [
            str(pattern_base),
            str(pattern_base).replace("lib.*", "lib.linux-x86_64"),
            str(pattern_base).replace("lib.*", "lib.linux-aarch64"),
        ]:
            matches = glob.glob(actual_pattern, recursive=True)
            for match in matches:
                if Path(match).is_file() and (match.endswith('.so') or match.endswith('.pyd')):
                    return Path(match)

    # Recursive search as last resort
    for root, dirs, files in os.walk(build_dir):
        for f in files:
            if f.startswith(module_name) and (f.endswith('.so') or f.endswith('.pyd')):
                return Path(root) / f

    return None


def compile_with_setuptools(
    name: str,
    cuda_source: str,
    cpp_source: str = '',
    output_dir: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    keep_sources: bool = False,
    extra_cuda_cflags: Optional[List[str]] = None,
    verbose: bool = True,
) -> Any:
    """
    Compile CUDA kernel using setuptools instead of load_inline.

    This method is more reliable in Google Colab when load_inline() fails.

    Args:
        name: Name for the compiled module
        cuda_source: CUDA source code as string
        cpp_source: Optional C++ source code as string
        output_dir: Directory where .so file should be copied to
        build_dir: Directory for build (default: temp directory)
        keep_sources: If True, keep source files after compilation
        extra_cuda_cflags: Additional CUDA compiler flags
        verbose: Whether to print detailed progress

    Returns:
        Compiled Python module

    Raises:
        RuntimeError: If compilation fails or module cannot be imported

    Example:
        >>> from utils.compile_setuptools import compile_with_setuptools
        >>> cuda_code = '''
        ... __global__ void add_kernel(float* c, float* a, float* b, int n) {
        ...     int i = blockIdx.x * blockDim.x + threadIdx.x;
        ...     if (i < n) c[i] = a[i] + b[i];
        ... }
        ... '''
        >>> module = compile_with_setuptools("my_kernel", cuda_code)
    """
    import torch
    import importlib.util

    is_colab = 'google.colab' in sys.modules
    use_colors = _use_colors()

    if verbose:
        _print_header(f"Setuptools Compilation: {name}")
        _print_step("Environment", use_colors=use_colors)
        if use_colors:
            print(f"    PyTorch:       {Colors.cyan(torch.__version__)}")
            print(f"    CUDA:          {Colors.cyan(torch.version.cuda or 'N/A')}")
            print(f"    Environment:   {Colors.green('Colab' if is_colab else 'Local')}")
        else:
            print(f"    PyTorch:       {torch.__version__}")
            print(f"    CUDA:          {torch.version.cuda or 'N/A'}")
            print(f"    Environment:   {'Colab' if is_colab else 'Local'}")

    # Determine output directory
    if output_dir is None:
        if is_colab:
            output_dir = Path('/content/StyleForge/kernels')
        else:
            output_dir = Path('build/kernels')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine build directory
    if build_dir is None:
        if is_colab:
            build_dir = Path('/tmp/styleforge_setuptools_build')
        else:
            build_dir = Path('build/setuptools')
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        _print_step("Preparing source files", use_colors=use_colors)
        if use_colors:
            print(f"    Build dir:     {Colors.cyan(str(build_dir))}")
            print(f"    Output dir:    {Colors.cyan(str(output_dir))}")
        else:
            print(f"    Build dir:     {build_dir}")
            print(f"    Output dir:    {output_dir}")

    # Write source files
    cuda_file = build_dir / f"{name}.cu"
    cpp_file = None

    with open(cuda_file, 'w') as f:
        f.write(cuda_source)

    if cpp_source:
        # Use a different name for cpp file to avoid ninja build conflicts
        # Both .cu and .cpp files with same base name would compile to same .o file
        cpp_file = build_dir / f"{name}_bindings.cpp"
        with open(cpp_file, 'w') as f:
            f.write(cpp_source)

    if verbose:
        cuda_lines = cuda_source.count('\n') + 1
        cpp_lines = cpp_source.count('\n') + 1 if cpp_source else 0
        if use_colors:
            print(f"    CUDA source:   {Colors.bold(f'{cuda_lines} lines')}")
            if cpp_source:
                print(f"    C++ source:    {Colors.bold(f'{cpp_lines} lines')}")
        else:
            print(f"    CUDA source:   {cuda_lines} lines")
            if cpp_source:
                print(f"    C++ source:    {cpp_lines} lines")

    # Generate and write setup.py
    _print_step("Generating setup.py", use_colors=use_colors)
    setup_content = _generate_setup_py(
        name=name,
        cuda_file=cuda_file,
        cpp_file=cpp_file,
        extra_cuda_cflags=extra_cuda_cflags,
        output_dir=output_dir,
    )

    setup_file = build_dir / 'setup.py'
    with open(setup_file, 'w') as f:
        f.write(setup_content)

    if verbose:
        if use_colors:
            print(f"    {Colors.green('✓ setup.py created')}")
        else:
            print(f"    ✓ setup.py created")

    # Run setup.py build_ext
    _print_step("Building extension", "running setuptools...", use_colors=use_colors)

    import time
    start_time = time.time()

    try:
        # Use subprocess to run setup.py
        cmd = [sys.executable, 'setup.py', 'build_ext']
        if verbose:
            cmd.append('--verbose')

        result = subprocess.run(
            cmd,
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        elapsed = time.time() - start_time

        if verbose:
            if use_colors:
                print(f"    Time elapsed:  {Colors.bold(f'{elapsed:.2f}s')}")
            else:
                print(f"    Time elapsed:  {elapsed:.2f}s")

        if result.returncode != 0:
            if verbose:
                if use_colors:
                    print(f"    {Colors.fail('✗ Build failed')}")
                else:
                    print(f"    ✗ Build failed")
                print("\n--- STDOUT ---")
                print(result.stdout)
                print("\n--- STDERR ---")
                print(result.stderr)

            raise RuntimeError(
                f"setuptools build failed with exit code {result.returncode}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Compilation timed out after 5 minutes")
    except Exception as e:
        raise RuntimeError(f"Compilation failed: {e}")

    # Find the compiled .so file
    _print_step("Locating compiled module", use_colors=use_colors)

    so_path = _find_compiled_so(build_dir, name)

    if so_path is None or not so_path.exists():
        # List all files in build directory for debugging
        files = []
        for root, dirs, filenames in os.walk(build_dir):
            for f in filenames:
                if f.endswith('.so') or f.endswith('.pyd'):
                    files.append(str(Path(root) / f))

        raise RuntimeError(
            f"Could not find compiled .so file for module '{name}'.\n"
            f"Build directory: {build_dir}\n"
            f"Files found: {files if files else 'None'}"
        )

    if verbose:
        file_size = so_path.stat().st_size
        if use_colors:
            print(f"    Found:         {Colors.green('✓')}")
            print(f"    Location:      {Colors.cyan(str(so_path))}")
            print(f"    Size:          {Colors.bold(f'{file_size / 1024 / 1024:.2f} MB')}")
        else:
            print(f"    Found:         ✓")
            print(f"    Location:      {so_path}")
            print(f"    Size:          {file_size / 1024 / 1024:.2f} MB")

    # Copy to output directory
    output_so_path = output_dir / f"{name}.so"
    shutil.copy2(so_path, output_so_path)

    if verbose:
        if use_colors:
            print(f"    Copied to:     {Colors.cyan(str(output_so_path))}")
        else:
            print(f"    Copied to:     {output_so_path}")

    # Import the module
    _print_step("Importing module", use_colors=use_colors)

    try:
        # Add output directory to Python path
        output_dir_str = str(output_dir.absolute())
        if output_dir_str not in sys.path:
            sys.path.insert(0, output_dir_str)

        # Import the module
        spec = importlib.util.spec_from_file_location(name, str(output_so_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create module spec for {name}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    except Exception as e:
        raise RuntimeError(
            f"Failed to import compiled module '{name}' from {output_so_path}\n"
            f"Error: {e}"
        )

    if verbose:
        # Show available functions
        try:
            available_funcs = [attr for attr in dir(module) if not attr.startswith('_')]
            if available_funcs:
                func_str = ", ".join(available_funcs[:8])
                if len(available_funcs) > 8:
                    func_str += f" ... ({len(available_funcs)} total)"
                if use_colors:
                    print(f"    Available:     {Colors.green(func_str)}")
                else:
                    print(f"    Available:     {func_str}")
        except Exception:
            pass

    # Clean up source files if requested
    if not keep_sources and verbose:
        _print_step("Cleanup", use_colors=use_colors)
        # Keep the .so file and setup.py, remove source files
        if cuda_file.exists():
            cuda_file.unlink()
        if cpp_file and cpp_file.exists():
            cpp_file.unlink()
        if use_colors:
            print(f"    {Colors.green('✓ Source files removed')}")
        else:
            print(f"    ✓ Source files removed")

    if verbose:
        if use_colors:
            print(f"\n{Colors.bold(Colors.green('✅ Module ready!'))}")
        else:
            print(f"\n✅ Module ready!")

    return module


def compile_with_setuptools_fallback(
    name: str,
    cuda_source: str,
    cpp_source: str = '',
    functions: Optional[List[str]] = None,
    build_directory: Optional[Path] = None,
    verbose: bool = True,
    **kwargs
) -> Any:
    """
    Attempt JIT compilation first, fall back to setuptools if it fails.

    This is the recommended way to compile CUDA kernels in Colab.

    Args:
        name: Name for the compiled module
        cuda_source: CUDA source code
        cpp_source: Optional C++ source code
        functions: List of function names to expose (for JIT only)
        build_directory: Optional build directory
        verbose: Whether to print progress
        **kwargs: Additional arguments for setuptools compilation

    Returns:
        Compiled Python module

    Example:
        >>> from utils.compile_setuptools import compile_with_setuptools_fallback
        >>> module = compile_with_setuptools_fallback(
        ...     name="fused_attention",
        ...     cuda_source=cuda_code,
        ...     cpp_source=cpp_code
        ... )
    """
    from utils.cuda_build import compile_inline

    if verbose:
        _print_header("CUDA Compilation with Auto-Fallback")

    # First, try PyTorch's JIT compilation
    if verbose:
        _print_step("Attempting JIT compilation (load_inline)", use_colors=_use_colors())

    jit_success = False
    module = None

    try:
        module = compile_inline(
            name=name,
            cuda_source=cuda_source,
            cpp_source=cpp_source,
            functions=functions,
            build_directory=build_directory,
            verbose=verbose,
        )
        jit_success = True
        if verbose:
            if _use_colors():
                print(f"    {Colors.green('✓ JIT compilation succeeded!')}")
            else:
                print(f"    ✓ JIT compilation succeeded!")
    except Exception as e:
        if verbose:
            if _use_colors():
                print(f"    {Colors.warning('⚠ JIT compilation failed')}")
                print(f"    Error: {str(e)[:100]}...")
            else:
                print(f"    ⚠ JIT compilation failed")
                print(f"    Error: {str(e)[:100]}...")

    # If JIT failed, try setuptools
    if not jit_success:
        if verbose:
            _print_step("Falling back to setuptools compilation", use_colors=_use_colors())

        module = compile_with_setuptools(
            name=name,
            cuda_source=cuda_source,
            cpp_source=cpp_source,
            output_dir=build_directory,
            verbose=verbose,
        )

    return module


# CLI interface
def main():
    """Command-line interface for setuptools compilation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compile CUDA kernels using setuptools (fallback for Colab)'
    )
    parser.add_argument('name', help='Module name')
    parser.add_argument('cuda_file', type=Path, help='Path to .cu source file')
    parser.add_argument('--cpp-file', type=Path, help='Path to .cpp source file')
    parser.add_argument('--output-dir', type=Path, default='build/kernels',
                       help='Output directory for .so file')
    parser.add_argument('--build-dir', type=Path, help='Build directory')
    parser.add_argument('--keep-sources', action='store_true',
                       help='Keep source files after compilation')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress')

    args = parser.parse_args()

    # Read source files
    with open(args.cuda_file, 'r') as f:
        cuda_source = f.read()

    cpp_source = ''
    if args.cpp_file:
        with open(args.cpp_file, 'r') as f:
            cpp_source = f.read()

    # Compile
    try:
        module = compile_with_setuptools(
            name=args.name,
            cuda_source=cuda_source,
            cpp_source=cpp_source,
            output_dir=args.output_dir,
            build_dir=args.build_dir,
            keep_sources=args.keep_sources,
            verbose=args.verbose,
        )
        print(f"\n✅ Successfully compiled and imported: {args.name}")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Compilation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
