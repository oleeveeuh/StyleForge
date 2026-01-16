"""
StyleForge - CUDA Build Utilities

Utilities for compiling and testing CUDA kernels with PyTorch.
"""

import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from torch.utils.cpp_extension import load_inline

# Optional: colorama for better color support on Windows
try:
    import colorama
    colorama.init()
except ImportError:
    colorama = None  # Colors will still work via ANSI codes on Unix


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


def _get_pytorch_build_root() -> str:
    """
    Get the root directory where PyTorch stores compiled extensions.

    Returns:
        Path to torch extensions cache directory
    """
    import os
    import sys

    # Check for TORCH_EXTENSIONS_DIR environment variable
    root_extensions_directory = os.environ.get('TORCH_EXTENSIONS_DIR')
    if root_extensions_directory is None:
        root_extensions_directory = torch._appdirs.user_cache_dir(appname='torch_extensions')

    # Determine accelerator prefix
    if torch.version.hip is not None:
        accelerator_str = f'rocm{torch.version.hip.replace(".", "")}'
    elif torch.version.cuda is not None:
        accelerator_str = f'cu{torch.version.cuda.replace(".", "")}'
    else:
        accelerator_str = 'cpu'

    python_version = f'py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, "abiflags", "")}'
    build_folder = f'{python_version}_{accelerator_str}'

    return os.path.join(root_extensions_directory, build_folder)


def find_so_file(module_name: str, search_root: Optional[str] = None) -> Optional[Path]:
    """
    Find the compiled .so file for a given module name.

    PyTorch's JIT compilation writes .so files to a versioned subdirectory
    (e.g., module_v1, module_v2). This function searches for the actual .so file.

    Args:
        module_name: Name of the module to find
        search_root: Root directory to search (defaults to PyTorch extensions cache)

    Returns:
        Path to the .so file, or None if not found
    """
    import os
    import glob

    if search_root is None:
        search_root = _get_pytorch_build_root()

    if not os.path.exists(search_root):
        return None

    # The module directory could be directly named or versioned (module_v1, module_v2, etc.)
    possible_dirs = [
        os.path.join(search_root, module_name),
        # Also search for versioned directories
    ]

    # Find all directories matching the module name pattern
    pattern = os.path.join(search_root, f"{module_name}*")
    for dir_path in glob.glob(pattern):
        if os.path.isdir(dir_path):
            possible_dirs.append(dir_path)

    # Search for .so files in these directories
    for dir_path in possible_dirs:
        if not os.path.exists(dir_path):
            continue

        # Look for .so files (or .pyd on Windows)
        so_pattern = os.path.join(dir_path, f"{module_name}*.so")
        so_files = glob.glob(so_pattern)

        # Also check for common Python extension patterns
        # PyTorch may create files like: module_name.cpython-310-x86_64-linux-gnu.so
        for root, _, files in os.walk(dir_path):
            for f in files:
                if f.startswith(module_name) and (f.endswith('.so') or f.endswith('.pyd')):
                    return Path(root) / f

        if so_files:
            # Return the most recently modified .so file
            return Path(max(so_files, key=os.path.getmtime))

    return None


def _get_color_support():
    """Check if color output is supported."""
    try:
        import colorama
        colorama.init()
        return True
    except ImportError:
        # Fallback: check if we're in a terminal that supports ANSI codes
        import sys
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


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
    UNDERLINE = '\033[4m'

    @staticmethod
    def header(s): return f"{Colors.HEADER}{s}{Colors.ENDC}"
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


def _print_header(text: str, use_colors: bool = True):
    """Print a formatted header."""
    if use_colors:
        print(f"\n{Colors.bold(Colors.blue('═' * 70))}")
        print(f"{Colors.bold(text)}")
        print(f"{Colors.bold(Colors.blue('═' * 70))}")
    else:
        print(f"\n{'=' * 70}")
        print(f"  {text}")
        print(f"{'=' * 70}")


def _print_step(step: str, status: str = "", use_colors: bool = True):
    """Print a step with optional status."""
    icon = "🔨"
    if "compiling" in step.lower():
        icon = "🔨"
    elif "running" in step.lower():
        icon = "⚙️ "
    elif "success" in step.lower() or "ready" in step.lower():
        icon = "✓"
    elif "import" in step.lower():
        icon = "📦"
    elif "error" in step.lower() or "failed" in step.lower():
        icon = "✗"

    if use_colors:
        if status:
            print(f"{icon} {step}: {Colors.green(status)}")
        else:
            print(f"{icon} {Colors.cyan(step)}")
    else:
        if status:
            print(f"{icon} {step}: {status}")
        else:
            print(f"{icon} {step}")


def _format_size(size_bytes: int) -> str:
    """Format byte size to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def _count_functions(source: str) -> List[str]:
    """Extract function names from CUDA/C++ source code."""
    import re
    # Match function declarations: return_type function_name(params)
    # Skip common headers and macros
    pattern = r'\b(?:__global__|__device__|__host__|extern|static)?\s*(?:\w+)\s+(\w+)\s*\('
    functions = re.findall(pattern, source)
    # Filter out common non-function names
    exclude = {'if', 'while', 'for', 'switch', 'catch', 'sizeof'}
    return [f for f in functions if f not in exclude and not f.startswith('_')]


def compile_inline(
    name: str,
    cuda_source: str,
    cpp_source: str = '',
    functions: Optional[List[str]] = None,
    build_directory: Optional[Path] = None,
    verbose: bool = True,
    create_symlink: bool = True,
):
    """
    Compile CUDA code inline using PyTorch's JIT compilation.

    This function works WITH PyTorch's caching system rather than against it.
    In Colab, it lets PyTorch use its default cache directory, then optionally
    creates a symlink to a custom build/ directory for convenience.

    Args:
        name: Name for the compiled module
        cuda_source: CUDA source code (.cu file contents)
        cpp_source: Optional C++ source code
        functions: List of function names to expose
        build_directory: Optional directory to create symlinks to compiled .so files.
            If None and in Colab, symlinks are created in ./build/ for convenience.
            The actual compilation happens in PyTorch's cache regardless.
        verbose: Whether to print compilation output (default: True)
        create_symlink: Whether to create symlinks to the compiled .so file (default: True)

    Returns:
        Compiled Python module

    Raises:
        RuntimeError: If compilation fails or the .so file cannot be loaded
    """
    import os
    import sys
    import time

    use_colors = _get_color_support()
    is_colab = 'google.colab' in sys.modules

    # ========================================================================
    # STEP 1: PRE-COMPILATION ANALYSIS
    # ========================================================================
    if verbose:
        _print_header(f"Compiling CUDA Kernel: {name}", use_colors)

        # Source code statistics
        cuda_lines = cuda_source.count('\n') + 1 if cuda_source else 0
        cpp_lines = cpp_source.count('\n') + 1 if cpp_source else 0
        total_lines = cuda_lines + cpp_lines

        _print_step("Source code analysis", use_colors=use_colors)
        if use_colors:
            print(f"    CUDA source:  {Colors.bold(f'{cuda_lines} lines')}")
            print(f"    C++ source:   {Colors.bold(f'{cpp_lines} lines')}")
            print(f"    Total:        {Colors.bold(f'{total_lines} lines')}")
        else:
            print(f"    CUDA source:  {cuda_lines} lines")
            print(f"    C++ source:   {cpp_lines} lines")
            print(f"    Total:        {total_lines} lines")

        # Detect functions
        detected_functions = _count_functions(cuda_source + cpp_source)
        if functions:
            func_str = ", ".join(functions[:5])
            if len(functions) > 5:
                func_str += f" ... ({len(functions)} total)"
            if use_colors:
                print(f"    Functions to expose: {Colors.green(func_str)}")
            else:
                print(f"    Functions to expose: {func_str}")
        elif detected_functions:
            func_str = ", ".join(detected_functions[:5])
            if len(detected_functions) > 5:
                func_str += f" ... ({len(detected_functions)} detected)"
            if use_colors:
                print(f"    Detected functions: {Colors.cyan(func_str)}")
            else:
                print(f"    Detected functions: {func_str}")

    # ========================================================================
    # STEP 2: ENVIRONMENT SETUP
    # ========================================================================
    # Get PyTorch's build directories (always needed for error handling)
    pytorch_cache_root = torch._appdirs.user_cache_dir(appname='torch_extensions')
    pytorch_build_root = _get_pytorch_build_root()

    if verbose:
        _print_step("Environment setup", use_colors=use_colors)

        if use_colors:
            print(f"    PyTorch cache: {Colors.cyan(pytorch_cache_root)}")
            print(f"    Build root:    {Colors.cyan(pytorch_build_root)}")
            print(f"    PyTorch:       {Colors.cyan(torch.__version__)}")
            print(f"    CUDA:          {Colors.cyan(torch.version.cuda or 'N/A')}")
            print(f"    Environment:   {Colors.green('Colab' if is_colab else 'Local')}")
        else:
            print(f"    PyTorch cache: {pytorch_cache_root}")
            print(f"    Build root:    {pytorch_build_root}")
            print(f"    PyTorch:       {torch.__version__}")
            print(f"    CUDA:          {torch.version.cuda or 'N/A'}")
            print(f"    Environment:   {'Colab' if is_colab else 'Local'}")

    # Determine symlink directory
    symlink_dir = None
    if build_directory is None and is_colab:
        symlink_dir = Path('build')
        symlink_dir.mkdir(parents=True, exist_ok=True)
    elif build_directory is not None:
        symlink_dir = Path(build_directory)
        symlink_dir.mkdir(parents=True, exist_ok=True)

    if symlink_dir and verbose:
        if use_colors:
            print(f"    Symlink dir:   {Colors.cyan(str(symlink_dir))}")
        else:
            print(f"    Symlink dir:   {symlink_dir}")

    # ========================================================================
    # STEP 3: PREPARE COMPILATION FLAGS
    # ========================================================================
    if verbose:
        _print_step("Preparing compilation flags", use_colors=use_colors)

    cuda_info = get_cuda_info()
    base_flags = ['-O3']
    if not is_colab:
        extra_cuda_cflags = cuda_info.get('extra_cuda_cflags', ['-O3'])
    else:
        extra_cuda_cflags = base_flags

    if verbose and extra_cuda_cflags:
        flag_str = " ".join(extra_cuda_cflags[:4])
        if len(extra_cuda_cflags) > 4:
            flag_str += " ..."
        if use_colors:
            print(f"    CUDA flags:    {Colors.cyan(flag_str)}")
        else:
            print(f"    CUDA flags:    {flag_str}")

    # Prepare kwargs
    load_inline_kwargs = {
        'name': name,
        'cpp_sources': [cpp_source] if cpp_source else [],
        'cuda_sources': [cuda_source] if cuda_source else [],
        'extra_cuda_cflags': extra_cuda_cflags,
        'verbose': verbose,
    }

    # ========================================================================
    # STEP 4: COMPILATION
    # ========================================================================
    if verbose:
        _print_step("Starting compilation", "running ninja build...", use_colors=use_colors)

    start_time = time.time()
    module = None
    compilation_error = None
    actual_so_path = None  # Initialize for use in later steps

    try:
        # Try with with_pybind11 (newer PyTorch)
        try:
            module = load_inline(**load_inline_kwargs, with_pybind11=True)
        except TypeError:
            # Fall back to older PyTorch API (Colab uses older PyTorch)
            if verbose:
                if use_colors:
                    print(f"    {Colors.warning('Falling back to older PyTorch API...')}")
                else:
                    print(f"    Falling back to older PyTorch API...")
            module = load_inline(**{k: v for k, v in load_inline_kwargs.items()
                                   if k != 'with_pybind11'})

        elapsed = time.time() - start_time

    except (ImportError, OSError, RuntimeError) as e:
        elapsed = time.time() - start_time
        compilation_error = e

    # ========================================================================
    # STEP 5: POST-COMPILATION ANALYSIS
    # ========================================================================
    if verbose:
        _print_step("Compilation results", use_colors=use_colors)

        if use_colors:
            print(f"    Time elapsed:  {Colors.bold(f'{elapsed:.2f}s')}")
        else:
            print(f"    Time elapsed:  {elapsed:.2f}s")

    # Handle compilation errors
    if compilation_error is not None:
        import traceback
        error_details = traceback.format_exc()

        # Try to find the .so file even if import failed
        actual_so_path = find_so_file(name)

        if verbose:
            if actual_so_path:
                if use_colors:
                    print(f"    .so file:      {Colors.green('FOUND (but import failed)')}")
                    print(f"    Location:     {Colors.cyan(str(actual_so_path))}")
                    file_size = actual_so_path.stat().st_size if actual_so_path.exists() else 0
                    print(f"    Size:         {Colors.bold(_format_size(file_size))}")
                else:
                    print(f"    .so file:      FOUND (but import failed)")
                    print(f"    Location:     {actual_so_path}")
                    file_size = actual_so_path.stat().st_size if actual_so_path.exists() else 0
                    print(f"    Size:         {_format_size(file_size)}")
            else:
                if use_colors:
                    print(f"    .so file:      {Colors.fail('NOT FOUND')}")
                else:
                    print(f"    .so file:      NOT FOUND")

        # Build detailed error message
        if "cannot open shared object file" in str(compilation_error) or "No such file or directory" in str(compilation_error):
            error_msg = Colors.fail("CUDA JIT compilation failed (shared object loading error).") if use_colors else "CUDA JIT compilation failed (shared object loading error)."
            error_msg += "\n\n"

            if actual_so_path:
                error_msg += Colors.green("The .so file WAS compiled successfully!") if use_colors else "The .so file WAS compiled successfully!"
                error_msg += f"\nLocation: {actual_so_path}\n\n"
                error_msg += "This indicates a linking or import issue, not a compilation issue.\n"
            else:
                error_msg += Colors.fail("The .so file was NOT found. Compilation may have failed.") if use_colors else "The .so file was NOT found. Compilation may have failed."
                error_msg += "\n\n"

            error_msg += (
                f"PyTorch build root: {pytorch_build_root}\n"
                f"PyTorch cache: {pytorch_cache_root}\n\n"
                "Possible solutions:\n"
                "1. Use the PyTorch baseline model (nn.MultiheadAttention) instead\n"
                "2. Try restarting the runtime and running the notebook again\n"
                f"3. Clear the torch extensions cache: rm -rf {pytorch_cache_root}\n"
                "4. Install ninja: pip install ninja\n"
                "5. Check that CUDA toolkit is installed: nvcc --version\n\n"
                f"Original error: {compilation_error}"
            )
            raise RuntimeError(error_msg)
        else:
            error_msg = Colors.fail("CUDA JIT compilation encountered an error.") if use_colors else "CUDA JIT compilation encountered an error."
            error_msg += f"\n\nPyTorch build root: {pytorch_build_root}\n"
            error_msg += f"PyTorch cache: {pytorch_cache_root}\n"
            error_msg += f"Actual .so location: {actual_so_path if actual_so_path else 'Not found'}\n\n"
            error_msg += "This is common in Colab due to PyTorch JIT limitations. "
            error_msg += "Use the baseline PyTorch model instead.\n\n"
            error_msg += f"Error: {compilation_error}\n\n{error_details}"
            raise RuntimeError(error_msg)

    # ========================================================================
    # STEP 6: VERIFY OUTPUT FILES
    # ========================================================================
    # Find the actual .so file location (needed for symlink step)
    actual_so_path = find_so_file(name)

    if verbose:
        if actual_so_path and actual_so_path.exists():
            file_size = actual_so_path.stat().st_size
            if use_colors:
                print(f"    Output:        {Colors.green('✓ Compilation successful!')}")
                print(f"    Location:      {Colors.cyan(str(actual_so_path))}")
                print(f"    Size:          {Colors.bold(_format_size(file_size))}")
            else:
                print(f"    Output:        ✓ Compilation successful!")
                print(f"    Location:      {actual_so_path}")
                print(f"    Size:          {_format_size(file_size)}")
        else:
            if use_colors:
                print(f"    Output:        {Colors.warning('Warning: .so file not found in expected location')}")
            else:
                print(f"    Output:        Warning: .so file not found in expected location")

    # ========================================================================
    # STEP 7: CREATE SYMLINK
    # ========================================================================
    if create_symlink and symlink_dir and actual_so_path:
        if verbose:
            _print_step("Creating convenience symlink", use_colors=use_colors)

        symlink_path = symlink_dir / f"{name}.so"
        try:
            # Remove existing symlink if it exists
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            # Create symlink
            os.symlink(actual_so_path, symlink_path)
            if verbose:
                if use_colors:
                    print(f"    Symlink:       {Colors.green('✓ Created')}")
                    print(f"    From:          {Colors.cyan(str(symlink_path))}")
                    print(f"    To:            {Colors.cyan(str(actual_so_path))}")
                else:
                    print(f"    Symlink:       ✓ Created")
                    print(f"    From:          {symlink_path}")
                    print(f"    To:            {actual_so_path}")
        except OSError as e:
            if verbose:
                if use_colors:
                    print(f"    Symlink:       {Colors.warning(f'Failed: {e}')}")
                else:
                    print(f"    Symlink:       Failed: {e}")

    # ========================================================================
    # STEP 8: MODULE IMPORT INFO
    # ========================================================================
    if verbose:
        _print_step("Module import information", use_colors=use_colors)

        if actual_so_path:
            if use_colors:
                print(f"    Import path:   {Colors.cyan(str(actual_so_path))}")
            else:
                print(f"    Import path:   {actual_so_path}")

            # Try to get available functions from the module
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

    # ========================================================================
    # FINAL SUCCESS MESSAGE
    # ========================================================================
    if verbose:
        if use_colors:
            print(f"\n{Colors.bold(Colors.green('✅ Module ready!'))}")
        else:
            print(f"\n✅ Module ready!")

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


def compile_with_fallback(
    name: str,
    cuda_source: str,
    cpp_source: str = '',
    functions: Optional[List[str]] = None,
    build_directory: Optional[Path] = None,
    verbose: bool = True,
    prefer_setuptools: bool = False,
) -> Any:
    """
    Compile CUDA kernel with automatic fallback between JIT and setuptools.

    This is the RECOMMENDED way to compile CUDA kernels in Google Colab.

    Attempts:
    1. PyTorch's load_inline() (JIT) - faster but can fail in Colab
    2. Setuptools compilation - slower but more reliable

    Args:
        name: Name for the compiled module
        cuda_source: CUDA source code (.cu file contents)
        cpp_source: Optional C++ source code
        functions: List of function names to expose (for JIT only)
        build_directory: Optional directory for build artifacts
        verbose: Whether to print compilation output
        prefer_setuptools: If True, skip JIT and use setuptools directly

    Returns:
        Compiled Python module

    Raises:
        RuntimeError: If both compilation methods fail

    Example:
        >>> from utils.cuda_build import compile_with_fallback
        >>> module = compile_with_fallback(
        ...     name="fused_attention",
        ...     cuda_source=cuda_code,
        ...     cpp_source=cpp_code
        ... )
    """
    # If setuptools is preferred, skip JIT
    if prefer_setuptools:
        if verbose:
            _print_step("Using setuptools compilation (preferred)", use_colors=_get_color_support())
        try:
            from utils.compile_setuptools import compile_with_setuptools
            return compile_with_setuptools(
                name=name,
                cuda_source=cuda_source,
                cpp_source=cpp_source,
                output_dir=build_directory,
                verbose=verbose,
            )
        except ImportError:
            if verbose:
                print("Warning: compile_setuptools not available, falling back to JIT")
            # Fall through to JIT below

    # Try JIT first (default)
    if not prefer_setuptools:
        if verbose:
            _print_step("Attempting JIT compilation (load_inline)", use_colors=_get_color_support())

        jit_success = False
        module = None
        jit_error = None

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
                use_colors = _get_color_support()
                if use_colors:
                    print(f"    {Colors.green('✓ JIT compilation succeeded!')}")
                else:
                    print(f"    ✓ JIT compilation succeeded!")
        except Exception as e:
            jit_error = e
            if verbose:
                use_colors = _get_color_support()
                if use_colors:
                    print(f"    {Colors.warning('⚠ JIT compilation failed')}")
                    print(f"    Error: {str(e)[:150] if str(e) else 'Unknown error'}...")
                else:
                    print(f"    ⚠ JIT compilation failed")
                    print(f"    Error: {str(e)[:150] if str(e) else 'Unknown error'}...")

        # If JIT failed, try setuptools as fallback
        if not jit_success:
            if verbose:
                _print_step("Falling back to setuptools compilation", use_colors=_get_color_support())

            try:
                from utils.compile_setuptools import compile_with_setuptools
                module = compile_with_setuptools(
                    name=name,
                    cuda_source=cuda_source,
                    cpp_source=cpp_source,
                    output_dir=build_directory,
                    verbose=verbose,
                )
            except ImportError:
                # Setuptools not available, re-raise the original JIT error
                if verbose:
                    use_colors = _get_color_support()
                    if use_colors:
                        print(f"    {Colors.fail('✗ Setuptools not available')}")
                    else:
                        print(f"    ✗ Setuptools not available")
                if jit_error:
                    raise RuntimeError(
                        f"Both JIT and setuptools compilation failed. "
                        f"JIT error: {jit_error}"
                    )
                else:
                    raise RuntimeError("Setuptools fallback failed (module not available)")

        return module

    return module  # Should not reach here, but for type checkers
