"""
StyleForge - CUDA Environment Verification

Checks all prerequisites for CUDA JIT compilation in Google Colab and other environments.
Run this script to diagnose CUDA compilation issues.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple


def check_ninja() -> Tuple[bool, str, str]:
    """
    Check if ninja build system is installed (required for PyTorch JIT).

    Returns:
        Tuple of (passed, message, version)
    """
    try:
        result = subprocess.run(
            ['ninja', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, "Ninja is installed", version
        else:
            return False, "Ninja not found", ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "Ninja not found (install with: pip install ninja)", ""


def check_cuda_toolkit() -> Tuple[bool, str, str]:
    """
    Check if CUDA toolkit is available and get version.

    Returns:
        Tuple of (passed, message, version)
    """
    # Try nvcc from PATH
    nvcc_path = shutil.which('nvcc')

    # Check common CUDA locations
    cuda_locations = [
        Path('/usr/local/cuda'),
        Path('/opt/cuda'),
        Path('/usr/cuda'),
    ]

    actual_cuda_path = None
    if nvcc_path:
        actual_cuda_path = Path(nvcc_path).parent.parent
    else:
        for loc in cuda_locations:
            if (loc / 'bin' / 'nvcc').exists():
                nvcc_path = str(loc / 'bin' / 'nvcc')
                actual_cuda_path = loc
                break

    if nvcc_path:
        try:
            result = subprocess.run(
                [nvcc_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            output = result.stdout + result.stderr

            # Parse CUDA version from nvcc output
            import re
            match = re.search(r'release (\d+\.\d+)', output, re.IGNORECASE)
            if match:
                version = match.group(1)
                return True, f"CUDA toolkit found at {actual_cuda_path}", version
            else:
                return True, f"CUDA toolkit found (version unknown)", "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return False, "CUDA toolkit not found. Install CUDA or set CUDA_HOME.", ""


def check_pytorch_cuda() -> Tuple[bool, str, str]:
    """
    Check PyTorch CUDA availability and version.

    Returns:
        Tuple of (passed, message, version)
    """
    try:
        import torch
        if torch.cuda.is_available():
            version = torch.version.cuda or "unknown"
            return True, f"PyTorch built with CUDA {version}", version
        else:
            return False, "PyTorch CUDA not available (install with pip install torch)", ""
    except ImportError:
        return False, "PyTorch not installed (install with: pip install torch)", ""


def check_cuda_version_match(cuda_version: str, pytorch_cuda: str) -> Tuple[bool, str]:
    """
    Check if system CUDA version matches PyTorch's CUDA version.

    Returns:
        Tuple of (passed, message)
    """
    try:
        from torch.version import cuda as torch_cuda
    except ImportError:
        return False, "Cannot check: PyTorch not installed"

    if not cuda_version or not pytorch_cuda:
        return False, "Cannot check: missing version info"

    # Parse major.minor versions
    try:
        cuda_major, cuda_minor = map(int, cuda_version.split('.')[:2])
        pt_major, pt_minor = map(int, pytorch_cuda.split('.')[:2])
    except ValueError:
        return False, "Cannot parse CUDA versions"

    # Check for major version mismatch (critical)
    if cuda_major != pt_major:
        return False, f"CRITICAL: CUDA major version mismatch! System {cuda_version}.{cuda_minor} vs PyTorch {pytorch_cuda}"

    # Warn on minor version mismatch
    if cuda_minor != pt_minor:
        return True, f"WARNING: CUDA minor version mismatch. System {cuda_version} vs PyTorch {pytorch_cuda}"

    return True, f"CUDA versions match: {cuda_version}"


def check_gcc_version() -> Tuple[bool, str, str]:
    """
    Check GCC/G++ version for CUDA compatibility.

    Returns:
        Tuple of (passed, message, version)
    """
    # Check minimum GCC versions for different CUDA versions
    # From PyTorch cpp_extension.py source code
    MINIMUM_GCC_VERSION = (5, 0, 0)

    # CUDA-specific bounds (simplified)
    CUDA_GCC_BOUNDS = {
        '11.x': ((5, 0), (12, 0)),
        '12.x': ((6, 0), (14, 0)),
    }

    try:
        result = subprocess.run(
            ['gcc', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout

        # Parse GCC version
        import re
        match = re.search(r'gcc.*?(\d+\.\d+\.\d+)', output, re.IGNORECASE)
        if not match:
            match = re.search(r'\(GCC\) (\d+\.\d+\.\d+)', output)

        if match:
            version_str = match.group(1)
            version_tuple = tuple(map(int, version_str.split('.')))

            # Check against minimum
            if version_tuple >= MINIMUM_GCC_VERSION:
                return True, f"GCC version compatible", version_str
            else:
                return False, f"GCC version too old. Need >={'.'.join(map(str, MINIMUM_GCC_VERSION))}", version_str
        else:
            return False, "Could not parse GCC version", ""
    except FileNotFoundError:
        return False, "GCC not found", ""


def check_compiler_abi() -> Tuple[bool, str]:
    """
    Check if compiler is ABI-compatible with PyTorch.
    This is a simplified check; the actual PyTorch function does more.
    """
    try:
        import torch
        # Check if this is a binary build or source build
        version_pattern = r'\d+\.\d+\.\d+[a-z0-9.]*\+'
        import re
        if re.search(version_pattern, torch.__version__):
            return True, "PyTorch built from source (ABI checks may be relaxed)"
        else:
            return True, "PyTorch binary build (compiler ABI compatibility required)"
    except Exception:
        return False, "Cannot verify ABI compatibility"


def check_pytorch_version() -> Tuple[bool, str, str]:
    """
    Check PyTorch version and load_inline API support.
    """
    try:
        import torch
        version = torch.__version__

        # Parse version
        major, minor = map(int, version.split('.')[:2])

        # Check for load_inline support
        has_load_inline = hasattr(torch.utils.cpp_extension, 'load_inline')

        if not has_load_inline:
            return False, f"PyTorch {version} - load_inline not available", version

        # Check for is_standalone in load_inline signature
        import inspect
        sig = inspect.signature(torch.utils.cpp_extension.load_inline)
        has_is_standalone = 'is_standalone' in sig.parameters

        api_support = "load_inline supported"
        if has_is_standalone:
            api_support += " (has is_standalone param)"
        else:
            api_support += " (NO is_standalone param - this is correct!)"

        return True, api_support, version
    except ImportError:
        return False, "PyTorch not installed", ""


def check_build_directory_writable() -> Tuple[bool, str]:
    """
    Check if build directory is writable.
    """
    import torch

    default_build_dir = torch._appdirs.user_cache_dir(appname='torch_extensions')

    try:
        build_path = Path(default_build_dir)
        build_path.mkdir(parents=True, exist_ok=True)

        # Try to create a test file
        test_file = build_path / '.write_test'
        test_file.touch()
        test_file.unlink()

        return True, f"Build directory writable: {default_build_dir}"
    except (PermissionError, OSError) as e:
        return False, f"Build directory not writable: {default_build_dir} ({e})"


def check_load_inline_params() -> Dict[str, bool]:
    """
    Check which parameters are supported by load_inline.
    """
    try:
        import torch
        import inspect
        from torch.utils.cpp_extension import load_inline

        sig = inspect.signature(load_inline)
        params = sig.parameters

        return {
            'name': 'name' in params,
            'cpp_sources': 'cpp_sources' in params,
            'cuda_sources': 'cuda_sources' in params,
            'functions': 'functions' in params,
            'extra_cuda_cflags': 'extra_cuda_cflags' in params,
            'build_directory': 'build_directory' in params,
            'verbose': 'verbose' in params,
            'with_pybind11': 'with_pybind11' in params,
            'is_standalone': 'is_standalone' in params,  # Should be False!
            'is_python_module': 'is_python_module' in params,
        }
    except Exception:
        return {}


def run_all_checks() -> Dict[str, Any]:
    """
    Run all CUDA environment checks.

    Returns:
        Dictionary with check results
    """
    results = {
        'all_passed': True,
        'checks': {},
        'recommendations': [],
    }

    # 1. Check Ninja
    ninja_passed, ninja_msg, ninja_version = check_ninja()
    results['checks']['ninja'] = {
        'passed': ninja_passed,
        'message': ninja_msg,
        'version': ninja_version,
    }
    if not ninja_passed:
        results['all_passed'] = False
        results['recommendations'].append("Install ninja: pip install ninja")

    # 2. Check CUDA toolkit
    cuda_passed, cuda_msg, cuda_version = check_cuda_toolkit()
    results['checks']['cuda_toolkit'] = {
        'passed': cuda_passed,
        'message': cuda_msg,
        'version': cuda_version,
    }
    if not cuda_passed:
        results['all_passed'] = False
        results['recommendations'].append("Install CUDA toolkit or verify CUDA_HOME is set")

    # 3. Check PyTorch CUDA
    pt_passed, pt_msg, pt_cuda_version = check_pytorch_cuda()
    results['checks']['pytorch_cuda'] = {
        'passed': pt_passed,
        'message': pt_msg,
        'version': pt_cuda_version,
    }
    if not pt_passed:
        results['all_passed'] = False
        results['recommendations'].append("Install PyTorch with CUDA: pip install torch")

    # 4. Check CUDA version match
    if cuda_passed and pt_passed:
        match_passed, match_msg = check_cuda_version_match(cuda_version, pt_cuda_version)
        results['checks']['cuda_version_match'] = {
            'passed': match_passed,
            'message': match_msg,
        }
        if not match_passed:
            results['all_passed'] = False
            results['recommendations'].append(match_msg)

    # 5. Check GCC version
    gcc_passed, gcc_msg, gcc_version = check_gcc_version()
    results['checks']['gcc'] = {
        'passed': gcc_passed,
        'message': gcc_msg,
        'version': gcc_version,
    }
    if not gcc_passed:
        results['all_passed'] = False

    # 6. Check PyTorch version and API
    pt_ver_passed, pt_ver_msg, pt_version = check_pytorch_version()
    results['checks']['pytorch_version'] = {
        'passed': pt_ver_passed,
        'message': pt_ver_msg,
        'version': pt_version,
    }
    if not pt_ver_passed:
        results['all_passed'] = False

    # 7. Check build directory
    build_passed, build_msg = check_build_directory_writable()
    results['checks']['build_directory'] = {
        'passed': build_passed,
        'message': build_msg,
    }
    if not build_passed:
        results['all_passed'] = False
        results['recommendations'].append("Fix permissions for build directory or use sudo")

    # 8. Check load_inline parameters
    params = check_load_inline_params()
    results['checks']['load_inline_params'] = params
    if params.get('is_standalone', False):
        results['recommendations'].append(
            "WARNING: is_standalone parameter found in load_inline. "
            "This may cause issues - is_standalone is only for load(), not load_inline()"
        )

    return results


def print_report(results: Dict[str, Any]):
    """Print a formatted report of all checks."""
    print("\n" + "=" * 70)
    print("  CUDA Environment Verification Report")
    print("=" * 70)

    checks = results['checks']

    # Ninja
    ninja = checks['ninja']
    status = "✓" if ninja['passed'] else "✗"
    print(f"\n{status} Ninja:")
    print(f"    {ninja['message']}")
    if ninja['version']:
        print(f"    Version: {ninja['version']}")

    # CUDA Toolkit
    cuda = checks['cuda_toolkit']
    status = "✓" if cuda['passed'] else "✗"
    print(f"\n{status} CUDA Toolkit:")
    print(f"    {cuda['message']}")
    if cuda['version']:
        print(f"    Version: {cuda['version']}")

    # PyTorch CUDA
    pt_cuda = checks['pytorch_cuda']
    status = "✓" if pt_cuda['passed'] else "✗"
    print(f"\n{status} PyTorch CUDA:")
    print(f"    {pt_cuda['message']}")
    if pt_cuda['version']:
        print(f"    Built with CUDA: {pt_cuda['version']}")

    # CUDA Version Match
    if 'cuda_version_match' in checks:
        match = checks['cuda_version_match']
        status = "✓" if match['passed'] else "✗"
        print(f"\n{status} CUDA Version Match:")
        print(f"    {match['message']}")

    # GCC
    gcc = checks['gcc']
    status = "✓" if gcc['passed'] else "✗"
    print(f"\n{status} GCC/G++:")
    print(f"    {gcc['message']}")
    if gcc['version']:
        print(f"    Version: {gcc['version']}")

    # PyTorch Version
    pt_ver = checks['pytorch_version']
    status = "✓" if pt_ver['passed'] else "✗"
    print(f"\n{status} PyTorch Version:")
    print(f"    {pt_ver['message']}")
    if pt_ver['version']:
        print(f"    Version: {pt_ver['version']}")

    # Build Directory
    build = checks['build_directory']
    status = "✓" if build['passed'] else "✗"
    print(f"\n{status} Build Directory:")
    print(f"    {build['message']}")

    # load_inline Parameters
    params = checks['load_inline_params']
    print(f"\n  load_inline() Parameters:")
    for param, supported in sorted(params.items()):
        status = "✓" if supported else "✗"
        print(f"    {status} {param}")

    # Recommendations
    if results['recommendations']:
        print("\n" + "=" * 70)
        print("  Recommendations:")
        print("=" * 70)
        for rec in results['recommendations']:
            print(f"  • {rec}")

    # Summary
    print("\n" + "=" * 70)
    if results['all_passed']:
        print("  ✓ All critical checks passed! CUDA JIT should work.")
    else:
        print("  ✗ Some checks failed. CUDA JIT may not work.")
    print("=" * 70 + "\n")

    return results


def verify_and_fix(auto_fix: bool = False) -> Dict[str, Any]:
    """
    Run all checks and optionally auto-fix issues.

    Args:
        auto_fix: If True, attempt to install missing packages

    Returns:
        Dictionary with check results
    """
    results = run_all_checks()

    # Auto-fix: Install ninja if missing
    if auto_fix and not results['checks']['ninja']['passed']:
        print("Attempting to install ninja...")
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'ninja'],
                check=True,
                capture_output=True
            )
            print("✓ ninja installed successfully")
            # Re-check ninja
            ninja_passed, ninja_msg, ninja_version = check_ninja()
            results['checks']['ninja'] = {
                'passed': ninja_passed,
                'message': ninja_msg,
                'version': ninja_version,
            }
        except subprocess.CalledProcessError:
            print("✗ Failed to install ninja. Try: pip install ninja")

    return results


def main():
    """Main entry point for running verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Verify CUDA environment for PyTorch JIT compilation'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Automatically attempt to fix issues (e.g., install ninja)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    results = verify_and_fix(auto_fix=args.fix)

    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

    # Return exit code based on whether all checks passed
    sys.exit(0 if results['all_passed'] else 1)


if __name__ == '__main__':
    main()
