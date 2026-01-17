#!/usr/bin/env python3
"""
StyleForge Verification Test Runner

Runs all verification tests and generates a summary report.

Usage:
    # Run all tests
    python run_verification_tests.py

    # Run specific test module
    python run_verification_tests.py --test model_loading

    # Run with specific style
    python run_verification_tests.py --style candy

    # Skip visual tests (no image outputs)
    python run_verification_tests.py --skip-visual

    # Generate report only
    python run_verification_tests.py --report-only
"""

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.config import DEVICE, IS_CUDA_AVAILABLE, AVAILABLE_STYLES


# Test modules
TEST_MODULES = {
    "model_loading": "tests.test_model_loading",
    "forward_pass": "tests.test_forward_pass",
    "visual_quality": "tests.test_visual_quality",
    "cuda_kernel": "tests.test_cuda_kernel_usage",
    "numerical_accuracy": "tests.test_numerical_accuracy",
    "memory_leaks": "tests.test_memory_leaks",
}


class TestResult:
    """Store result of a single test."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.failed = False
        self.skipped = False
        self.duration = 0.0
        self.error = None
        self.output = []


class TestRunner:
    """Run and track test results."""

    def __init__(self, style: str = "candy", skip_visual: bool = False):
        self.style = style
        self.skip_visual = skip_visual
        self.results = []
        self.start_time = None

    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test function."""
        result = TestResult(test_name)

        # Skip visual tests if requested
        if self.skip_visual and "visual" in test_name.lower():
            result.skipped = True
            return result

        # Skip CUDA tests if CUDA not available
        if "cuda" in test_name.lower() and not IS_CUDA_AVAILABLE:
            result.skipped = True
            return result

        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)

        start = time.time()

        try:
            test_func()
            result.passed = True
            result.duration = time.time() - start

        except Exception as e:
            result.failed = True
            result.duration = time.time() - start
            result.error = str(e)

        return result

    def run_module(self, module_name: str) -> list[TestResult]:
        """Run all tests from a module."""
        module_results = []

        try:
            module = importlib.import_module(module_name)

            # Find all test functions
            test_funcs = []
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if attr_name.startswith("test_") and callable(attr):
                    test_funcs.append((attr_name, attr))

            # Run each test
            for test_name, test_func in test_funcs:
                result = self.run_test(test_name, test_func)
                module_results.append(result)
                self.results.append(result)

        except Exception as e:
            # Module import or execution failed
            result = TestResult(module_name)
            result.failed = True
            result.error = f"Module error: {e}"
            module_results.append(result)
            self.results.append(result)

        return module_results

    def run_all(self) -> dict:
        """Run all test modules."""
        self.start_time = time.time()

        print("=" * 60)
        print("STYLE FORGE VERIFICATION TEST SUITE")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {DEVICE}")
        print(f"CUDA available: {IS_CUDA_AVAILABLE}")
        print(f"Test style: {self.style}")
        print("=" * 60)

        summary = {}

        for test_key, module_name in TEST_MODULES.items():
            # Skip visual tests if requested
            if self.skip_visual and test_key == "visual_quality":
                print(f"\n⏭️  Skipping {test_key} (--skip-visual)")
                summary[test_key] = {"passed": 0, "failed": 0, "skipped": 1}
                continue

            module_results = self.run_module(module_name)

            passed = sum(1 for r in module_results if r.passed)
            failed = sum(1 for r in module_results if r.failed)
            skipped = sum(1 for r in module_results if r.skipped)

            summary[test_key] = {
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "results": module_results,
            }

        total_duration = time.time() - self.start_time

        return {
            "summary": summary,
            "total_duration": total_duration,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
        }

    def print_summary(self, results: dict):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        summary = results["summary"]

        # Module-level summary
        print(f"\n{'Module':<25} {'Passed':<8} {'Failed':<8} {'Skipped':<8}")
        print("-" * 60)

        total_passed = 0
        total_failed = 0
        total_skipped = 0

        for module_name, stats in summary.items():
            passed = stats["passed"]
            failed = stats["failed"]
            skipped = stats["skipped"]

            total_passed += passed
            total_failed += failed
            total_skipped += skipped

            status = "✅" if failed == 0 else "❌"
            print(f"{status} {module_name:<25} {passed:<8} {failed:<8} {skipped:<8}")

        print("-" * 60)
        print(f"{'TOTAL':<25} {total_passed:<8} {total_failed:<8} {total_skipped:<8}")
        print()

        # Duration
        duration = results["total_duration"]
        print(f"Duration: {duration:.2f} seconds")

        # Overall result
        if total_failed == 0:
            print("\n✅ ALL TESTS PASSED")
        else:
            print(f"\n❌ {total_failed} TEST(S) FAILED")

        print("=" * 60)

        # Print failed test details
        if total_failed > 0:
            print("\nFailed Tests:")
            print("-" * 60)

            for module_name, stats in summary.items():
                for result in stats.get("results", []):
                    if result.failed:
                        print(f"\n{module_name}.{result.name}")
                        print(f"  Error: {result.error}")

        # Print warnings
        if total_skipped > 0:
            print(f"\n⚠️  {total_skipped} test(s) skipped")


def generate_report(results: dict, output_path: Path):
    """Generate a detailed verification report."""

    # First, convert results to JSON-serializable format
    summary = results["summary"]
    serializable_summary = {}

    for module_name, stats in summary.items():
        serializable_summary[module_name] = {
            "passed": stats["passed"],
            "failed": stats["failed"],
            "skipped": stats["skipped"],
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "failed": r.failed,
                    "skipped": r.skipped,
                    "duration": r.duration,
                    "error": r.error,
                }
                for r in stats.get("results", [])
            ]
        }

    serializable_results = {
        "summary": serializable_summary,
        "total_duration": results["total_duration"],
        "start_time": results["start_time"],
        "end_time": results["end_time"],
    }

    report = {
        "styleforge_verification_report": {
            "version": "0.1.0",
            "timestamp": datetime.now().isoformat(),
            "device": str(DEVICE),
            "cuda_available": IS_CUDA_AVAILABLE,
            "results": serializable_results,
        },
    }

    # Write JSON
    json_path = output_path / "verification_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Write Markdown
    md_path = output_path / "verification_report.md"
    with open(md_path, "w") as f:
        f.write("# StyleForge Verification Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Device:** {DEVICE}\n\n")
        f.write(f"**CUDA Available:** {IS_CUDA_AVAILABLE}\n\n")

        summary = results["summary"]

        f.write("## Test Summary\n\n")
        f.write(f"| Module | Passed | Failed | Skipped |\n")
        f.write(f"|--------|--------|--------|---------|\n")

        total_passed = 0
        total_failed = 0
        total_skipped = 0

        for module_name, stats in summary.items():
            passed = stats["passed"]
            failed = stats["failed"]
            skipped = stats["skipped"]

            total_passed += passed
            total_failed += failed
            total_skipped += skipped

            status = "✅" if failed == 0 else "❌"
            f.write(f"| {status} {module_name} | {passed} | {failed} | {skipped} |\n")

        f.write(f"| **TOTAL** | **{total_passed}** | **{total_failed}** | **{total_skipped}** |\n\n")

        f.write(f"**Duration:** {results['total_duration']:.2f} seconds\n\n")

        if total_failed > 0:
            f.write("## Failed Tests\n\n")
            for module_name, stats in summary.items():
                for result in stats.get("results", []):
                    if result.failed:
                        f.write(f"### {module_name}.{result.name}\n")
                        f.write(f"```\n{result.error}\n```\n\n")

        f.write("## Success Criteria\n\n")
        all_passed = total_failed == 0
        f.write(f"- [x] All tests pass: {'✅ Yes' if all_passed else '❌ No'}\n")
        f.write(f"- [ ] Visual outputs verified (manual inspection required)\n")
        f.write(f"- [ ] Custom kernels confirmed running (if CUDA available)\n")
        f.write(f"- [ ] No numerical issues\n")
        f.write(f"- [ ] No memory leaks\n")

    print(f"\n📄 Reports saved to {output_path}")
    print(f"  - {json_path}")
    print(f"  - {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run StyleForge verification tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_verification_tests.py

  # Run specific test module
  python run_verification_tests.py --test model_loading

  # Run with specific style
  python run_verification_tests.py --style candy

  # Skip visual tests (faster, no image outputs)
  python run_verification_tests.py --skip-visual

  # Generate report only
  python run_verification_tests.py --report-only
        """
    )

    parser.add_argument(
        "--test", "-t",
        choices=list(TEST_MODULES.keys()),
        help="Run specific test module"
    )
    parser.add_argument(
        "--style", "-s",
        default="candy",
        choices=AVAILABLE_STYLES + ["all"],
        help="Style model to test (default: candy)"
    )
    parser.add_argument(
        "--skip-visual",
        action="store_true",
        help="Skip visual quality tests (no image outputs)"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report from existing results"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for reports (default: test_outputs)"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "test_outputs"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    runner = TestRunner(style=args.style, skip_visual=args.skip_visual)

    if args.test:
        # Run single test module
        module_name = TEST_MODULES[args.test]
        module_results = runner.run_module(module_name)
        passed = sum(1 for r in module_results if r.passed)
        failed = sum(1 for r in module_results if r.failed)
        skipped = sum(1 for r in module_results if r.skipped)
        results = {
            "summary": {
                args.test: {
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "results": module_results,
                }
            },
            "total_duration": sum(r.duration for r in runner.results),
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
        }
    else:
        # Run all tests
        results = runner.run_all()

    # Print summary
    runner.print_summary(results)

    # Generate report
    generate_report(results, output_dir)

    # Print visual inspection reminder
    if not args.skip_visual:
        print("\n📁 Remember to visually inspect generated images:")
        print(f"   {output_dir / '*.jpg'}")

    # Return exit code
    total_failed = sum(s["failed"] for s in results["summary"].values())
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
