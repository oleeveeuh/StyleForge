# ============================================
# STYLEFORGE - FORCE CACHE CLEAR AND REBUILD
# ============================================
# Run this cell FIRST to clear all cached kernels
# and ensure fresh compilation with latest code.
# ============================================

import os
import sys
import shutil
from pathlib import Path

print("=" * 70)
print("  FORCE CLEARING ALL CUDA KERNEL CACHES")
print("=" * 70)

# Clear Python import cache
modules_to_clear = [k for k in sys.modules.keys() if 'kernels' in k.lower() or 'fused' in k.lower()]
if modules_to_clear:
    print(f"\nRemoving {len(modules_to_clear)} modules from Python cache:")
    for m in modules_to_clear:
        del sys.modules[m]
        print(f"  ✓ Removed: {m}")
else:
    print("\nNo kernel modules in Python cache")

# Clear PyTorch extension cache
torch_extensions_cache = Path.home() / ".cache" / "torch_extensions"
if torch_extensions_cache.exists():
    print(f"\nClearing PyTorch extension cache: {torch_extensions_cache}")

    # Count items before deletion
    items_before = list(torch_extensions_cache.iterdir())
    print(f"  Found {len(items_before)} cached extension(s)")

    # Remove ALL torch extensions (not just StyleForge ones)
    for item in torch_extensions_cache.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                print(f"  ✓ Removed: {item.name}")
            elif item.is_file():
                item.unlink()
                print(f"  ✓ Removed: {item.name}")
        except Exception as e:
            print(f"  ⚠ Could not remove {item.name}: {e}")

    print("\n✓ PyTorch extension cache cleared")
else:
    print(f"\n✓ No PyTorch extension cache found at {torch_extensions_cache}")

# Clear build directories
build_dirs = [
    Path("build"),
    Path("kernels/build"),
    Path("../build"),
]

for build_dir in build_dirs:
    if build_dir.exists():
        print(f"\nClearing build directory: {build_dir}")
        try:
            shutil.rmtree(build_dir, ignore_errors=True)
            print(f"  ✓ Removed: {build_dir}")
        except Exception as e:
            print(f"  ⚠ Could not remove {build_dir}: {e}")

# Clear any .pyc files in kernels directory
for pyc_file in Path(".").rglob("*.pyc"):
    try:
        pyc_file.unlink()
    except:
        pass

for __pycache__ in Path(".").rglob("__pycache__"):
    try:
        if __pycache__.is_dir():
            shutil.rmtree(__pycache__, ignore_errors=True)
    except:
        pass

print("\n" + "=" * 70)
print("  CACHE CLEARING COMPLETE")
print("=" * 70)
print("\n⚠️  IMPORTANT: You must now:")
print("   1. Run: Runtime → Restart session")
print("   2. Run: !git pull")
print("   3. Then run the notebook cells from cell-2")
print("=" * 70)
