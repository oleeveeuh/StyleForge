"""
Script to clear ALL PyTorch caches and verify kernel recompilation.
Run this in Colab before the test.
"""
import os
import shutil
import sys

print("Clearing ALL PyTorch extension caches...")

# List of all possible cache locations
cache_locations = [
    '~/.cache/torch_extensions',
    '~/.local/share/torch_extensions',
    '~/torch_extensions',
    '/tmp/torch_extensions',
    '~/.cache/python',
]

cleared = []
for cache_dir in cache_locations:
    cache_dir = os.path.expanduser(cache_dir)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        cleared.append(cache_dir)
        print(f"  Cleared: {cache_dir}")

# Clear local build directories
local_dirs = ['build', 'kernels/__pycache__', '__pycache__']
for d in local_dirs:
    if os.path.exists(d):
        shutil.rmtree(d)
        cleared.append(d)
        print(f"  Cleared: {d}")

# Clear Python module cache
modules_to_clear = [k for k in sys.modules.keys() if k.startswith('kernels')]
for mod in modules_to_clear:
    del sys.modules[mod]
print(f"  Cleared {len(modules_to_clear)} Python modules")

print(f"\nTotal cleared: {len(cleared)} directories")
print("\nNow run: %run kernels/debug_cuda_kernel.py")
print("You should see:")
print("  *** ATTENTION_KERNEL_START: HEAD_DIM=16 ***")
print("  *** KERNEL VERSION: FIXED_QKV_PROJECTION ***")
print("  === KERNEL DEBUG: batch=0, head=0, q_pos=0, k_pos=0 ===")
