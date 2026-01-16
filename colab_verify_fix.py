# ============================================
# VERIFY THE FIX IS APPLIED
# ============================================
# Check if the kernel has the shared memory fix.
# ============================================

import sys
from pathlib import Path

print("=" * 70)
print("  VERIFYING KERNEL FIX")
print("=" * 70)

cuda_src_path = Path("kernels/attention.cu")
if not cuda_src_path.exists():
    print(f"\n❌ CUDA source not found at {cuda_src_path}")
    sys.exit(1)

with open(cuda_src_path, 'r') as f:
    cuda_source = f.read()

print("\nChecking for fix markers...")

# Check 1: Is the shared memory declared at function scope?
has_shared_mem_decl = "extern __shared__ float s_reduce[];" in cuda_source

# Check 2: Is the fix applied (shared memory reduction)?
has_shared_mem_fix = "s_reduce[head_idx] = partial_sum;" in cuda_source

# Check 3: Is the shared memory reduction the main path?
has_shared_mem_path = "General case: use shared memory for reduction" in cuda_source

print(f"\n✓ Shared memory declared at function scope: {has_shared_mem_decl}")
print(f"✓ Shared memory reduction fix present: {has_shared_mem_fix}")
print(f"✓ Shared memory is main reduction path: {has_shared_mem_path}")

if has_shared_mem_decl and has_shared_mem_fix and has_shared_mem_path:
    print("\n✅ FIX VERIFIED! The kernel has the shared memory reduction.")
    print("\nNext steps:")
    print("  1. Run: !git pull")
    print("  2. Run the cache clear script")
    print("  3. Restart the runtime")
    print("  4. Run the test again")
else:
    print("\n⚠️ Fix may not be fully applied. Please pull latest changes.")

print("\n" + "=" * 70)
