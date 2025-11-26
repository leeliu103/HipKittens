# RDNA4 Tests for HipKittens

This directory contains RDNA4-specific tests for the HipKittens library, targeting AMD RDNA4 GPUs (gfx1201) with wave32 support.

## Quick Start

```bash
# Run all tests that compile on RDNA4
./build_individual_tests.sh

# Run a specific test
./build_individual_tests.sh warp_memory_tile_global_to_register
```

## Environment Requirements

### Hardware
- AMD RDNA4 GPU (gfx1201)
- ROCm-capable system

### Software
- ROCm 7.0 or later
- hipCC compiler

### Environment Variables
```bash
export ROCM_PATH=/opt/rocm            # ROCm installation path
export HIP_VISIBLE_DEVICES=0          # Use RDNA4 GPU (important if multiple GPUs)
```

**Important:** This system has 2 GPUs. Always use `HIP_VISIBLE_DEVICES=0` to target the RDNA4 (gfx1201) GPU.

---

## Test Organization

### Total: 13 Real Tests

**Status Breakdown:**
- ✅ **6 tests** compile successfully on RDNA4
- ❌ **7 tests** blocked by RDNA4 incompatibilities

---

## ✅ Tests That Compile (6 total)

### Warp Memory Tests (2/4)
1. `warp/memory/tile/global_to_register.cu` - Global→register tile transfers
2. `warp/memory/vec/global_to_register.cu` - Global→register vector transfers

### Warp Register Tile Tests (2/4)
3. `warp/register/tile/conversions.cu` - Layout conversions, swap tests
4. `warp/register/tile/maps.cu` - Element-wise operations

### Warp Register Vec Tests (2/3)
5. `warp/register/vec/maps.cu` - Vector element-wise ops
6. `warp/register/vec/reductions.cu` - Vector reductions

**Current Status:** These tests compile but **fail at runtime** due to shared memory limit issues (requesting 80KB, likely exceeding RDNA4's 64KB limit).

---

## ❌ Tests That Don't Compile (7 total)

### Issue 1: Missing `llvm.amdgcn.raw.buffer.load.lds` (4 tests)
**CDNA-only direct global→LDS DMA not available on RDNA4**

- `warp/memory/tile/shared_to_register.cu`
- `warp/memory/vec/shared_to_register.cu`
- `warp/shared/tile/conversions.cu`
- `warp/shared/vec/conversions.cu`

### Issue 2: Missing `__builtin_amdgcn_permlane32_swap` (1 test)
**permlane32-swap feature not exposed on gfx1201**

- `warp/register/tile/reductions.cu`

### Issue 3: Wave32 MFMA Incompatibility (1 test)
**MFMA wrappers hard-coded for wave64 layouts**

- `warp/register/tile/mma.cu`

### Issue 4: Missing `v_cvt_pk_bf16_f32` (1 test)
**BF16 pack instruction not supported on RDNA4**

- `warp/register/vec/conversions.cu`

---

## Usage Examples

### Run All Compilable Tests
```bash
./build_individual_tests.sh
```

### Run Individual Test
```bash
./build_individual_tests.sh warp_register_vec_maps
```

### Manual Build
```bash
# Example: Build global_to_register test
make clean
ROCM_PATH=/opt/rocm \
SKIP_TESTS="warp/shared/tile/conversions.cu warp/shared/vec/conversions.cu warp/memory/tile/shared_to_register.cu warp/memory/vec/shared_to_register.cu warp/register/tile/reductions.cu warp/register/tile/mma.cu warp/register/vec/conversions.cu" \
TEST_DEFINES="-DTEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER" \
make -j8

# Run with correct GPU
HIP_VISIBLE_DEVICES=0 ./unit_tests
```

---

## Known Issues

### 1. Shared Memory Limit Exceeded (Affects ALL runtime tests)
**Error:** `hipFuncSetAttribute: Returned hipErrorInvalidValue`

**Cause:** Tests request 80KB dynamic shared memory, but RDNA4 likely has 64KB limit

**Temporary Workaround:** Lower TEST_INTENSITY or modify test parameters

**Logs:**
```
hipFuncSetAttribute ( ..., 8, 80000 )  # Requesting 80KB
hipFuncSetAttribute: Returned hipErrorInvalidValue
hipLaunchKernel: Returned hipErrorInvalidValue
```

### 2. RDNA4-Specific Compilation Failures
See detailed fixes required in `todo.md`

---

## Directory Structure

```
rdna4_tests/
├── README.md                      # This file
├── todo.md                        # Detailed RDNA4 issues and fixes
├── RDNA4_TEST_RESULTS.md         # Full test execution report
├── build_individual_tests.sh     # Helper script to build/run tests
├── Makefile                       # Build system
├── unit_tests.cu                  # Main test entry point
├── warp/                          # Warp-level tests
│   ├── memory/                    # Memory operations
│   │   ├── tile/
│   │   │   ├── global_to_register.cu ✅
│   │   │   └── shared_to_register.cu ❌
│   │   └── vec/
│   │       ├── global_to_register.cu ✅
│   │       └── shared_to_register.cu ❌
│   ├── register/                  # Register operations
│   │   ├── tile/
│   │   │   ├── conversions.cu ✅
│   │   │   ├── maps.cu ✅
│   │   │   ├── mma.cu ❌
│   │   │   └── reductions.cu ❌
│   │   └── vec/
│   │       ├── conversions.cu ❌
│   │       ├── maps.cu ✅
│   │       └── reductions.cu ✅
│   └── shared/                    # Shared memory operations
│       ├── tile/
│       │   └── conversions.cu ❌
│       └── vec/
│           └── conversions.cu ❌
└── unit/                          # Shared test utilities (symlinked from tests/unit/)
```

---

## Available Test Names

For use with `./build_individual_tests.sh [test_name]`:

```
warp_memory_tile_global_to_register
warp_memory_vec_global_to_register
warp_register_tile_conversions
warp_register_tile_maps
warp_register_vec_maps
warp_register_vec_reductions
```

---

## Test Flags Reference

| Test Name | Flag | Status |
|-----------|------|--------|
| `warp/memory/tile/global_to_register.cu` | `TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER` | ✅ Compiles |
| `warp/memory/vec/global_to_register.cu` | `TEST_WARP_MEMORY_VEC_GLOBAL_TO_REGISTER` | ✅ Compiles |
| `warp/register/tile/conversions.cu` | `TEST_WARP_REGISTER_TILE_CONVERSIONS` | ✅ Compiles |
| `warp/register/tile/maps.cu` | `TEST_WARP_REGISTER_TILE_MAPS` | ✅ Compiles |
| `warp/register/vec/maps.cu` | `TEST_WARP_REGISTER_VEC_MAPS` | ✅ Compiles |
| `warp/register/vec/reductions.cu` | `TEST_WARP_REGISTER_VEC_REDUCTIONS` | ✅ Compiles |
| `warp/memory/tile/shared_to_register.cu` | `TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER` | ❌ raw.buffer.load.lds |
| `warp/memory/vec/shared_to_register.cu` | `TEST_WARP_MEMORY_VEC_SHARED_TO_REGISTER` | ❌ raw.buffer.load.lds |
| `warp/register/tile/reductions.cu` | `TEST_WARP_REGISTER_TILE_REDUCTIONS` | ❌ permlane32_swap |
| `warp/register/tile/mma.cu` | `TEST_WARP_REGISTER_TILE_MMA` | ❌ wave32 MFMA |
| `warp/register/vec/conversions.cu` | `TEST_WARP_REGISTER_VEC_CONVERSIONS` | ❌ BF16 pack |
| `warp/shared/tile/conversions.cu` | `TEST_WARP_SHARED_TILE_CONVERSIONS` | ❌ raw.buffer.load.lds |
| `warp/shared/vec/conversions.cu` | `TEST_WARP_SHARED_VEC_CONVERSIONS` | ❌ raw.buffer.load.lds |

---

## Troubleshooting

### "no ROCm-capable device is detected"
```bash
# Check GPU visibility
rocminfo | grep gfx

# Ensure using RDNA4 GPU
export HIP_VISIBLE_DEVICES=0
```

### "/bin/hipcc: not found"
```bash
export ROCM_PATH=/opt/rocm
```

### "hipCheckError() failed: invalid argument"
This is the shared memory limit issue. The tests currently request 80KB but RDNA4 likely has a 64KB limit.

### Build Failures
For tests that don't compile, see `todo.md` for detailed explanations and required code fixes.

---

## Next Steps

See `todo.md` for:
- Detailed analysis of each RDNA4 incompatibility
- Required code fixes with examples
- Priority action items
- Long-term roadmap

---

## Working Example

The HipKittens add example works successfully on RDNA4:
```bash
cd /workspace/HipKittens/examples/add
make
./hipkittens_add
# Output: "HipKittens vector add completed successfully for 8192 elements"
```

This confirms basic HipKittens functionality works on RDNA4.
