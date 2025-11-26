#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cleanup_artifacts() {
    make -s clean >/dev/null 2>&1 || true
}
trap cleanup_artifacts EXIT

export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_VISIBLE_DEVICES=0

if command -v nproc >/dev/null 2>&1; then
    DEFAULT_JOBS=$(nproc)
else
    DEFAULT_JOBS=8
fi
JOBS=${JOBS:-$DEFAULT_JOBS}

echo "Using ROCM_PATH=$ROCM_PATH"
echo "Using HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "Using parallel jobs=$JOBS"
echo ""

TEST_ORDER=(
    warp_memory_tile_global_to_register
    warp_memory_tile_shared_to_register
    warp_memory_vec_global_to_register
    warp_memory_vec_shared_to_register
    warp_register_tile_conversions
    warp_register_tile_maps
    warp_register_tile_reductions
    warp_register_tile_mma
    warp_register_vec_conversions
    warp_register_vec_maps
    warp_register_vec_reductions
    warp_shared_tile_conversions
    warp_shared_vec_conversions
)

declare -A TEST_DEFINES=(
    [warp_memory_tile_global_to_register]=TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER
    [warp_memory_tile_shared_to_register]=TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER
    [warp_memory_vec_global_to_register]=TEST_WARP_MEMORY_VEC_GLOBAL_TO_REGISTER
    [warp_memory_vec_shared_to_register]=TEST_WARP_MEMORY_VEC_SHARED_TO_REGISTER
    [warp_register_tile_conversions]=TEST_WARP_REGISTER_TILE_CONVERSIONS
    [warp_register_tile_maps]=TEST_WARP_REGISTER_TILE_MAPS
    [warp_register_tile_reductions]=TEST_WARP_REGISTER_TILE_REDUCTIONS
    [warp_register_tile_mma]=TEST_WARP_REGISTER_TILE_MMA
    [warp_register_vec_conversions]=TEST_WARP_REGISTER_VEC_CONVERSIONS
    [warp_register_vec_maps]=TEST_WARP_REGISTER_VEC_MAPS
    [warp_register_vec_reductions]=TEST_WARP_REGISTER_VEC_REDUCTIONS
    [warp_shared_tile_conversions]=TEST_WARP_SHARED_TILE_CONVERSIONS
    [warp_shared_vec_conversions]=TEST_WARP_SHARED_VEC_CONVERSIONS
)

declare -A TEST_DESCRIPTIONS=(
    [warp_memory_tile_global_to_register]="Warp Memory Tile / Global to Register"
    [warp_memory_tile_shared_to_register]="Warp Memory Tile / Shared to Register"
    [warp_memory_vec_global_to_register]="Warp Memory Vec / Global to Register"
    [warp_memory_vec_shared_to_register]="Warp Memory Vec / Shared to Register"
    [warp_register_tile_conversions]="Warp Register Tile / Conversions"
    [warp_register_tile_maps]="Warp Register Tile / Maps"
    [warp_register_tile_reductions]="Warp Register Tile / Reductions"
    [warp_register_tile_mma]="Warp Register Tile / MMA"
    [warp_register_vec_conversions]="Warp Register Vec / Conversions"
    [warp_register_vec_maps]="Warp Register Vec / Maps"
    [warp_register_vec_reductions]="Warp Register Vec / Reductions"
    [warp_shared_tile_conversions]="Warp Shared Tile / Conversions"
    [warp_shared_vec_conversions]="Warp Shared Vec / Conversions"
)

declare -A TEST_SOURCES=(
    [warp_memory_tile_global_to_register]=warp/memory/tile/global_to_register.cu
    [warp_memory_tile_shared_to_register]=warp/memory/tile/shared_to_register.cu
    [warp_memory_vec_global_to_register]=warp/memory/vec/global_to_register.cu
    [warp_memory_vec_shared_to_register]=warp/memory/vec/shared_to_register.cu
    [warp_register_tile_conversions]=warp/register/tile/conversions.cu
    [warp_register_tile_maps]=warp/register/tile/maps.cu
    [warp_register_tile_reductions]=warp/register/tile/reductions.cu
    [warp_register_tile_mma]=warp/register/tile/mma.cu
    [warp_register_vec_conversions]=warp/register/vec/conversions.cu
    [warp_register_vec_maps]=warp/register/vec/maps.cu
    [warp_register_vec_reductions]=warp/register/vec/reductions.cu
    [warp_shared_tile_conversions]=warp/shared/tile/conversions.cu
    [warp_shared_vec_conversions]=warp/shared/vec/conversions.cu
)

declare -A TEST_KNOWN_ISSUES=(
    [warp_memory_tile_shared_to_register]="Missing llvm.amdgcn.raw.buffer.load.lds intrinsic on RDNA4"
    [warp_memory_vec_shared_to_register]="Missing llvm.amdgcn.raw.buffer.load.lds intrinsic on RDNA4"
    [warp_register_tile_reductions]="Requires __builtin_amdgcn_permlane32_swap"
    [warp_register_tile_mma]="Wave32 MFMA wrapper not implemented"
    [warp_register_vec_conversions]="Needs RDNA4 BF16 pack fallback"
    [warp_shared_tile_conversions]="Missing llvm.amdgcn.raw.buffer.load.lds intrinsic on RDNA4"
    [warp_shared_vec_conversions]="Missing llvm.amdgcn.raw.buffer.load.lds intrinsic on RDNA4"
)

usage() {
    cat >&2 <<'EOF'
Usage: ./build_individual_tests.sh [all|list|test_name ...]
  all             Build and run all 13 RDNA4 warp tests (default)
  list            Show available test identifiers
  test_name ...   Build and run the specified tests in order
EOF
}

list_tests() {
    echo "Available RDNA4 warp tests:" >&2
    for test_name in "${TEST_ORDER[@]}"; do
        printf "  %-35s %s (%s)\n" "$test_name" "${TEST_DESCRIPTIONS[$test_name]}" "${TEST_SOURCES[$test_name]}" >&2
    done
}

resolve_tests_to_run() {
    local args=("$@")
    local resolved=()
    if [ ${#args[@]} -eq 0 ]; then
        resolved=("${TEST_ORDER[@]}")
    else
        for arg in "${args[@]}"; do
            case "$arg" in
                -h|--help|help)
                    usage
                    exit 0
                    ;;
                -l|--list|list)
                    list_tests
                    exit 0
                    ;;
                -a|--all|all)
                    resolved=("${TEST_ORDER[@]}")
                    break
                    ;;
                *)
                    resolved+=("$arg")
                    ;;
            esac
        done
    fi

    if [ ${#resolved[@]} -eq 0 ]; then
        echo "No tests selected."
        usage
        exit 1
    fi

    echo "${resolved[@]}"
}

build_test() {
    local test_name=$1
    local test_define=$2

    printf "=========================================\n"
    printf "Building %s\n" "$test_name"
    printf "  Description : %s\n" "${TEST_DESCRIPTIONS[$test_name]}"
    printf "  Source      : %s\n" "${TEST_SOURCES[$test_name]}"
    printf "  Define      : -D%s\n" "$test_define"
    printf "=========================================\n"

    make -s clean >/dev/null 2>&1 || true

    if ROCM_PATH="$ROCM_PATH" TEST_DEFINES="-D$test_define" SKIP_TESTS="" make -j"$JOBS"; then
        echo "✅ Build SUCCESS: $test_name"
        return 0
    else
        echo "❌ Build FAILED: $test_name"
        return 1
    fi
}

run_test() {
    local test_name=$1
    printf "\nRunning %s (HIP_VISIBLE_DEVICES=%s)\n" "$test_name" "$HIP_VISIBLE_DEVICES"
    printf "-----------------------------------------\n"
    if HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" ./unit_tests; then
        echo "✅ Run SUCCESS: $test_name"
        return 0
    else
        local exit_code=$?
        echo "⚠️  Run FAILED: $test_name (exit code: $exit_code)"
        return $exit_code
    fi
}

tests_to_run=($(resolve_tests_to_run "$@"))

declare -A BUILD_STATUS
declare -A RUN_STATUS

build_success=0
build_fail=0
run_success=0
run_fail=0

for test_name in "${tests_to_run[@]}"; do
    if [ -z "${TEST_DEFINES[$test_name]:-}" ]; then
        echo "Unknown test: $test_name"
        list_tests
        exit 1
    fi

    test_define=${TEST_DEFINES[$test_name]}

    if build_test "$test_name" "$test_define"; then
        BUILD_STATUS[$test_name]="Success"
        ((build_success++))

        if run_test "$test_name"; then
            RUN_STATUS[$test_name]="Success"
            ((run_success++))
        else
            RUN_STATUS[$test_name]="Runtime Failed"
            ((run_fail++))
        fi
    else
        BUILD_STATUS[$test_name]="Failed"
        RUN_STATUS[$test_name]="Skipped"
        ((build_fail++))
        if [ -n "${TEST_KNOWN_ISSUES[$test_name]:-}" ]; then
            echo "Known RDNA4 issue: ${TEST_KNOWN_ISSUES[$test_name]}"
        fi
    fi

    make -s clean >/dev/null 2>&1 || true
    echo ""
done

echo "========================================="
echo "RDNA4 Test Summary"
echo "========================================="
printf "%-38s %-15s %-15s\n" "Test" "Build" "Runtime"
printf -- "%-38s %-15s %-15s\n" "--------------------------------------" "---------------" "---------------"
for test_name in "${TEST_ORDER[@]}"; do
    if [ -z "${BUILD_STATUS[$test_name]:-}" ] && [ -z "${RUN_STATUS[$test_name]:-}" ]; then
        continue
    fi
    printf "%-38s %-15s %-15s" "$test_name" "${BUILD_STATUS[$test_name]}" "${RUN_STATUS[$test_name]}"
    if [ "${BUILD_STATUS[$test_name]}" = "Failed" ] && [ -n "${TEST_KNOWN_ISSUES[$test_name]:-}" ]; then
        printf "  (%s)" "${TEST_KNOWN_ISSUES[$test_name]}"
    fi
    echo ""
done

echo ""
echo "Totals:"
echo "  ✅ Build success : $build_success"
echo "  ❌ Build failed  : $build_fail"
echo "  ✅ Runtime pass  : $run_success"
echo "  ⚠️  Runtime fail : $run_fail"
echo ""
echo "Done. Logs above show details for each test."
