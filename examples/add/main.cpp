#include "kittens.cuh"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int TILE_ROWS = 16;
constexpr int TILE_COLS = 16;
constexpr int ELEMENTS_PER_TILE = TILE_ROWS * TILE_COLS;
constexpr int NUM_WARPS = 1;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

struct hipkittens_add_globals {
    using gl_t = kittens::gl<float, 1, 1, -1, TILE_COLS>;
    gl_t a;
    gl_t b;
    gl_t out;
    int tile_count;
};

__global__ void hipkittens_add_kernel(hipkittens_add_globals g) {
    const int warp_id = kittens::warpid();
    const int tile_idx = blockIdx.x * NUM_WARPS + warp_id;
    if (tile_idx >= g.tile_count) {
        return;
    }

    using tile_t = kittens::rt<float, TILE_ROWS, TILE_COLS>;
    tile_t a_tile;
    tile_t b_tile;
    tile_t out_tile;

    kittens::coord<tile_t> tile_coord(0, 0, tile_idx, 0);
    auto debug_tile = [&](const char *label, const tile_t &tile) {
        if (tile_idx == 0 && kittens::laneid() == 0) {
            printf("%s (tile %d, lane %d):", label, tile_idx, threadIdx.x);
            #pragma unroll
            for (int idx = 0; idx < tile_t::packed_per_base_tile; ++idx) {
                const auto val = tile.tiles[0][0].data[idx];
                printf(" [%d](%f,%f)", idx, static_cast<double>(val.x), static_cast<double>(val.y));
            }
            printf("\n");
        }
    };

    kittens::load<2>(a_tile, g.a, tile_coord);
    kittens::load<2>(b_tile, g.b, tile_coord);
    debug_tile("a_tile after load", a_tile);
    debug_tile("b_tile after load", b_tile);
    kittens::add(out_tile, a_tile, b_tile);
    debug_tile("out_tile after add", out_tile);
    kittens::store(g.out, out_tile, tile_coord);
}

#define HIP_CHECK(cmd)                                                                            \
    do {                                                                                          \
        hipError_t err = (cmd);                                                                   \
        if (err != hipSuccess) {                                                                  \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " -> "                 \
                      << hipGetErrorString(err) << std::endl;                                     \
            std::exit(EXIT_FAILURE);                                                              \
        }                                                                                         \
    } while (0)

}  // namespace

int main() {
    constexpr int tile_count = 32;
    constexpr int total_elements = tile_count * ELEMENTS_PER_TILE;
    const size_t bytes = total_elements * sizeof(float);

    std::vector<float> host_a(total_elements);
    std::vector<float> host_b(total_elements);
    std::vector<float> host_out(total_elements, 0.0f);

    for (int i = 0; i < total_elements; ++i) {
        host_a[i] = static_cast<float>(i);
        host_b[i] = 0.5f * static_cast<float>(i);
    }

    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_out = nullptr;

    HIP_CHECK(hipMalloc(&dev_a, bytes));
    HIP_CHECK(hipMalloc(&dev_b, bytes));
    HIP_CHECK(hipMalloc(&dev_out, bytes));

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    std::cout << "Running on device " << device_id << ": " << props.name << " ("
              << props.gcnArchName << ")" << std::endl;

    if (props.warpSize != kittens::WARP_THREADS) {
        std::cerr << "HipKittens was compiled for warp size " << kittens::WARP_THREADS
                  << " but the active device reports warp size " << props.warpSize << ".\n"
                  << "Rebuild with: make clean && make OFFLOAD_ARCHS=" << props.gcnArchName
                  << " KITTENS_WARP_THREADS=" << props.warpSize << std::endl;
        HIP_CHECK(hipFree(dev_a));
        HIP_CHECK(hipFree(dev_b));
        HIP_CHECK(hipFree(dev_out));
        return EXIT_FAILURE;
    }

    HIP_CHECK(hipMemcpy(dev_a, host_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_b, host_b.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dev_out, 0, bytes));

    const size_t rows_runtime = static_cast<size_t>(tile_count) * TILE_ROWS;

    hipkittens_add_globals globals{
        hipkittens_add_globals::gl_t(dev_a, nullptr, nullptr, rows_runtime, nullptr),
        hipkittens_add_globals::gl_t(dev_b, nullptr, nullptr, rows_runtime, nullptr),
        hipkittens_add_globals::gl_t(dev_out, nullptr, nullptr, rows_runtime, nullptr),
        tile_count};

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim((tile_count + NUM_WARPS - 1) / NUM_WARPS);

    auto explain_invalid_device = [&](const char *phase) {
        std::cerr << "HIP error during " << phase << ": invalid device function.\n"
                  << "Detected GPU '" << props.name << "' with arch '" << props.gcnArchName
                  << "'. Rebuild with:\n  make clean && make OFFLOAD_ARCHS=" << props.gcnArchName
                  << "\n" << std::endl;
    };

    hipLaunchKernelGGL(hipkittens_add_kernel, grid_dim, block_dim, 0, 0, globals);
    hipError_t launch_status = hipGetLastError();
    if (launch_status == hipErrorInvalidDeviceFunction) {
        explain_invalid_device("kernel launch");
    }

    hipError_t sync_status = hipDeviceSynchronize();
    if (sync_status == hipErrorInvalidDeviceFunction) {
        explain_invalid_device("hipDeviceSynchronize");
    }

    if (launch_status == hipErrorInvalidDeviceFunction || sync_status == hipErrorInvalidDeviceFunction) {
        HIP_CHECK(hipFree(dev_a));
        HIP_CHECK(hipFree(dev_b));
        HIP_CHECK(hipFree(dev_out));
        return EXIT_FAILURE;
    }

    HIP_CHECK(launch_status);
    HIP_CHECK(sync_status);

    HIP_CHECK(hipMemcpy(host_out.data(), dev_out, bytes, hipMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < total_elements; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (std::fabs(expected - host_out[i]) > 1e-4f) {
            std::cerr << "Mismatch at " << i << ": got " << host_out[i] << " expected " << expected
                      << std::endl;
            success = false;
            break;
        }
    }

    HIP_CHECK(hipFree(dev_a));
    HIP_CHECK(hipFree(dev_b));
    HIP_CHECK(hipFree(dev_out));

    if (success) {
        std::cout << "HipKittens vector add completed successfully for " << total_elements
                  << " elements ("
                  << tile_count << " tiles of " << TILE_ROWS << "x" << TILE_COLS << ")." << std::endl;
        std::cout << "Sample results:";
        for (int i = 0; i < 8; ++i) {
            std::cout << " (" << host_a[i] << " + " << host_b[i] << " = " << host_out[i] << ")";
        }
        std::cout << std::endl;
        return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}
