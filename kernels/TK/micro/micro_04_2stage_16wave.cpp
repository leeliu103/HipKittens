#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

/*************************************************************************/

struct lds_lane_ofs { uint32_t off0, off1; };

// Lane-only prefill 
template <typename U>
__device__ inline lds_lane_ofs make_lds_lane_offsets() {
    static_assert(sizeof(U) == 2, "only 16-bit dtypes");
    constexpr int TR = kittens::TILE_ROW_DIM<U>;
    constexpr int TC = kittens::TILE_COL_DIM<U>;
    constexpr int subtile_stride = (TR * TC * sizeof(U)) / 2;

    const int lane = kittens::laneid() % kittens::WARP_THREADS;
    const int subtile_id = (lane % 32) / 16;           
    const int col_bytes  = (lane / 32) * 16;           
    const int row        = (lane % 16);                

    const int b0   = row * TC * sizeof(U) + col_bytes; 
    const int b1   = b0 + 32;                          
    const int sw0  = ((b0 >> 8) << 4);
    const int sw1  = ((b1 >> 8) << 4);
    const int base = subtile_id * subtile_stride;

    return { uint32_t(base + (b0 ^ sw0)), uint32_t(base + (b1 ^ sw1)) };
}

template <ducks::rt::all RT, ducks::st::all ST>
__device__ inline void load_pc_swizzled(RT& dst, const ST& src, lds_lane_ofs lane) {
    using U = typename ST::dtype;
    static_assert(sizeof(U) == 2, "only 16-bit dtypes");

    constexpr int TR = kittens::TILE_ROW_DIM<U>, TC = kittens::TILE_COL_DIM<U>;
    constexpr int subtile_stride = (TR * TC * sizeof(U)) / 2;
    constexpr int tile_stride    = subtile_stride * 2;                 
    constexpr int row_stride     = tile_stride * ST::underlying_width; 

    // base is the lds pointer for the subtile 
    const uint32_t base = uint32_t(reinterpret_cast<uintptr_t>(&src.data[0]));
    const uint32_t v0   = base + lane.off0;
    const uint32_t v1   = base + lane.off1;

    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
      #pragma unroll
      for (int j = 0; j < RT::width; ++j) {
        const int off = i * row_stride + j * tile_stride;
        asm volatile("ds_read_b128 %0, %1 offset:%2"
          : "=v"(*reinterpret_cast<float4*>(&dst.tiles[i][j].data[0]))
          : "v"(v0), "i"(off) : "memory");
        asm volatile("ds_read_b128 %0, %1 offset:%2"
          : "=v"(*reinterpret_cast<float4*>(&dst.tiles[i][j].data[4]))
          : "v"(v1), "i"(off) : "memory");
      }
    }
}


/*************************************************************************/

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 3;
constexpr int N_BLOCK = 4;
constexpr int DOT_SLICE = 32;

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;

#define NUM_PRODUCER_WORKERS (4)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;
using A_slice = rt_bf<BLOCK_SIZE, DOT_SLICE, row_l>;
using B_slice = rt_bf<BLOCK_SIZE, DOT_SLICE, row_l>;

#define M 192*40
#define K 192*40
#define N 192*40

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
  }

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> a, b;
    gl<bf16, -1, -1, -1, -1> c;
    dim3 grid()  { return dim3(N / NEW_COL_BLOCK_SIZE, M / NEW_ROW_BLOCK_SIZE); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return (120000); }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row> (&As)[2][M_BLOCK] = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row>, 2, M_BLOCK>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row> (&Bs)[2][N_BLOCK] = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row>, 2, N_BLOCK>();
    rt_fl<BLOCK_SIZE, BLOCK_SIZE, accum_col_l> C_accum;

    // L2 cache rate
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS  = gridDim.x * gridDim.y;
    const int NUM_XCDS = 8;  
    wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS);
    const int WGM = 4;  
    const int num_pid_m = ceil_div(M, NEW_ROW_BLOCK_SIZE); // 7680 / 192 = 40
    const int num_pid_n = ceil_div(N, NEW_COL_BLOCK_SIZE); // 7680 / 256 = 30
    const int num_wgid_in_group = WGM * num_pid_n;
    const int group_id     = wgid / num_wgid_in_group;
    const int first_pid_m  = group_id * WGM;
    const int group_size_m = min(num_pid_m - first_pid_m, WGM);
    const int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    const int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    int row = pid_m * M_BLOCK;  
    int col = pid_n * N_BLOCK;  

    // producer and consumer
    const int warp_id = kittens::warpid();
    const int local_warp_id = warp_id % 4;
    const int warp_group_id = kittens::warpgroupid();
    const bool is_producer = (warp_group_id == 0);
    const bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    const int consumer_idx = is_consumer ? warp_group_id - 1 : 0;
    __syncthreads();

    // preswizzled offsets
    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);
    const lds_lane_ofs lane_ofs = make_lds_lane_offsets<bf16>();
    
    // preload
    int tic = 0;
    int toc = 1;
    if (is_producer) {
        #pragma unroll
        for (int m = 0; m < M_BLOCK; m++) {
            G::load<2, false>(As[tic][m], g.a, {0, 0, row + m, 0}, swizzled_offsets_A);
        }
        #pragma unroll
        for (int n = 0; n < N_BLOCK; n++) {
            G::load<2, false>(Bs[tic][n], g.b, {0, 0, col + n, 0}, swizzled_offsets_B);
        }
        asm volatile("s_waitcnt vmcnt(0)");
    }
    __syncthreads();


    // hot loop
    if (is_consumer) {zero(C_accum);}
    int num_tiles = K / BLOCK_SIZE;
    #pragma unroll
    for (int tile = 0; tile < num_tiles-1; ++tile, tic ^= 1, toc ^= 1) {

        if (is_producer) {
            #pragma unroll
            for (int m = 0; m < M_BLOCK; m++) {
                G::load<2, false>(As[toc][m], g.a, {0, 0, row + m, tile + 1}, swizzled_offsets_A);
            }
            #pragma unroll
            for (int n = 0; n < N_BLOCK; n++) {
                G::load<2, false>(Bs[toc][n], g.b, {0, 0, col + n,tile + 1}, swizzled_offsets_B);
            }
        } else {
            A_slice a0, a1; 
            B_slice b0, b1;

            load_pc_swizzled(a0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[tic][consumer_idx], {0,0}), lane_ofs);
            load_pc_swizzled(b0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[tic][local_warp_id], {0,0}), lane_ofs);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);

            load_pc_swizzled(a1, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[tic][consumer_idx], {0,1}), lane_ofs);
            load_pc_swizzled(b1, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[tic][local_warp_id], {0,1}), lane_ofs);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a1, b1, C_accum);
            __builtin_amdgcn_s_setprio(0);
        }
        __builtin_amdgcn_s_setprio(1);
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();

    }
    if (is_consumer) { 
        A_slice a0;
        B_slice b0;
        load_pc_swizzled(a0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[tic][consumer_idx], {0,0}), lane_ofs);
        load_pc_swizzled(b0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[tic][local_warp_id], {0,0}), lane_ofs);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, a0, b0, C_accum);
        __builtin_amdgcn_s_setprio(0);
        load_pc_swizzled(a0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[tic][consumer_idx], {0,1}), lane_ofs);
        load_pc_swizzled(b0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[tic][local_warp_id], {0,1}), lane_ofs);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, a0, b0, C_accum);
        __builtin_amdgcn_s_setprio(0);
    }
    if (is_consumer) {
        store(g.c, C_accum, {0, 0, row + consumer_idx, col + local_warp_id});
    }
}

void dispatch_micro(micro_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
  
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    hipEventRecord(start);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms=0.f; hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start); hipEventDestroy(stop);
  
    // printf("kernel_ms=%.3f\n", ms);
    hipDeviceSynchronize();
  }

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}

