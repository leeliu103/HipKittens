#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"

using namespace kittens;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)
#define ASSEMBLY_MODE 1

struct micro_globals {
  gl<bf16, -1, -1, -1, -1> A;
  gl<bf16, -1, -1, -1, -1> B;
  gl<float, -1, -1, -1, -1> C;
  dim3 grid() {return dim3(1); }
  dim3 block() {return dim3(NUM_THREADS);}
  size_t dynamic_shared_memory() {return MAX_SHARED_MEMORY;}
};

__device__ __forceinline__ void process_via_assembly(const micro_globals g) {

  const __hip_bfloat16* A = g.A.raw_ptr;
  const __hip_bfloat16* B = g.B.raw_ptr;
  float* C = g.C.raw_ptr;
  
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);
  st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&A_smem) = al.allocate<st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>>();
  st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&B_smem) = al.allocate<st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>>();
  
  const int laneid = kittens::laneid();

  // Load A and B from global memory to shared memroy
  const int row = laneid % 16;
  const int col = laneid / 16 * 8;

  // Each thread writes two input values to shared memory
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    A_smem.data[row * 32 + col + i] = A[row * 32 + col + i];
    B_smem.data[row * 32 + col + i] = B[row * 32 + col + i];
  }
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  // Read two floats from shared memory to VGPR pair using ds_read_b64
  // ds_read_b128<VGPR_ID>(threadIdx.x * 8);
  ds_read_b128_agpr<10>(reinterpret_cast<uintptr_t>(&A_smem.data[row * 32 + col]));
  ds_read_b128_agpr<20>(reinterpret_cast<uintptr_t>(&B_smem.data[row * 32 + col]));
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  float4 C_mfma = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 D_mfma = {0.0f, 0.0f, 0.0f, 0.0f};

  mfma_f32_16x16x32_bf16_agpr_agpr<10, 20>(D_mfma, C_mfma);
  asm volatile("s_nop 2" ::: "memory");

  const int output_row = laneid / 16 * 4;
  const int output_col = laneid % 16;

  C[(output_row + 0) * 16 + output_col] = D_mfma.x;
  C[(output_row + 1) * 16 + output_col] = D_mfma.y;
  C[(output_row + 2) * 16 + output_col] = D_mfma.z;
  C[(output_row + 3) * 16 + output_col] = D_mfma.w;
}

__device__ __forceinline__ void process_via_assembly_tk(const micro_globals g) {

  const __hip_bfloat16* A = g.A.raw_ptr;
  const __hip_bfloat16* B = g.B.raw_ptr;
  float* C = g.C.raw_ptr;
  
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);
  st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&A_smem) = al.allocate<st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>>();
  st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&B_smem) = al.allocate<st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>>();
  
  using A_ranges = ducks::rt_asm::type_list<ducks::rt_asm::range<10, 13>>;
  using B_ranges = ducks::rt_asm::type_list<ducks::rt_asm::range<20, 23>>;
  rt_asm<bf16, 16, 32, row_l, mfma_16x16x32, A_ranges> A_reg;
  rt_asm<bf16, 16, 32, row_l, mfma_16x16x32, B_ranges> B_reg;

  // Each thread writes A and B to shared memory
  load(A_smem, g.A, {0, 0, 0, 0});
  load(B_smem, g.B, {0, 0, 0, 0});
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  load_asm(A_reg, A_smem);
  load_asm(B_reg, B_smem);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  // Read two floats from shared memory to VGPR pair using ds_read_b64
  rt<float, 16, 16, accum_col_l, mfma_16x16x32> C_reg;
  zero(C_reg);
  mma_ABt_asm(C_reg, A_reg, B_reg, C_reg);
  asm volatile("s_nop 2" ::: "memory");

  store(g.C, C_reg, {0, 0, 0, 0});
}

__device__ __forceinline__ void process_via_tk(const micro_globals g) {
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);
  st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&A_smem) = al.allocate<st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>>();
  st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&B_smem) = al.allocate<st_bf<16, 32, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>>();

  rt<bf16, 16, 32, row_l, mfma_16x16x32> A_reg;
  rt<bf16, 16, 32, row_l, mfma_16x16x32> B_reg;

  // Each thread writes A and B to shared memory
  load(A_smem, g.A, {0, 0, 0, 0});
  load(B_smem, g.B, {0, 0, 0, 0});
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  load(A_reg, A_smem);
  load(B_reg, B_smem);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  rt<float, 16, 16, accum_col_l, mfma_16x16x32> C_reg;
  zero(C_reg);
  mma_ABt(C_reg, A_reg, B_reg, C_reg);

  store(g.C, C_reg, {0, 0, 0, 0});
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
  if (ASSEMBLY_MODE == 0) {
    process_via_tk(g);
  } else {
    // process_via_assembly(g);
    process_via_assembly_tk(g);
  }
}


void dispatch_micro(micro_globals g) {
  unsigned long mem_size = g.dynamic_shared_memory();
  hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
  micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
  hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
  m.doc() = "tk_kernel python module";
  py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::A, &micro_globals::B, &micro_globals::C);
}


