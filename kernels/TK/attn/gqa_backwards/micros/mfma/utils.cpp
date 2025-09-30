#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#define STR2(x) #x
#define STR(x)  STR2(x)

#define DS_READ_B32_FUNC(VGPR) \
__device__ __forceinline__ void ds_read_b32_v##VGPR(const uint32_t smem_ptr) { \
  asm volatile("ds_read_b32 v" STR(VGPR) ", %0 offset:0" \
               :: "v"(smem_ptr) \
               : "memory", "v" STR(VGPR)); \
}

#define DS_READ_B64_FUNC(VGPR0, VGPR1) \
__device__ __forceinline__ void ds_read_b64_v##VGPR0##_v##VGPR1(const uint32_t smem_ptr) { \
  asm volatile("ds_read_b64 v[" STR(VGPR0) ":" STR(VGPR1) "], %0 offset:0" \
               :: "v"(smem_ptr) \
               : "memory", "v" STR(VGPR0), "v" STR(VGPR1)); \
}

#define DS_READ_B128_FUNC(VGPR0, VGPR1, VGPR2, VGPR3) \
__device__ __forceinline__ void ds_read_b128_v##VGPR0##_v##VGPR1##_v##VGPR2##_v##VGPR3(const uint32_t smem_ptr) { \
  asm volatile("ds_read_b128 v[" STR(VGPR0) ":" STR(VGPR3) "], %0 offset:0" \
               :: "v"(smem_ptr) \
               : "memory", "v" STR(VGPR0), "v" STR(VGPR1), "v" STR(VGPR2), "v" STR(VGPR3)); \
}

#define DS_READ_B128_AGPR_FUNC(AGPR0, AGPR1, AGPR2, AGPR3) \
__device__ __forceinline__ void ds_read_b128_agpr_a##AGPR0##_a##AGPR1##_a##AGPR2##_a##AGPR3(const uint32_t smem_ptr) { \
  asm volatile("ds_read_b128 a[" STR(AGPR0) ":" STR(AGPR3) "], %0 offset:0" \
               :: "v"(smem_ptr) \
               : "memory", "a" STR(AGPR0), "a" STR(AGPR1), "a" STR(AGPR2), "a" STR(AGPR3)); \
}

#define READ_B64_FUNC(VGPR0, VGPR1) \
__device__ __forceinline__ void v_mov_b64_v##VGPR0##_v##VGPR1(uint32_t& result1, uint32_t& result2) { \
  asm volatile("v_mov_b32 %0, v" STR(VGPR0) : "=v"(result1) :: "v" STR(VGPR0)); \
  asm volatile("v_mov_b32 %0, v" STR(VGPR1) : "=v"(result2) :: "v" STR(VGPR1)); \
}

#define READ_B128_FUNC(VGPR0, VGPR1, VGPR2, VGPR3) \
__device__ __forceinline__ void v_mov_b128_v##VGPR0##_v##VGPR1##_v##VGPR2##_v##VGPR3(uint32_t& result1, uint32_t& result2, uint32_t& result3, uint32_t& result4) { \
  asm volatile("v_mov_b32 %0, v" STR(VGPR0) : "=v"(result1) :: "v" STR(VGPR0)); \
  asm volatile("v_mov_b32 %0, v" STR(VGPR1) : "=v"(result2) :: "v" STR(VGPR1)); \
  asm volatile("v_mov_b32 %0, v" STR(VGPR2) : "=v"(result3) :: "v" STR(VGPR2)); \
  asm volatile("v_mov_b32 %0, v" STR(VGPR3) : "=v"(result4) :: "v" STR(VGPR3)); \
}

#define ACCVGPR_READ_B128_FUNC(AGPR0, AGPR1, AGPR2, AGPR3) \
__device__ __forceinline__ void accvgpr_read_b128_a##AGPR0##_a##AGPR1##_a##AGPR2##_a##AGPR3(uint32_t& result1, uint32_t& result2, uint32_t& result3, uint32_t& result4) { \
  asm volatile("v_accvgpr_read_b32 %0, a" STR(AGPR0) : "=v"(result1) :: "a" STR(AGPR0)); \
  asm volatile("v_accvgpr_read_b32 %0, a" STR(AGPR1) : "=v"(result2) :: "a" STR(AGPR1)); \
  asm volatile("v_accvgpr_read_b32 %0, a" STR(AGPR2) : "=v"(result3) :: "a" STR(AGPR2)); \
  asm volatile("v_accvgpr_read_b32 %0, a" STR(AGPR3) : "=v"(result4) :: "a" STR(AGPR3)); \
}

#define MFMA_F32_16x16x32_BF16_AGPR_AGPR_FUNC(AGPR0, AGPR1, AGPR2, AGPR3, AGPR4, AGPR5, AGPR6, AGPR7) \
__device__ __forceinline__ void mfma_f32_16x16x32_bf16_agpr_agpr_a##AGPR0##_a##AGPR1##_a##AGPR2##_a##AGPR3##_a##AGPR4##_a##AGPR5##_a##AGPR6##_a##AGPR7(float4& D, float4& C) { \
  typedef float float4_t __attribute__((ext_vector_type(4))); \
  float4_t& d_vec = reinterpret_cast<float4_t&>(D); \
  const float4_t& c_vec = reinterpret_cast<const float4_t&>(C); \
  asm volatile("v_mfma_f32_16x16x32_bf16 %0, a[" STR(AGPR0) ":" STR(AGPR3) "], a[" STR(AGPR4) ":" STR(AGPR7) "], %1" \
               : "=v"(d_vec) \
               : "v"(c_vec) \
               : "a" STR(AGPR0), "a" STR(AGPR1), "a" STR(AGPR2), "a" STR(AGPR3), "a" STR(AGPR4), "a" STR(AGPR5), "a" STR(AGPR6), "a" STR(AGPR7)); \
}

#define MFMA_F32_16x16x32_BF16_AGPR_VGPR_FUNC(AGPR0, AGPR1, AGPR2, AGPR3, VGPR0, VGPR1, VGPR2, VGPR3) \
__device__ __forceinline__ void mfma_f32_16x16x32_bf16_agpr_vgpr_a##AGPR0##_a##AGPR1##_a##AGPR2##_a##AGPR3##_v##VGPR0##_v##VGPR1##_v##VGPR2##_v##VGPR3(float4& D, float4& C) { \
  typedef float float4_t __attribute__((ext_vector_type(4))); \
  float4_t& d_vec = reinterpret_cast<float4_t&>(D); \
  const float4_t& c_vec = reinterpret_cast<const float4_t&>(C); \
  asm volatile("v_mfma_f32_16x16x32_bf16 %0, a[" STR(AGPR0) ":" STR(AGPR3) "], v[" STR(VGPR0) ":" STR(VGPR3) "], %1" \
               : "=v"(d_vec) \
               : "v"(c_vec) \
               : "a" STR(AGPR0), "a" STR(AGPR1), "a" STR(AGPR2), "a" STR(AGPR3), "v" STR(VGPR0), "v" STR(VGPR1), "v" STR(VGPR2), "v" STR(VGPR3)); \
}

#define MFMA_F32_16x16x32_BF16_VGPR_AGPR_FUNC(VGPR0, VGPR1, VGPR2, VGPR3, AGPR0, AGPR1, AGPR2, AGPR3) \
__device__ __forceinline__ void mfma_f32_16x16x32_bf16_vgpr_agpr_v##VGPR0##_v##VGPR1##_v##VGPR2##_v##VGPR3##_a##AGPR0##_a##AGPR1##_a##AGPR2##_a##AGPR3(float4& D, float4& C) { \
  typedef float float4_t __attribute__((ext_vector_type(4))); \
  float4_t& d_vec = reinterpret_cast<float4_t&>(D); \
  const float4_t& c_vec = reinterpret_cast<const float4_t&>(C); \
  asm volatile("v_mfma_f32_16x16x32_bf16 %0, v[" STR(VGPR0) ":" STR(VGPR3) "], a[" STR(AGPR0) ":" STR(AGPR3) "], %1" \
               : "=v"(d_vec) \
               : "v"(c_vec) \
               : "v" STR(VGPR0), "v" STR(VGPR1), "v" STR(VGPR2), "v" STR(VGPR3), "a" STR(AGPR0), "a" STR(AGPR1), "a" STR(AGPR2), "a" STR(AGPR3)); \
}

#define MFMA_F32_16x16x32_BF16_VGPR_VGPR_FUNC(VGPR0, VGPR1, VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7) \
__device__ __forceinline__ void mfma_f32_16x16x32_bf16_vgpr_vgpr_v##VGPR0##_v##VGPR1##_v##VGPR2##_v##VGPR3##_v##VGPR4##_v##VGPR5##_v##VGPR6##_v##VGPR7(float4& D, float4& C) { \
  typedef float float4_t __attribute__((ext_vector_type(4))); \
  float4_t& d_vec = reinterpret_cast<float4_t&>(D); \
  const float4_t& c_vec = reinterpret_cast<const float4_t&>(C); \
  asm volatile("v_mfma_f32_16x16x32_bf16 %0, v[" STR(VGPR0) ":" STR(VGPR3) "], v[" STR(VGPR4) ":" STR(VGPR7) "], %1" \
               : "=v"(d_vec) \
               : "v"(c_vec) \
               : "v" STR(VGPR0), "v" STR(VGPR1), "v" STR(VGPR2), "v" STR(VGPR3), "v" STR(VGPR4), "v" STR(VGPR5), "v" STR(VGPR6), "v" STR(VGPR7)); \
}

DS_READ_B128_FUNC(10, 11, 12, 13)
DS_READ_B128_FUNC(20, 21, 22, 23)
DS_READ_B128_FUNC(30, 31, 32, 33)
DS_READ_B128_FUNC(40, 41, 42, 43)

DS_READ_B128_AGPR_FUNC(10, 11, 12, 13)
DS_READ_B128_AGPR_FUNC(20, 21, 22, 23)
DS_READ_B128_AGPR_FUNC(30, 31, 32, 33)
DS_READ_B128_AGPR_FUNC(40, 41, 42, 43)

DS_READ_B64_FUNC(10, 11)
DS_READ_B64_FUNC(20, 21)
DS_READ_B64_FUNC(30, 31)
DS_READ_B64_FUNC(40, 41)

READ_B128_FUNC(10, 11, 12, 13)
READ_B128_FUNC(20, 21, 22, 23)
READ_B128_FUNC(30, 31, 32, 33)
READ_B128_FUNC(40, 41, 42, 43)

READ_B64_FUNC(10, 11)
READ_B64_FUNC(20, 21)
READ_B64_FUNC(30, 31)
READ_B64_FUNC(40, 41)

ACCVGPR_READ_B128_FUNC(10, 11, 12, 13)
ACCVGPR_READ_B128_FUNC(20, 21, 22, 23)
ACCVGPR_READ_B128_FUNC(30, 31, 32, 33)
ACCVGPR_READ_B128_FUNC(40, 41, 42, 43)

MFMA_F32_16x16x32_BF16_AGPR_AGPR_FUNC(10, 11, 12, 13, 20, 21, 22, 23)
MFMA_F32_16x16x32_BF16_AGPR_AGPR_FUNC(20, 21, 22, 23, 30, 31, 32, 33)
MFMA_F32_16x16x32_BF16_AGPR_AGPR_FUNC(30, 31, 32, 33, 40, 41, 42, 43)

MFMA_F32_16x16x32_BF16_AGPR_VGPR_FUNC(10, 11, 12, 13, 10, 11, 12, 13)
MFMA_F32_16x16x32_BF16_AGPR_VGPR_FUNC(20, 21, 22, 23, 20, 21, 22, 23)
MFMA_F32_16x16x32_BF16_AGPR_VGPR_FUNC(30, 31, 32, 33, 30, 31, 32, 33)
MFMA_F32_16x16x32_BF16_AGPR_VGPR_FUNC(40, 41, 42, 43, 40, 41, 42, 43)

MFMA_F32_16x16x32_BF16_VGPR_AGPR_FUNC(10, 11, 12, 13, 10, 11, 12, 13)
MFMA_F32_16x16x32_BF16_VGPR_AGPR_FUNC(20, 21, 22, 23, 20, 21, 22, 23)
MFMA_F32_16x16x32_BF16_VGPR_AGPR_FUNC(30, 31, 32, 33, 30, 31, 32, 33)
MFMA_F32_16x16x32_BF16_VGPR_AGPR_FUNC(40, 41, 42, 43, 40, 41, 42, 43)

MFMA_F32_16x16x32_BF16_VGPR_VGPR_FUNC(10, 11, 12, 13, 20, 21, 22, 23)
MFMA_F32_16x16x32_BF16_VGPR_VGPR_FUNC(20, 21, 22, 23, 30, 31, 32, 33)
MFMA_F32_16x16x32_BF16_VGPR_VGPR_FUNC(30, 31, 32, 33, 40, 41, 42, 43)

template<int VGPR_START>
__device__ __forceinline__ void ds_read_b64(const uint32_t smem_ptr) {
  if constexpr (VGPR_START == 10) {
    ds_read_b64_v10_v11(smem_ptr);
  } else if constexpr (VGPR_START == 20) {
    ds_read_b64_v20_v21(smem_ptr);
  } else if constexpr (VGPR_START == 30) {
    ds_read_b64_v30_v31(smem_ptr);
  } else if constexpr (VGPR_START == 40) {
    ds_read_b64_v40_v41(smem_ptr);
  }
}

template<int VGPR_START>
__device__ __forceinline__ void ds_read_b128(const uint32_t smem_ptr) {
  if constexpr (VGPR_START == 10) {
    ds_read_b128_v10_v11_v12_v13(smem_ptr);
  } else if constexpr (VGPR_START == 20) {
    ds_read_b128_v20_v21_v22_v23(smem_ptr);
  } else if constexpr (VGPR_START == 30) {
    ds_read_b128_v30_v31_v32_v33(smem_ptr);
  } else if constexpr (VGPR_START == 40) {
    ds_read_b128_v40_v41_v42_v43(smem_ptr);
  }
}

template<int AGPR_START>
__device__ __forceinline__ void ds_read_b128_agpr(const uint32_t smem_ptr) {
  if constexpr (AGPR_START == 10) {
    ds_read_b128_agpr_a10_a11_a12_a13(smem_ptr);
  } else if constexpr (AGPR_START == 20) {
    ds_read_b128_agpr_a20_a21_a22_a23(smem_ptr);
  } else if constexpr (AGPR_START == 30) {
    ds_read_b128_agpr_a30_a31_a32_a33(smem_ptr);
  } else if constexpr (AGPR_START == 40) {
    ds_read_b128_agpr_a40_a41_a42_a43(smem_ptr);
  }
}

template<int VGPR_START>
__device__ __forceinline__ void v_mov_b64(uint32_t& result1, uint32_t& result2) {
  if constexpr (VGPR_START == 10) {
    v_mov_b64_v10_v11(result1, result2);
  } else if constexpr (VGPR_START == 20) {
    v_mov_b64_v20_v21(result1, result2);
  } else if constexpr (VGPR_START == 30) {
    v_mov_b64_v30_v31(result1, result2);
  } else if constexpr (VGPR_START == 40) {
    v_mov_b64_v40_v41(result1, result2);
  }
}

template<int VGPR_START>
__device__ __forceinline__ void v_mov_b128(uint32_t& result1, uint32_t& result2, uint32_t& result3, uint32_t& result4) {
  if constexpr (VGPR_START == 10) {
    v_mov_b128_v10_v11_v12_v13(result1, result2, result3, result4);
  } else if constexpr (VGPR_START == 20) {
    v_mov_b128_v20_v21_v22_v23(result1, result2, result3, result4);
  } else if constexpr (VGPR_START == 30) {
    v_mov_b128_v30_v31_v32_v33(result1, result2, result3, result4);
  } else if constexpr (VGPR_START == 40) {
    v_mov_b128_v40_v41_v42_v43(result1, result2, result3, result4);
  }
}

template<int AGPR_START>
__device__ __forceinline__ void accvgpr_read_b128(uint32_t& result1, uint32_t& result2, uint32_t& result3, uint32_t& result4) {
  if constexpr (AGPR_START == 10) {
    accvgpr_read_b128_a10_a11_a12_a13(result1, result2, result3, result4);
  } else if constexpr (AGPR_START == 20) {
    accvgpr_read_b128_a20_a21_a22_a23(result1, result2, result3, result4);
  } else if constexpr (AGPR_START == 30) {
    accvgpr_read_b128_a30_a31_a32_a33(result1, result2, result3, result4);
  } else if constexpr (AGPR_START == 40) {
    accvgpr_read_b128_a40_a41_a42_a43(result1, result2, result3, result4);
  }
}

template<int AGPR_START_A, int AGPR_START_B>
__device__ __forceinline__ void mfma_f32_16x16x32_bf16_agpr_agpr(float4& D, float4& C) {
  if constexpr (AGPR_START_A == 10 && AGPR_START_B == 20) {
    mfma_f32_16x16x32_bf16_agpr_agpr_a10_a11_a12_a13_a20_a21_a22_a23(D, C);
  } else if constexpr (AGPR_START_A == 20 && AGPR_START_B == 30) {
    mfma_f32_16x16x32_bf16_agpr_agpr_a20_a21_a22_a23_a30_a31_a32_a33(D, C);
  } else if constexpr (AGPR_START_A == 30 && AGPR_START_B == 40) {
    mfma_f32_16x16x32_bf16_agpr_agpr_a30_a31_a32_a33_a40_a41_a42_a43(D, C);
  }
}

template<int AGPR_START_A, int VGPR_START_B>
__device__ __forceinline__ void mfma_f32_16x16x32_bf16_agpr_vgpr(float4& D, float4& C) {
  if constexpr (AGPR_START_A == 10 && VGPR_START_B == 10) {
    mfma_f32_16x16x32_bf16_agpr_vgpr_a10_a11_a12_a13_v10_v11_v12_v13(D, C);
  } else if constexpr (AGPR_START_A == 20 && VGPR_START_B == 20) {
    mfma_f32_16x16x32_bf16_agpr_vgpr_a20_a21_a22_a23_v20_v21_v22_v23(D, C);
  } else if constexpr (AGPR_START_A == 30 && VGPR_START_B == 30) {
    mfma_f32_16x16x32_bf16_agpr_vgpr_a30_a31_a32_a33_v30_v31_v32_v33(D, C);
  } else if constexpr (AGPR_START_A == 40 && VGPR_START_B == 40) {
    mfma_f32_16x16x32_bf16_agpr_vgpr_a40_a41_a42_a43_v40_v41_v42_v43(D, C);
  }
}

template<int VGPR_START_A, int AGPR_START_B>
__device__ __forceinline__ void mfma_f32_16x16x32_bf16_vgpr_agpr(float4& D, float4& C) {
  if constexpr (VGPR_START_A == 10 && AGPR_START_B == 10) {
    mfma_f32_16x16x32_bf16_vgpr_agpr_v10_v11_v12_v13_a10_a11_a12_a13(D, C);
  } else if constexpr (VGPR_START_A == 20 && AGPR_START_B == 20) {
    mfma_f32_16x16x32_bf16_vgpr_agpr_v20_v21_v22_v23_a20_a21_a22_a23(D, C);
  } else if constexpr (VGPR_START_A == 30 && AGPR_START_B == 30) {
    mfma_f32_16x16x32_bf16_vgpr_agpr_v30_v31_v32_v33_a30_a31_a32_a33(D, C);
  } else if constexpr (VGPR_START_A == 40 && AGPR_START_B == 40) {
    mfma_f32_16x16x32_bf16_vgpr_agpr_v40_v41_v42_v43_a40_a41_a42_a43(D, C);
  }
}

template<int VGPR_START_A, int VGPR_START_B>
__device__ __forceinline__ void mfma_f32_16x16x32_bf16_vgpr_vgpr(float4& D, float4& C) {
  if constexpr (VGPR_START_A == 10 && VGPR_START_B == 20) {
    mfma_f32_16x16x32_bf16_vgpr_vgpr_v10_v11_v12_v13_v20_v21_v22_v23(D, C);
  } else if constexpr (VGPR_START_A == 20 && VGPR_START_B == 30) {
    mfma_f32_16x16x32_bf16_vgpr_vgpr_v20_v21_v22_v23_v30_v31_v32_v33(D, C);
  } else if constexpr (VGPR_START_A == 30 && VGPR_START_B == 40) {
    mfma_f32_16x16x32_bf16_vgpr_vgpr_v30_v31_v32_v33_v40_v41_v42_v43(D, C);
  }
}