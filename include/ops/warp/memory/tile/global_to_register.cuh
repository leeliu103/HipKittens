/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

/**
 * @brief Load data from a source array into a row-major layout tile.
 *
 * @tparam RT The row-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param idx[in] The index of the tile to load data from.
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T2 = RT::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    constexpr int packed_per_stride = RT::base_tile_stride / packing;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    static_assert(!std::is_same_v<typename kittens::base_types::packing<typename RT::dtype>::unpacked_type, fp8e4m3>, "Unsupported type for load");

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();

    const int row_offset = laneid%(dst.base_tile_rows);
    const int col_offset = dst.base_tile_stride*(laneid/dst.base_tile_rows);

    uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(U);
    bool has_buffer_resource = false;
    i32x4 buffer_srsrc{};
    auto get_buffer_srsrc = [&]() {
        if (!has_buffer_resource) {
            std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src_ptr);
            std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
            buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
            buffer_srsrc = std::bit_cast<i32x4>(br);
            has_buffer_resource = true;
        }
        return buffer_srsrc;
    };
    constexpr bool supports_buffer_intrinsics =
        std::is_same_v<U2, bf16_2> || std::is_same_v<U2, half_2> || std::is_same_v<U2, float2>;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        const int row = dst.base_tile_rows*i + row_offset;
        
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            #pragma unroll
            for(int k = 0; k < dst.base_tile_num_strides; k++) {
                const int col = dst.base_tile_cols*j + col_offset + k*dst.base_tile_elements_per_stride_group;

                auto store_lane = [&](const U2 *lane_vals) {
                    #pragma unroll
                    for(int l = 0; l < packed_per_stride; l++) {
                        int idx = l + k * packed_per_stride;
                        dst.tiles[i][j].data[idx] = base_types::convertor<T2, U2>::convert(lane_vals[l]);
                    }
                };
                auto load_with_intrinsics = [&]() {
                    if constexpr (supports_buffer_intrinsics) {
                        const uint32_t offset_bytes = (row*row_stride + col) * sizeof(U);
                        if constexpr (std::is_same_v<U2, bf16_2> || std::is_same_v<U2, half_2>) {

                            // Use buffer_load_b64 for stride == 4, dtype == bf16
                            if constexpr (RT::base_tile_stride == 4) {
                                float2 loaded = std::bit_cast<float2>(llvm_amdgcn_raw_buffer_load_b64(
                                    get_buffer_srsrc(),
                                    offset_bytes,
                                    0,
                                    0
                                ));
                                store_lane(reinterpret_cast<const U2*>(&loaded));
                            // Use buffer_load_b128 for stride == 8, dtype == bf16
                            } else if constexpr (RT::base_tile_stride == 8) {
                                float4 loaded = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                                    get_buffer_srsrc(),
                                    offset_bytes,
                                    0,
                                    0
                                ));
                                store_lane(reinterpret_cast<const U2*>(&loaded));
                            } else {
                                static_assert(false, "Unsupported stride");
                            }

                        } else if constexpr (std::is_same_v<U2, float2>) {

                            // Use buffer_load_b128 for stride == 4, dtype == float
                            if constexpr (RT::base_tile_stride == 4) {
                                float4 loaded = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                                    get_buffer_srsrc(),
                                    offset_bytes,
                                    0,
                                    0
                                ));
                                store_lane(reinterpret_cast<const U2*>(&loaded));
                            // Use buffer_load_b128 for stride == 8, dtype == float
                            } else if constexpr (RT::base_tile_stride == 8) {
                                float4 loaded[2];
                                loaded[0] = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                                    get_buffer_srsrc(),
                                    offset_bytes,
                                    0,
                                    0
                                ));
                                loaded[1] = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                                    get_buffer_srsrc(),
                                    (row*row_stride + col + 4) * sizeof(U),
                                    0,
                                    0
                                ));
                                store_lane(reinterpret_cast<const U2*>(loaded));
                            } else {
                                static_assert(false, "Unsupported stride");
                            }
                        }
                    }
                };

                const U *lane_src = src_ptr + row * row_stride + col;
                auto load_with_memcpy = [&]() {
                    U2 lane_vals[packed_per_stride];
                    __builtin_memcpy(lane_vals, lane_src, sizeof(U2) * packed_per_stride);
                    store_lane(lane_vals);
                };
                if constexpr (WARP_THREADS == 32) {
                    load_with_memcpy();
                } else {
                    static_assert(supports_buffer_intrinsics, "Unsupported type");
                    load_with_intrinsics();
                }
            }
        }
    }
}

/**
 * @brief Load data from a source array into a column-major layout tile.
 *
 * @tparam RT The column-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using T2 = base_types::packing<typename RT::dtype>::packed_type;
    using U = typename GL::dtype;

    static_assert(!std::is_same_v<T, fp8e4m3>, "Unsupported type for load/store");

    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    
    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();

    const int row_offset = dst.base_tile_stride * (laneid / dst.base_tile_cols);
    const int col_offset = laneid % dst.base_tile_cols;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.base_tile_num_strides; k++) {
                int row = i*dst.base_tile_rows + row_offset + k*dst.base_tile_elements_per_stride_group;
                int col = j*dst.base_tile_cols + col_offset;

                #pragma unroll
                for (int l = 0; l < dst.base_tile_stride / packing; l++) {
                    int idx = l + k * dst.base_tile_stride / packing;
                    dst.tiles[i][j].data[idx].x = base_types::convertor<T, U>::convert(src_ptr[(row+l * 2)*row_stride + col]);
                    dst.tiles[i][j].data[idx].y = base_types::convertor<T, U>::convert(src_ptr[(row+l * 2 + 1)*row_stride + col]);
                }
            }
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    load<2, RT, GL>(dst, src, idx);
}

/**
 * @brief Store data from a register tile to a destination array in global memory with a row-major layout.
 *
 * @tparam RT The register tile type with a row-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using T = base_types::packing<T2>::unpacked_type;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    constexpr int packed_per_stride = RT::base_tile_stride / packing;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid();

    const int row_offset = laneid%(src.base_tile_rows);
    const int col_offset = src.base_tile_stride*(laneid/src.base_tile_rows);

    uint32_t buffer_size = dst.batch() * dst.depth() * dst.rows() * dst.cols() * sizeof(U);
    bool has_buffer_resource = false;
    i32x4 buffer_srsrc{};
    auto get_buffer_srsrc = [&]() {
        if (!has_buffer_resource) {
            std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
            std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
            buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
            buffer_srsrc = std::bit_cast<i32x4>(br);
            has_buffer_resource = true;
        }
        return buffer_srsrc;
    };
    constexpr bool supports_buffer_intrinsics =
        std::is_same_v<U2, bf16_2> || std::is_same_v<U2, half_2> || std::is_same_v<U2, float2>;

    U2 tmp[packed_per_stride];

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = src.base_tile_rows*i + row_offset;
        
        #pragma unroll
        for(int j = 0; j < src.width; j++) {

            #pragma unroll
            for(int k = 0; k < src.base_tile_num_strides; k++) {
                int col = src.base_tile_cols*j + col_offset + k*src.base_tile_elements_per_stride_group;
                #pragma unroll
                for(int l = 0; l < packed_per_stride; l++) {
                    int idx = l + k * packed_per_stride;
                    tmp[l] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[idx]);
                }
                auto store_with_intrinsics = [&]() {
                    if constexpr (supports_buffer_intrinsics) {
                        const uint32_t offset_bytes = (row*row_stride + col) * sizeof(U);
                        if constexpr (std::is_same_v<U2, bf16_2> || std::is_same_v<U2, half_2>) {

                            // use buffer_store_b64 for stride == 4, dtype == bf16
                            if constexpr (RT::base_tile_stride == 4) {
                                uint64_t val = *reinterpret_cast<uint64_t*>(tmp);
                                llvm_amdgcn_raw_buffer_store_b64(
                                    val,
                                    get_buffer_srsrc(),
                                    offset_bytes,
                                    0,
                                    0
                                );
                            // use buffer_store_b128 for stride == 8, dtype == bf16
                            } else if constexpr (RT::base_tile_stride == 8) {
                                __uint128_t val = *reinterpret_cast<__uint128_t*>(tmp);
                                llvm_amdgcn_raw_buffer_store_b128(
                                    val,
                                    get_buffer_srsrc(),
                                    offset_bytes,
                                    0,
                                    0
                                );
                            } else {
                                static_assert(false, "Unsupported stride");
                            }
                        } else if constexpr (std::is_same_v<U2, float2>) {

                            // use one buffer_store_b128 for stride == 4, dtype == float
                            if constexpr (RT::base_tile_stride == 4) {
                                __uint128_t val = *reinterpret_cast<__uint128_t*>(tmp);
                                llvm_amdgcn_raw_buffer_store_b128(
                                    val,
                                    get_buffer_srsrc(),
                                    offset_bytes,
                                    0,
                                    0
                                );
                            // use two buffer_store_b128 for stride == 8, dtype == float
                            } else if constexpr (RT::base_tile_stride == 8) {
                                __uint128_t val0 = *reinterpret_cast<__uint128_t*>(tmp);
                                llvm_amdgcn_raw_buffer_store_b128(
                                    val0,
                                    get_buffer_srsrc(),
                                    offset_bytes,
                                    0,
                                    0
                                );
                                __uint128_t val1 = *(reinterpret_cast<__uint128_t*>(tmp) + 1);
                                llvm_amdgcn_raw_buffer_store_b128(
                                    val1,
                                    get_buffer_srsrc(),
                                    (row*row_stride + col + 4) * sizeof(U),
                                    0,
                                    0
                                );
                            } else {
                                static_assert(false, "Unsupported stride");
                            }
                        }
                    }
                };

                if constexpr (WARP_THREADS == 32) {
                    U *lane_dst = dst_ptr + row * row_stride + col;
                    __builtin_memcpy(lane_dst, tmp, sizeof(U2) * packed_per_stride);
                } else {
                    static_assert(supports_buffer_intrinsics, "Unsupported type");
                    store_with_intrinsics();
                }
            }
        }
    }
}


/**
 * @brief Store data from a register tile to a destination array in global memory with a column-major layout.
 *
 * @tparam RT The register tile type with a column-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();

    static_assert(!std::is_same_v<T, fp8e4m3>, "Unsupported type for load/store");

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int laneid = kittens::laneid();

    const int row_offset = src.base_tile_stride*(laneid/src.base_tile_cols);
    const int col_offset = laneid%src.base_tile_cols;

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int col = j*src.base_tile_cols + col_offset;
            #pragma unroll
            for(int k = 0; k < src.base_tile_num_strides; k++) {
                int row = i*src.base_tile_rows + row_offset + k*src.base_tile_elements_per_stride_group;
                #pragma unroll
                for(int l = 0; l < src.base_tile_stride / packing; l++) {
                    int idx = l + k * src.base_tile_stride / packing;
                    dst_ptr[(row+l*2)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].x);
                    dst_ptr[(row+l*2+1)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].y);
                }
            }
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2, RT, GL, COORD>(dst, src, idx);
}

}
