/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {
// These probably need to be redone to reduce bank conflicts.
// They currently work fine with xor layout but it should be
// possible to reduce their bank conflicts with other layouts too.

/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */

#ifdef KITTENS_CDNA4
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    static_assert(sizeof(U) == 2, "only supporting 16-bit dtypes");

    const int laneid = kittens::laneid();

    const int subtile_stride = kittens::TILE_ROW_DIM<U> * kittens::TILE_COL_DIM<U> * sizeof(U) / 2;
    const int tile_stride = subtile_stride * 2;
    const int row_stride = tile_stride * ST::underlying_width;

    uint32_t addr;

    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        const int elem_per_thread = 16 / sizeof(U); // 8 
        addr = reinterpret_cast<uintptr_t>(&src.data[laneid * elem_per_thread]);
    }
    else {
        const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
        const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
        const int subtile_offset = row_offset * kittens::TILE_ROW_DIM<U> + col_offset;
        addr = reinterpret_cast<uintptr_t>(&src.data[subtile_offset]);
    }

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {

       #pragma unroll
       for(int j = 0; j < dst.width; j++) {

           #pragma unroll 
           for (int k = 0; k < 2; k++) {

                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    asm volatile(
                        "ds_read_b128 %0, %1 offset:%2\n"
                        : "=v"(*reinterpret_cast<float4*>(&dst.tiles[i][j].data[k*4]))
                        : "v"(addr), "i"(i * row_stride + j * tile_stride + k * subtile_stride)
                        : "memory"
                    );
                } else {
                    asm volatile(
                        "ds_read_b64_tr_b16 %0, %2 offset:%3\n"
                        "ds_read_b64_tr_b16 %1, %2 offset:%4\n"
                        : "=v"(*reinterpret_cast<float2*>(&dst.tiles[i][j].data[k*4])), 
                        "=v"(*reinterpret_cast<float2*>(&dst.tiles[i][j].data[k*4 + 2]))
                        : "v"(addr),
                        "i"(i * row_stride + j * tile_stride + k * subtile_stride),
                        "i"(i * row_stride + j * tile_stride + k * subtile_stride + (4 * kittens::TILE_ROW_DIM<U> * sizeof(U)))
                        : "memory"
                    );  
                }
            }
        }
    }
}
#else
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    const int laneid = kittens::laneid();
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        row_offset = laneid%16;
        col_offset = 4*(laneid/16);
    }
    else {
        row_offset = 4*(laneid/16);
        col_offset = laneid%16;
    }
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        const int row = i*dst.tile_size_row + row_offset;
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            const int col = j*dst.tile_size_col + col_offset;
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout

                if constexpr (sizeof(typename ST::dtype) == 4) {
                    // handle float32
                    float2 loaded0 = load_shared_vec(src.idx(src_ptr, {row, col}));
                    float2 loaded1 = load_shared_vec(src.idx(src_ptr, {row, col+2}));
                    dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(loaded0);
                    dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(loaded1);
                } else {
                    // handle fp16 and bf16
                    float2 loaded = load_shared_vec(src.idx(src_ptr, {row, col}));
                    U2* tmp = reinterpret_cast<U2*>(&loaded);
                    dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                    dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                }
            }
            else { // handle the column-major layout
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(U2{src[{row, col}], src[{row+1, col}]});
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(U2{src[{row+2, col}], src[{row+3, col}]});
            }
        }
    }
}
#endif


/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
#ifdef KITTENS_CDNA4
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void store(ST &dst, const RT &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    static_assert(sizeof(U) == 2, "only supporting 16-bit dtypes");

    const int laneid = kittens::laneid();

    const int subtile_stride = kittens::TILE_ROW_DIM<U> * kittens::TILE_COL_DIM<U> / 2;
    const int tile_stride = subtile_stride * 2;
    const int row_stride = tile_stride * dst.width;

    uint32_t addr;

    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        const int elem_per_thread = 16 / sizeof(U); // 8 
        addr = reinterpret_cast<uintptr_t>(&dst.data[laneid * elem_per_thread]);
    }
    else {
        const int row_offset = 8*(laneid/32);
        const int col_offset = laneid%32;
        addr = row_offset * kittens::TILE_ROW_DIM<U> + col_offset;
    }

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {

       #pragma unroll
       for(int j = 0; j < dst.width; j++) {

           #pragma unroll 
           for (int k = 0; k < 2; k++) {

                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    int32x4_t data = *(int32x4_t*)(&src.tiles[i][j].data[k*4]);
                    asm volatile(
                        "ds_write_b128 %0, %1 offset:%2\n"
                        :
                        : "v"(addr), "v"(data), "i"((i * row_stride + j * tile_stride + k * subtile_stride) * sizeof(U))
                        : "memory"
                    );
                } else {
                    U2 tmp[4];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k*4]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k*4 + 1]);
                    tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k*4 + 2]);
                    tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k*4 + 3]);

                    U* dst_ptr = &dst.data[i * row_stride + j * tile_stride + k * subtile_stride + addr];

                    dst_ptr[0] = tmp[0].x;
                    dst_ptr[kittens::TILE_ROW_DIM<U>] = tmp[0].y;
                    dst_ptr[kittens::TILE_ROW_DIM<U> * 2] = tmp[1].x;
                    dst_ptr[kittens::TILE_ROW_DIM<U> * 3] = tmp[1].y;
                    dst_ptr[kittens::TILE_ROW_DIM<U> * 4] = tmp[2].x;
                    dst_ptr[kittens::TILE_ROW_DIM<U> * 5] = tmp[2].y;
                    dst_ptr[kittens::TILE_ROW_DIM<U> * 6] = tmp[3].x;
                    dst_ptr[kittens::TILE_ROW_DIM<U> * 7] = tmp[3].y;
                }
            }
        }
    }
}
#else
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void store(ST &dst, const RT &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    int laneid = kittens::laneid();
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);

    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        row_offset = laneid%16;
        col_offset = 4*(laneid/16);
    }
    else {
        row_offset = 4*(laneid/16);
        col_offset = laneid%16;
    }
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = i*src.tile_size_row + row_offset;
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + col_offset;

            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout
                // *(U2*)(&dst[{row, col+0}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                // *(U2*)(&dst[{row, col+2}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                if constexpr (sizeof(typename ST::dtype) == 4) {
                    // handle float32
                    store_shared_vec(dst.idx(dst_ptr, {row, col}), base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]));
                    store_shared_vec(dst.idx(dst_ptr, {row, col+2}), base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]));
                } else {
                    // handle fp16 and bf16
                    float2 loaded = *reinterpret_cast<const float2*>(src.tiles[i][j].data);
                    store_shared_vec(dst.idx(dst_ptr, {row, col}), loaded);
                }
            }
            else { // handle the column-major layout
                U2 tmp[2];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            
                dst[{row+0, col}] = std::bit_cast<U>(tmp[0].x);
                dst[{row+1, col}] = std::bit_cast<U>(tmp[0].y);
                dst[{row+2, col}] = std::bit_cast<U>(tmp[1].x);
                dst[{row+3, col}] = std::bit_cast<U>(tmp[1].y);
            }            
        }
    }
}
#endif
}