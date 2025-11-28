/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Load data into a register vector from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void load(RV &dst, const GL &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(!std::is_same_v<T, fp8e4m3> && !std::is_same_v<U, fp8e4m3>, "Unsupported type for load");

    U *src_ptr = (U*)&src[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();

    // TODO: this uses no inter-thread communication and is therefore not optimal.
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int idx = w*RV::reductions + RV::stride*(laneid/RV::aligned_threads);
            // this should be a maximally coalesced load.
            #pragma unroll
            for(int i = 0; i < RV::strides_per_tile; i++) {
                #pragma unroll
                for(int j = 0; j < RV::packed_per_stride; j++) {
                    dst[w][i * RV::packed_per_stride + j] = 
                        base_types::convertor<T2, U2>::convert(*(U2*)&src_ptr[idx + i * RV::elements_per_stride_group + j * RV::packing]);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < RV::outer_dim; w++) {
            int idx = w * RV::reductions + (laneid % RV::reductions);
            // this should be a maximally coalesced load.
            dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[idx]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        const int offset = laneid * RV::inner_dim;
        if (offset >= RV::length) return;
        #pragma unroll
        for (int i = 0; i < RV::inner_dim; i++) {
            dst[0][i] = base_types::convertor<T, U>::convert(src_ptr[offset + i]);
        }
    }
}

/**
 * @brief Store data from a register vector to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store(const GL &dst, const RV &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(!std::is_same_v<T, fp8e4m3> && !std::is_same_v<U, fp8e4m3>, "Unsupported type for store");

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < RV::outer_dim; w++) {
            int idx = w*RV::reductions + RV::stride*(laneid/RV::aligned_threads);
            // this should be a maximally coalesced store. I hope!
            #pragma unroll
            for (int i = 0; i < RV::strides_per_tile; i++) {
                #pragma unroll
                for (int j = 0; j < RV::packed_per_stride; j++) {
                    *(U2*)&dst_ptr[idx + i * RV::elements_per_stride_group + j * RV::packing] = base_types::convertor<U2, T2>::convert(src[w][i * RV::packed_per_stride + j]);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            int idx = w * RV::reductions + (laneid % RV::reductions);
            // this should be a maximally coalesced load.
            dst_ptr[idx] = base_types::convertor<U, T>::convert(src[w][0]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        const int offset = laneid * src.inner_dim;
        if (offset >= src.length) return;
        #pragma unroll
        for (int i = 0; i < RV::inner_dim; i++) {
            dst_ptr[offset + i] = base_types::convertor<U, T>::convert(src[0][i]);
        }
    }
}

} // namespace kittens
