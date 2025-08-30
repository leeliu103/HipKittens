#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 1; // batch size
constexpr int ATTN_H = 1; // number of heads
constexpr int ATTN_N = 1024; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int BLOCK_SIZE_QO = 16; // block size for QO
constexpr int BLOCK_SIZE_KV = 256; // block size for KV
constexpr int WARP_SIZE_QO = 16; // warp size for QO
constexpr int WARP_SIZE_KV = 64; // warp size for KV

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;

template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using qo_tile = rt<T, WARP_SIZE_QO, D, L, M>;
template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using kv_tile = rt<T, WARP_SIZE_KV, D, L, M>;
template<int D, typename T=float, typename L=accum_col_l, typename M=mfma_16x16x32> using attn_tile = rt<T, WARP_SIZE_QO, WARP_SIZE_KV, L, M>;


template<int D> struct attn_prep_globals { 
    gl<bf16, -1, -1, -1, -1> Og;
    gl<float, -1, -1, -1, -1> dOg; 
    gl<float, -1, -1, -1, -1> delta;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / (WARP_SIZE_QO * NUM_WARPS)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 32000; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_prep_ker(const attn_prep_globals<D> g) {
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;

    const int warpid = kittens::warpid();

    qo_tile<D, float, row_l> dO, O;
    typename qo_tile<D, float, row_l>::col_vec delta_vec;

    load(dO, g.dOg, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid,0});
    load(O,  g.Og,  {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid,0});
    
    // Δ_i = row_sum(dO ⊙ O) 
    mul(dO, dO, O);
    row_sum(delta_vec, dO); 
    store(g.delta, delta_vec, {batch_idx, head_idx, 0, seq_idx * NUM_WARPS + warpid});
}

template<int D>
void dispatch_prep(attn_prep_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_prep_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_prep_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

template<int D> struct attn_bwd_combined_globals { 
    gl<bf16, -1, -1, -1, -1> Q, K, V, O;
    gl<float, -1, -1, -1, -1> dOg, dQg, dKg, dVg;
    gl<float, -1, -1, -1, -1> m_vec, l_vec, delta_vec;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / BLOCK_SIZE_KV); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 32000; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_ker(const attn_bwd_combined_globals<D> g) {
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;

    const int warpid = kittens::warpid();

    const float scale_factor = 1.0f / sqrt(D);

    // Register tiles
    kv_tile<D, bf16, row_l, mfma_16x16x32> K_j, V_j;
    kv_tile<D, float, accum_col_l, mfma_32x32x16> dK_j, dV_j;

    // 6. Load K_j and V_j from HBM to registers
    load(K_j, g.K, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid, 0});
    load(V_j, g.V, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid, 0});
    
    // 7. Initialize dK_j = 0 and dV_j = 0
    zero(dK_j);
    zero(dV_j);

    // 8. for 1 <= i <= T_r (1024 / 32 = 32)
    for (int i = 0; i < ATTN_N / BLOCK_SIZE_QO; ++i) {

        // 9. Load Q_i, O_i, dO_i, dQ_i, l_i, m_i, delta_i from HBM to registers
        qo_tile<D, bf16, row_l, mfma_16x16x32> Q_i;
        qo_tile<D, bf16, col_l, mfma_32x32x16> dO_i;
        attn_tile<D, float, accum_col_l, mfma_16x16x32>::col_vec l_i, m_i, delta_i;
        load(Q_i, g.Q, {batch_idx, head_idx, i * NUM_WARPS + warpid, 0});
        load(dO_i, g.dOg, {batch_idx, head_idx, i * NUM_WARPS + warpid, 0});
        load(l_i, g.l_vec, {batch_idx, head_idx, 0, i * NUM_WARPS + warpid});
        load(m_i, g.m_vec, {batch_idx, head_idx, 0, i * NUM_WARPS + warpid});
        load(delta_i, g.delta_vec, {batch_idx, head_idx, 0, i * NUM_WARPS + warpid});

        // 10. S_ij = Q_i K_j^T * scale
        attn_tile<D, float, accum_col_l, mfma_16x16x32> S_ij;
        zero(S_ij);
        mma_ABt(S_ij, Q_i, K_j, S_ij);
        mul(S_ij, S_ij, scale_factor);

        // 11. P_ij = exp(S_ij - m_i) / l_i
        sub_row(S_ij, S_ij, m_i);
        exp(S_ij, S_ij);
        div_row(S_ij, S_ij, l_i);

        // 12. dV_j += P_ij^T @ dO_i
        attn_tile<D, bf16, accum_col_l, mfma_16x16x32> P_ij_bf16_acc_col;
        copy(P_ij_bf16_acc_col, S_ij);
        attn_tile<D, bf16, col_l, mfma_32x32x16> P_ij_bf16_col = *(attn_tile<D, bf16, col_l, mfma_32x32x16>*)(&P_ij_bf16_acc_col);
        mma_AtB(dV_j, P_ij_bf16_col, dO_i, dV_j);

        // // 13. dP_ij = dO_i @ V_j^T
        // attn_tile<D,float,accum_col_l> dP_ij;
        // zero(dP_ij);
        // qkvo_tile<D,bf16,row_l> dO_i_bf16_row = swap_layout_inplace<row_l>(dO_i);
        // mma_ABt(dP_ij, dO_i_bf16_row, V_j, dP_ij);

        // // 14. dS_ij = P_ij o (dP_ij - delta_i)
        // sub_row(dP_ij, dP_ij, delta_i);
        // mul(dP_ij, dP_ij, S_ij);
        // mul(dP_ij, dP_ij, scale_factor);

        // // 15. dQ_i += dS_ij @ K_j (load from HBM and write back)
        // qkvo_tile<D, float, accum_col_l> dQ_i;
        // load(dQ_i, g.dQg, {b,h,i,0});
        // attn_tile<D, bf16, accum_col_l> dS_ij_bf16_acc_col;
        // copy(dS_ij_bf16_acc_col, dP_ij);
        // attn_tile<D, bf16, row_l> dS_ij_bf16_row = swap_layout_inplace<row_l>(dS_ij_bf16_acc_col);
        // qkvo_tile<D, bf16, col_l> K_j_col;
        // swap_layout(K_j_col, K_j);
        // mma_AB(dQ_i, dS_ij_bf16_row, K_j_col, dQ_i);
        // store(g.dQg, dQ_i, {b,h,i,0});

        // // 16. dK_j += dS_ij^T @ Q_i
        // attn_tile<D,bf16,col_l> dS_ij_bf16_col = swap_layout_inplace<col_l>(dS_ij_bf16_row);
        // qkvo_tile<D, bf16, col_l> Q_i_col = swap_layout_inplace<col_l>(Q_i);
        // mma_AtB(dK_j, dS_ij_bf16_col, Q_i_col, dK_j);
    }

    // 18. Write dK_j and dV_j back to HBM
    store(g.dKg, dK_j, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid, 0});
    store(g.dVg, dV_j, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid, 0});
}

template<int D>
void dispatch_bwd_combined(attn_bwd_combined_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_combined_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_combined_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";

    py::bind_function<dispatch_prep<ATTN_D>>(m, "dispatch_prep", 
        &attn_prep_globals<ATTN_D>::Og, 
        &attn_prep_globals<ATTN_D>::dOg,
        &attn_prep_globals<ATTN_D>::delta
    );

    py::bind_function<dispatch_bwd_combined<ATTN_D>>(m, "dispatch_bwd_combined", 
        &attn_bwd_combined_globals<ATTN_D>::Q, 
        &attn_bwd_combined_globals<ATTN_D>::K, 
        &attn_bwd_combined_globals<ATTN_D>::V, 
        &attn_bwd_combined_globals<ATTN_D>::O, 
        &attn_bwd_combined_globals<ATTN_D>::dOg, 
        &attn_bwd_combined_globals<ATTN_D>::dQg,
        &attn_bwd_combined_globals<ATTN_D>::dKg,
        &attn_bwd_combined_globals<ATTN_D>::dVg,
        &attn_bwd_combined_globals<ATTN_D>::m_vec, 
        &attn_bwd_combined_globals<ATTN_D>::l_vec,
        &attn_bwd_combined_globals<ATTN_D>::delta_vec
    );
}