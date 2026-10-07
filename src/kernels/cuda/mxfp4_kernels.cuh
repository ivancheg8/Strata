// src/kernels/cuda/mxfp4_kernels.cuh - MXFP4 (ggml type 39): the OCP block format, one E8M0 scale per 32 values
// and E2M1 nibbles under it, which is not one of the i-quant formats.  Everything MXFP4 is here; iq_kernels.cu
// keeps only the dispatch that names 39 (the format lists, dq_dispatch, is_iq, iq_row_bytes,
// native_expert_supported).
//
// Why a header and not a translation unit of its own: this build does not enable CUDA relocatable device code
// (CMakeLists.txt sets no CUDA_SEPARABLE_COMPILATION, and hipcc compiles whole programs by default), so a
// __device__ function defined in another .cu cannot be called from iq_kernels.cu.  Fmt<39> and Split<39> are
// specializations of the templates iq_kernels.cu declares, and iq_kernels.cu instantiates them for TY = 39
// (launch_mmvq<39>, gu_qk / d_qk / gu_split, native_gu_kernel / native_down_kernel), where they must be complete.
// llama.cpp gives vecdotq.cuh and dequantize.cuh the same shape, and this directory gives s26_tsum.cuh one.
//
// So this file opens no namespace: iq_kernels.cu includes it inside `namespace strata::kernels { namespace {`,
// after the cvt specializations - the last thing these need, together with get_int_from_table_16, the
// ggml_cuda_dp4a macro and ggml-common.h's block_mxfp4, kvalues_mxfp4 and QK_MXFP4 - and before the first place
// that instantiates Fmt<39> or Split<39>.
#pragma once

// Read 4 bytes as int32 with byte shift (safe for unaligned 17 byte block memory)
// offset is an index in 4-byte units (like in get_int_b2/get_int_b4); i.e., bytes 4*offset through 4*offset+3 are read.
__device__ __forceinline__ int get_int_b1(const void* x, const int& offset) {
    const uint8_t* x8 = (const uint8_t*) x;
    return (int)x8[4*offset + 0] | ((int)x8[4*offset + 1] << 8) |
           ((int)x8[4*offset + 2] << 16) | ((int)x8[4*offset + 3] << 24);
}

// Conversion of E8M0 to float, divided by 2 (since kvalues_mxfp4 are doubled)
__device__ __forceinline__ float e8m0_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) { bits = 0x00200000u << x; }          // 2^-128, 2^-127
    else       { bits = (uint32_t)(x - 1) << 23; }   // 2^(x-128)
    return __int_as_float(bits);
}

#define VDR_MXFP4_Q8_1_MMVQ 2

static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(
        const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_mxfp4* bq4 = (const block_mxfp4*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int2 v = get_int_from_table_16(get_int_b1(bq4->qs, iqs + l), kvalues_mxfp4);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    return e8m0_half(bq4->e) * __low2float(bq8_1->ds) * sumi;
}

template<> struct Fmt<39> {
    static constexpr int qk = 32, ipb = 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) {
        return vec_dot_mxfp4_q8_1(v, y, kbx, iqs);
    }
};

template<> inline constexpr bool kSplit<39> = true;
template<> struct Split<39> {   // MXFP4
    struct W { int2 v[2]; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_mxfp4* bq4 = (const block_mxfp4*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l) r.v[l] = get_int_from_table_16(get_int_b1(bq4->qs, iqs + l), kvalues_mxfp4);
        r.dw = e8m0_half(bq4->e);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* bq8_1, int iqs) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            sumi = ggml_cuda_dp4a(r.v[l].x, q8[l + 0], sumi);
            sumi = ggml_cuda_dp4a(r.v[l].y, q8[l + 4], sumi);
        }
        return r.dw * __low2float(bq8_1->ds) * sumi;
    }
};

template<typename dst_t>
__device__ void dq_mxfp4(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // ibs — superblock index. In Strata, a superblock = 256 elements = 8 MXFP4 blocks (256 / 32).
    const block_mxfp4* x = (const block_mxfp4*) vx + ibs * (256 / QK_MXFP4);
    const int il = tid / 8; // 0..3
    const int ib = tid % 8; // 0..7
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = e8m0_half(x[ib].e);

    for (int j = 0; j < 4; ++j) {
        y[j + 0]  = cvt<dst_t>(d * kvalues_mxfp4[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_mxfp4[q4[j] >> 4]);
    }
}
