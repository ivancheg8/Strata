// src/kernels/cpu/mxfp4_avx512.cpp - see include/strata/kernels/cpu/mxfp4_avx512.hpp.
//
// Per pair of 32-value blocks (64 values), once: the 64 E2M1 codes in element order, their magnitudes from
// |kvalues_fp4| (0..12, the doubled E2M1 values), a sign mask (code bit 3), and the two blocks' E8M0 scales. Per
// token: the sign moves onto the activation (a masked subtract - Q8_0 is quantized to [-127, 127], so negation never
// overflows), one vpdpbusd gives 16 int32 lanes = four per 16 values, and one FMA applies the block's E8M0 scale
// times the Q8_0 block's scale. ggml-cpu's scalar dot sums a whole block in integers before that multiply; these
// sum four values per lane, so the integer part is exact either way and only the float order differs.
//
// Needs AVX512F/BW/VL and VNNI - what cpu_avx512_ok() checks before anything here is called.
#include "strata/kernels/cpu/mxfp4_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

// A compile-time table, not a constructor: this file is compiled for a wide ISA, and a namespace-scope object with a
// constructor runs at program start on EVERY CPU - the compiler vectorises the loop with AVX/FMA, and a CPU without them
// (Westmere, Sandy Bridge, ...) died with an illegal instruction before main (#2324).  Powers of two: exact.
constexpr float pow2i(int e) {
    float r = 1.0f;
    for (; e > 0; --e) r *= 2.0f;
    for (; e < 0; ++e) r *= 0.5f;
    return r;
}
struct E8m0Half {   // GGML_E8M0_TO_FP32_HALF (ggml-impl.h): 2^(e - 128), halved for the doubled kvalues_fp4.
    float v[256];   // 0xFF is ggml's infinity; e = 0 and 1 land on the two subnormals, exact by repeated halving
    constexpr E8m0Half() : v() {
        for (int e = 0; e < 256; ++e) v[e] = e == 0xFF ? INFINITY : pow2i(e - 128);
    }
};
constexpr E8m0Half kE8m0;

// software prefetch distance in bytes (STRATA_MXFP4_PREFETCH, 0 = off): the rows stream from DRAM through 4 KB
// pages, where the hardware prefetchers stop at every page boundary; the NVFP4 rows' E-2 knob, same default
// (read on first use, behind the CPU check: a static initialiser here would run on every CPU, see above)
inline int prefetch_distance() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_MXFP4_PREFETCH");
        return e ? std::atoi(e) : 2048;
    }();
    return v;
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// The two blocks' 64 codes in element order. MXFP4 keeps value j in the LOW nibble of qs[j] and value j + 16 in its
// HIGH one (dequant.hpp's dequantize_mxfp4, ggml's own dequantize_row_mxfp4), so a block is two nibble planes - the
// low nibbles of qs[0..15] are its values 0-15, the high nibbles its values 16-31 - and one permute per block
// gathers them.  NVFP4's codes64 interleaves byte by byte instead, because NVFP4 packs values 2j and 2j + 1 in one
// byte; copying that here would permute the weights and leave the dot product wrong by a shuffle, not by rounding.
inline __m512i codes64(const uint8_t* qa, const uint8_t* qb) {
    const __m256i q = _mm256_set_m128i(_mm_loadu_si128((const __m128i*) qb), _mm_loadu_si128((const __m128i*) qa));
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i lo = _mm256_and_si256(q, m4);                         // lane 0 = A 0-15, lane 1 = B 0-15
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);  // lane 0 = A 16-31, lane 1 = B 16-31
    const __m256i a = _mm256_permute2x128_si256(lo, hi, 0x20);         // block A, in element order
    const __m256i b = _mm256_permute2x128_si256(lo, hi, 0x31);         // block B, in element order
    return _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);
}

// lanes 0-7 = 2^(qa's exponent - 128), lanes 8-15 = block B's: the vpdpbusd's lanes 0-7 are block A's 32 values
inline __m512 block_scales(uint8_t ea, uint8_t eb) {
    const __m128 s2 = _mm_setr_ps(kE8m0.v[ea], kE8m0.v[eb], 0.0f, 0.0f);   // lanes 2-3 unused: idx selects 0 and 1
    const __m512i idx = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    return _mm512_permutexvar_ps(idx, _mm512_castps128_ps512(s2));
}

// Per token and block pair, the activation's scales as lanes: 0-7 -> the first Q8_0 block's d (the weight block's
// 32 values), 8-15 -> the second's. Built once per call, reused by every row.
constexpr int kMaxPairs = 64;                                        // rows of up to 4096 values (n_embd 2560)
struct Acts {
    int nt = 0, nb = 0;                                              // nb = pairs of 32-value blocks
    const block_q8_0* y[7] = {};
    __m512 dy[7 * kMaxPairs];                                        // [t * nb + ip]; on the stack, no allocation
    Acts(const void* const* act, int nt_, int nb_) : nt(nt_), nb(nb_) {
        for (int t = 0; t < nt; ++t) {
            y[t] = (const block_q8_0*) act[t];
            for (int ip = 0; ip < nb; ++ip) {
                const block_q8_0* yb = y[t] + 2 * ip;
                dy[(size_t) t * nb + ip] = _mm512_mask_blend_ps((__mmask16) 0xFF00, _mm512_set1_ps(h2f(yb[0].d)),
                                                                 _mm512_set1_ps(h2f(yb[1].d)));
            }
        }
    }
};

template <int NT>
inline void row_dot(const uint8_t* row, const Acts& a, float* res) {
    const __m512i mag = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    const __m512i eight = _mm512_set1_epi8(8);
    const __m512i zero = _mm512_setzero_si512();
    __m512 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
    const block_mxfp4* x = (const block_mxfp4*) row;
    const int prefetch_ahead = prefetch_distance();
    for (int ip = 0; ip < a.nb; ++ip) {
        const block_mxfp4* xb = x + 2 * ip;
        if (prefetch_ahead > 0) _mm_prefetch((const char*) xb + prefetch_ahead, _MM_HINT_T0);
        const __m512i codes = codes64(xb[0].qs, xb[1].qs);
        const __m512i g = _mm512_shuffle_epi8(mag, codes);
        const __mmask64 neg = _mm512_test_epi8_mask(codes, eight);
        const __m512 sw = block_scales(xb[0].e, xb[1].e);
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* yb = a.y[t] + 2 * ip;
            const __m512i yv = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) yb[0].qs)),
                                                  _mm256_loadu_si256((const __m256i*) yb[1].qs), 1);
            const __m512i ys = _mm512_mask_sub_epi8(yv, neg, zero, yv);
            const __m512i p = _mm512_dpbusd_epi32(zero, g, ys);
            acc[t] = _mm512_fmadd_ps(_mm512_mul_ps(sw, a.dy[(size_t) t * a.nb + ip]), _mm512_cvtepi32_ps(p), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(acc[t]);
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, const Acts& a, float* const* ff, int r0, int r1) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(blob + (size_t) r * gu_row, a, g);
        row_dot<NT>(blob + up_off + (size_t) r * gu_row, a, u);
        for (int t = 0; t < NT; ++t) {   // native_gu_rows' ggml path with both global scales at 1.0 (MXFP4 has none)
            const float gs = g[t];
            ff[t][r] = (gs / (1.f + std::exp(-gs))) * u[t];
        }
    }
}

template <int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, const Acts& a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(w + (size_t) r * row_bytes, a, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

}  // namespace

bool mxfp4_512_fits(int n) { return n % (2 * QK_MXFP4) == 0 && n / (2 * QK_MXFP4) <= kMaxPairs; }

// Windows wider than 7 run in balanced slices of at most 7: at 8 the accumulators no longer fit beside the loads and
// MSVC spills (the NVFP4 rows' measurement: 0.32 ms for 8 tokens against 2 x 0.13 for 4 + 4).
template <typename F>
inline void slices(int nt, F&& run) {
    const int k = (nt + 6) / 7;
    for (int i = 0, t0 = 0; i < k; ++i) {
        const int w = nt / k + (i < nt % k ? 1 : 0);
        run(t0, w);
        t0 += w;
    }
}

void mxfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1) {
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / (2 * QK_MXFP4));
        float* const* f = ff + t0;
        switch (w) {
            case 1: gu_rows<1>(blob, gu_row, up_off, a, f, r0, r1); break;
            case 2: gu_rows<2>(blob, gu_row, up_off, a, f, r0, r1); break;
            case 3: gu_rows<3>(blob, gu_row, up_off, a, f, r0, r1); break;
            case 4: gu_rows<4>(blob, gu_row, up_off, a, f, r0, r1); break;
            case 5: gu_rows<5>(blob, gu_row, up_off, a, f, r0, r1); break;
            case 6: gu_rows<6>(blob, gu_row, up_off, a, f, r0, r1); break;
            default: gu_rows<7>(blob, gu_row, up_off, a, f, r0, r1); break;
        }
    });
}

void mxfp4_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1) {
    slices(nt, [&](int t0, int k) {
        const Acts a(act + t0, k, n / (2 * QK_MXFP4));
        float* const* o = out + t0;
        switch (k) {
            case 1: dot_rows<1>(w, row_bytes, a, o, r0, r1); break;
            case 2: dot_rows<2>(w, row_bytes, a, o, r0, r1); break;
            case 3: dot_rows<3>(w, row_bytes, a, o, r0, r1); break;
            case 4: dot_rows<4>(w, row_bytes, a, o, r0, r1); break;
            case 5: dot_rows<5>(w, row_bytes, a, o, r0, r1); break;
            case 6: dot_rows<6>(w, row_bytes, a, o, r0, r1); break;
            default: dot_rows<7>(w, row_bytes, a, o, r0, r1); break;
        }
    });
}

}  // namespace strata::kernels::cpu
