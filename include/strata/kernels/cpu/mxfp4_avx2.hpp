// include/strata/kernels/cpu/mxfp4_avx2.hpp - MXFP4 expert rows in 256-bit lanes, several tokens at once, for CPUs
// without AVX-512 (Zen 2/3, Intel 12th-14th gen).
//
// ggml-cpu's MXFP4 dot product (ggml_vec_dot_mxfp4_q8_0) re-decodes a row's blocks for every token of a window;
// these decode a 32-value block once and apply it to each token: the same integer products and the same exact
// integer sum per four values, times the block's E8M0 scale (2^(e - 128)) and the Q8_0 block's scale, summed in
// ggml's hsum_float_8 order. ggml-cpu's dot is its scalar reference here, which sums a whole block in integers
// before the multiply, so these are equal to it to a relative tolerance, not bit for bit (mxfp4_avx512_parity).
// Activations are Q8_0.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Whether rows of `n` values fit the kernels (whole 32-value blocks, at most 128 of them).
bool mxfp4_256_fits(int n);

/// Gate/up rows [r0, r1) for `nt` tokens: ff[t][r] = silu(g) * u, as mxfp4_512_gu_rows.
void mxfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1);

/// Plain rows [r0, r1) against `nt` Q8_0 activations of `n` values, as mxfp4_512_rows.
void mxfp4_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1);

}  // namespace strata::kernels::cpu
