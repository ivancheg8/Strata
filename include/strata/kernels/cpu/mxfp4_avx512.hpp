// include/strata/kernels/cpu/mxfp4_avx512.hpp - MXFP4 expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's MXFP4 dot product (ggml_vec_dot_mxfp4_q8_0) is single-token: every token of a verify window re-decodes
// the same weights. These decode a pair of 32-value blocks at a time - 64 nibbles in one 512-bit register, one
// vpdpbusd - and apply them to every token. The arithmetic is ggml's: an exact integer sum per four values, times
// the block's E8M0 scale (2^(e - 128), halved for the doubled kvalues_fp4) and the Q8_0 block's scale; only the
// order of the float additions differs from ggml-cpu's, so mxfp4_avx512_parity checks a relative tolerance.
//
// MXFP4 has no per-expert global scale - the E8M0 byte is the whole scale a block has - so, unlike the NVFP4 rows,
// these take no scales and read no blob tail. Activations are Q8_0, MXFP4's vec_dot_type in ggml-cpu.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Whether rows of `n` values fit the kernels (whole 32-value blocks, taken in pairs, at most 64 pairs).
bool mxfp4_512_fits(int n);

/// Gate/up rows [r0, r1) of an MXFP4 expert blob for `nt` tokens: ff[t][r] = silu(g) * u, where g and u are the
/// MXFP4 dot products of gate row r and up row r (at `up_off`) with activation t (Q8_0, `n` values).
void mxfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1);

/// Plain rows [r0, r1) of an MXFP4 matrix (`row_bytes` apart) against `nt` Q8_0 activations of `n` values.
void mxfp4_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1);

}  // namespace strata::kernels::cpu
