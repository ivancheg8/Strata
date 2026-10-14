// src/prefill/mmq_mxfp4_w4a8.cu - MXFP4 MMQ with 8-bit activations (W4A8), the NVFP4 unit's trick for the other FP4.
//
// mmq.cuh takes a 4-bit FP weight to Blackwell's FP4 tensor cores when BLACKWELL_MMA_AVAILABLE.  For MXFP4 that is
// mma with .kind::mxf4, which ptxas accepts only in the sm_100a / 101a / 103a target images: on sm_120 (the RTX 50
// cards, this project's GPU) ggml's own mmq-instance-mxfp4.cu fails to assemble.  The int8 path - weights decoded to
// int8, activations Q8_1, int8 tensor cores - is what decode runs at (vec_dot_mxfp4_q8_1, the same doubled e2m1
// codes as kvalues_mxfp4), so the prompt path takes it too: this TU compiles mmq.cuh's int8 MXFP4 for every
// architecture by hiding the FP4 MMA on both sides - the device code (BLACKWELL_MMA_AVAILABLE) and the host's tile
// config (blackwell_mma_available(cc)) - they must agree, the host sizes shared memory for the device's layout.
// The two non-static templates are renamed so the linker cannot fold them with ggml's instance or with the NVFP4 unit.
//
// MXFP4 needs no per-expert FP32 scale: the block's E8M0 byte is the whole scale (NVFP4's weight_scale_2 tail is a
// per-expert FP32 and the MMQ tail machinery is only for it), so nothing here multiplies the product afterwards.
// Built only with -DSTRATA_MMQ_MXFP4=ON: mmq.cuh is not vendored, so its int8 MXFP4 path is verified by the build.
#include "common.cuh"

#undef BLACKWELL_MMA_AVAILABLE
#define blackwell_mma_available(cc) false
#define mul_mat_q_switch_J strata_mxfp4_w4a8_mul_mat_q_switch_J
#define mul_mat_q_case strata_mxfp4_w4a8_mul_mat_q_case
#include "mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_MXFP4);

namespace strata::prefill::mmq {
void run_mxfp4_w4a8(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    strata_mxfp4_w4a8_mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, a, s);
}
}  // namespace strata::prefill::mmq
