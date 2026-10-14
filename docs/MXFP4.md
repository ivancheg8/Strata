# MXFP4 routed experts (AtomicChat's IQ4_XS build)

The engine can serve AtomicChat's build of Qwen3.8-Flash-Next, **Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64**: its
routed experts' down matrices are MXFP4 (ggml type 39) and its gate and up matrices are IQ2_S, quantized with
AtomicChat's own imatrix. Everything else (attention, GDN, shared experts, routers, the head) stays in the formats
the native path already reads.

MXFP4 is the OCP microscaling format: 32-value blocks of E2M1 codes with **one E8M0 scale byte per block** - 17 bytes
for 32 values. ggml stores it as `block_mxfp4 { uint8_t e; uint8_t qs[16]; }`, and its 16-entry table
`kvalues_fp4` (`ggml-common.h`) holds the doubled E2M1 values, `0,1,2,3,4,6,8,12` and their negatives, so a weight is
`2^(e - 128) * kvalues_fp4[code]`. Value *j* is in the low nibble of `qs[j]` and value *j + 16* in its high nibble.

**MXFP4 has no per-expert global scale, so an MXFP4 expert blob has no tail.** The E8M0 byte inside every block is
the whole scale a block has: `tools/iq_pack.py` writes its blobs as `[gate | up | down]` with nothing after them,
and the kernels take no scale arguments.

Numbers on this page are marked with what they were measured on. No speed number here is measured on the current
build - see [What is not measured here](#what-is-not-measured-here).

## Getting a model

```
START-HERE.bat --setup --family atomic --model IQ4_XS
```

The family is AtomicChat's own folder of [AtomicChat/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF),
28 shard files named `Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64-00001-of-00028.gguf`: 84.9 GB of downloads, 41.05 GB
of experts (MXFP4 downs, IQ2_S gate and up), 45.8 GB of it resident, 48 GB of RAM. Those numbers are read from the
repository's 28 shard files and their tensor directories, not measured on a PC.

The file's dense side (`hc_attn_*`, `hc_ffn_*`, `output_hc_*`) is Q8_0, and the engine's dense residents are BF16, so
setup packs it with `tools/iq_pack.py --compat-bf16` (the family's `pack_args`); without it the packer's output is
refused. The 38.4 GB n-gram table is alone in shard 2 and is read from the SSD, as for every other model.

## Where MXFP4 runs

| Path | What runs | Where |
| --- | --- | --- |
| Decode, experts on the GPU | MXFP4 down, IQ2_S gate/up: `X(39)` in `STRATA_GU_FMTS`, `STRATA_D_FMTS`, `STRATA_MMVQ_FMTS` | `src/kernels/cuda/iq_kernels.cu`, device code in `src/kernels/cuda/mxfp4_kernels.cuh` |
| Decode, experts on the CPU | `mxfp4_512_*` (AVX-512) / `mxfp4_256_*` (AVX2), else ggml-cpu's dot | `src/kernels/cpu/mxfp4_avx512.cpp`, `src/kernels/cpu/mxfp4_avx2.cpp`, wired in `src/kernels/cpu/native_expert.cpp` |
| Prompt, experts on the GPU | fused int8 kernels: IQ2_S gate/up with MXFP4 down (`d_covered`) | `src/prefill/moe_fused_iq.cu`, with `STRATA_PF_FUSED=1` |
| Prompt, experts on the GPU | ggml's MMQ, int8 path | `src/prefill/mmq_mxfp4_w4a8.cu`, dispatched by `src/prefill/moe_mmq.cu` |
| Prompt, dequantize fallback | dequantize to FP16, then cuBLAS | `src/kernels/cuda/dequant_bf16.cu` (`group32<39>`), driven by `src/prefill/prefill.cpp` |
| The n-gram table | MXFP4 rows: 85 bytes for 160 values | `PLE_ROW_BYTES_MXFP4` in `include/strata/kernels/ngram.hpp`, `PleFormat::MXFP4` |

MXFP4 is an expert format only: the parity tool `src/kernels/iq_parity.cpp` runs it through the native MMVQ kernels,
not the i-quant MMVQ ones, because it has no codebooks.

`native_expert_supported` asks for a 32-value alignment for MXFP4 (the i-quants ask for 256): a 17-byte block puts its
16 weight bytes at an odd address in every other block, so the kernels read them as bytes.

## The MMQ unit

ggml's own MXFP4 MMQ instance (`mmq-instance-mxfp4.cu`) emits Blackwell's FP4 block-scale MMA, and ptxas takes
`.kind::mxf4` only in the `sm_100a / 101a / 103a` images: building that instance for `sm_120` fails. Strata's MXFP4
prompt kernels never use it - the fused path decodes MXFP4 to int8 codes and runs on `mma.sync m16n8k32 s8` (sm_80
and up), the MMVQ path on DP4A.

So the MXFP4 MMQ path is compiled in its own unit, `src/prefill/mmq_mxfp4_w4a8.cu`: mmq.cuh's int8 path with the FP4
MMA hidden from the device code and from the host's tile config, so it is the same on every architecture.
`moe_mmq.cu` takes it for type 39 with no mode switch (there is no `w4a4` for MXFP4); the unit is CUDA-only, behind
`!defined(GGML_USE_HIP)`.

## CPU experts

`src/kernels/cpu/mxfp4_avx512.cpp` takes a pair of 32-value blocks at a time - 64 nibbles in one 512-bit register -
and applies them to every token of a verify window with one `vpdpbusd`: 16 int32 lanes, four values each, then one
FMA per lane group with the block's E8M0 scale times the Q8_0 block's scale. The sign is moved onto the activation
(a masked subtract; Q8_0 is quantized to [-127, 127], so negation never overflows). It needs AVX512F/BW/VL and VNNI,
which `cpu_avx512_ok()` checks before anything here is called, and covers rows of up to 4,096 values in windows of
up to 7 tokens.

`src/kernels/cpu/mxfp4_avx2.cpp` is the same for CPUs with AVX2 and no AVX-512 (Zen 2/3, Intel 12th-14th gen): one
32-value block at a time, `_mm256_maddubs_epi16` and `_mm256_madd_epi16`, one FMA per block, up to 128 blocks and
8 tokens.

The arithmetic is ggml-cpu's `ggml_vec_dot_mxfp4_q8_0`. ggml sums a whole 32-value block in integers before the
scale multiply, these sum four values per lane, so the integer part is exact either way and only the order of the
float additions differs - `mxfp4_avx512_parity` checks a relative tolerance, not bit equality. Activations are Q8_0,
MXFP4's `vec_dot_type` in ggml-cpu.

| Knob | Effect |
| --- | --- |
| `STRATA_NO_MXFP4_512=1` | the AVX-512 rows off; the pool uses the AVX2 rows, or ggml-cpu |
| `STRATA_NO_MXFP4_256=1` | the AVX2 rows off; the pool uses ggml-cpu |
| `STRATA_MXFP4_PREFETCH=<bytes>` | software prefetch distance in the rows (default 2048, 0 = off) |

No speed numbers for these rows have been measured. `mxfp4_avx512_parity --bw THREADS TOKENS` prints the GB/s of
weights read and the ms per expert for ggml-cpu, AVX2 and AVX-512 side by side.

## Prompt geometry for a 2880-wide model

AtomicChat's file is 2880 wide (`n_embd`), against the 2560 the Coder and the GSQ-RCO files use. The prefill's
dequantize scratch halves are sized from the geometry (`2 * n_ff * n_embd` and `n_embd * n_ff` in
`Prefill::carve` and its mirror `bytes_needed_impl`); a fixed `1280 * 2560` and `2560 * 640` is 409,600 uint16
short for a 2880-wide model and the same number as the geometry-sized form for every 2560-wide model.

The CPU rows' shapes cover it: `mxfp4_512_fits(2880)` is 45 pairs of blocks (of 64), `mxfp4_256_fits(2880)` is 90
blocks (of 128).

## Other GPUs

The MXFP4 device code lives in `src/kernels/cuda/mxfp4_kernels.cuh`, included into `iq_kernels.cu`: the i-quant
dispatch (`Fmt<39>`, `Split<39>`, `dq_mxfp4`) is instantiated in that translation unit, so a separate `.cu` for
MXFP4 is not possible without relocatable device code. That TU carries the `__HIPCC__` guards the other IQ formats
use, so the MXFP4 decode kernels build for HIP; they use byte loads, `__ldcs`/`__ldg` and integer arithmetic, no
CUDA-only intrinsics. **Neither the CUDA nor the HIP build of the MXFP4 kernels has been run here.**

The MMQ unit and the fused prefill unit are CUDA-shaped; `moe_mmq.cu` keeps the MXFP4 case behind
`!defined(GGML_USE_HIP)`. On gfx12 the fused path stays off by default, as it is for every format.

## Tests

- `mxfp4_avx512_parity` (CTest `mxfp4_cpu_parity`): the AVX-512 and AVX2 rows against `ggml_vec_dot_mxfp4_q8_0` on
  random blocks that cover every code and exponents 120..135 (2^-8 .. 2^7), every row x every token of 1..9.
  - `mxfp4_avx512_parity --layers pack/experts.bin [expert]`: one expert's real down matrix (default 7) of every layer.
  - `mxfp4_avx512_parity --expert pack/experts.bin [layer expert]`: one whole AtomicChat expert as the CPU pool runs
    it - IQ2_S gate/up through ggml-cpu's dot, the hidden through Q8_0, the MXFP4 down through these kernels -
    against a double-precision reference.
  - `mxfp4_avx512_parity --bw THREADS TOKENS [ggml|avx2]`: DRAM rate over 1,536 blobs (3.8 GB, 40x the L3).
  - The gate/up rows are checked through `silu(g) * u` with no tail, which is what MXFP4's lack of a global scale
    means for the CPU pool.
- `prefill_fused_iq_test` (`tests/cuda/prefill_fused_iq_test.cpp`) covers the fused formats; it has no MXFP4 case.
- `iq_parity` and `native_grouped_parity` take type 39 through the native paths; `tools/iq_fixture.py` writes MXFP4
  blocks for them (exponents 124..130 plus the two subnormal exponents).

## What is not measured here

No number on this page is measured on the current build. The ones below were measured on this PC - an RTX 5070
(12 GB), a Ryzen 7 5700X and 64 GB of RAM, PCIe 4.0, 204,800 context - on an earlier build of the same MXFP4 work.

| | atomic-iq4_xs | swift-1.5-iq3_xxs, same PC |
| --- | ---: | ---: |
| Prefill | 1,390 tokens/s | 2,336 tokens/s |
| Decode | 50 tokens/s | 62 tokens/s |
| First token at 181K context | 133 s | 78 s |
| Expert cache hit | ~77% | ~80% |

The same run's diagnostic lines, which is what the prefill slowdown reads: the fused path is active for AtomicChat's
experts (IQ2_S gate/up with MXFP4 down is covered), so the gap is expert streaming and the wider model, not a
fallback to dequantize + cuBLAS:

```
strata: fused experts (native): 48 SMs, 1 blocks a SM (the min over the formats); MXFP4 down ww=2: 44 KiB shared, 126 registers a thread, 1 blocks a SM
strata generate: expert arena read unbuffered ... loaded 38.23 GiB at 0.50 GiB/s
```

126 registers a thread is the occupancy limiter (two 512-thread blocks would need 64 or fewer), not the 44 KiB of
shared memory. The 0.50 GiB/s is the measured read rate of the expert arena, not the PCIe limit: PCIe 4.0 x16
carries ~25 GiB/s.

To measure the current build, on the same PC:

1. `mxfp4_avx512_parity` with no arguments, then `--layers`, `--expert` and `--bw 16 1` / `--bw 16 4` on the pack.
2. `prefill_fused_iq_test` and `iq_parity` / `native_grouped_parity` with a MXFP4 fixture.
3. Decode and prefill rates of `--family atomic` against the same PC's IQ3_XXS run, with `STRATA_PF_INFO=1`, and the
   same with `STRATA_PF_FUSED_NATIVE=0` (MMQ) to compare the two prompt paths.
4. Quality: `tools/quant_quality.py` against Q8_0 - no KL or perplexity number for this pack exists.
5. The CPU rows: `mxfp4_avx512_parity --bw`, and decode with `STRATA_NO_MXFP4_512=1` /
   `STRATA_NO_MXFP4_256=1` on a PC whose experts run on the CPU.

## See also

- [MODELS.md](MODELS.md#atomicchat-ad-384bpw-iq4_xs-m64) - where this build fits by RAM.
- [QUANTS.md](QUANTS.md) - the quality yardstick for the formats the other packs use.
