# MXFP4 support in Strata for AtomicChat Qwen3.8-Flash-Next

Technical report on the branch that adds MXFP4 (ggml type 39) to Strata for the AtomicChat
`Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64` model.

- Tree: Strata `b4a488f` "strata v0.1.44.1", branch `custom-0.1.44`.
- Change: 24 tracked files modified (+308 / -63) and 8 new files; `git diff --stat` and `git status --short` are the
  authoritative list.
- Every new line below is quoted from the working tree of this branch.

## 1. Summary

MXFP4 is the OCP microscaling FP4: 32 values per block, each value a 4-bit e2m1 number, the block scaled by one
E8M0 byte. AtomicChat stores its MoE down projections in it, so a Strata build that is to run this model has to
decode, cache and compute MXFP4.

MXFP4 now runs on both the decode path and the prompt path: the decode kernels, the fused int8 prefill kernels,
the MMQ path, the FP16 dequantize fallback, the CPU expert rows and the n-gram table.

The scale is what decides the layout work:

| MXFP4 (39) | What it means |
| --- | --- |
| Block | `{ uint8_t e; uint8_t qs[16]; }` - 17 bytes for 32 values, `QK_MXFP4 32` |
| Scale | one E8M0 byte (a power of two) per 32 values, **inside** the block |
| Per expert | **no global scale, so no tail after the quantized rows** |
| Table | `kvalues_mxfp4` = `kvalues_fp4` (the e2m1 table doubled to 16 entries) |

Because there is no per-expert scale, the expert-layout API, the `native_fmt` scale plumbing, the CPU row
signatures and `tools/iq_pack.py` need nothing: the rows are the rows. The E8M0 byte in every block is the whole
scale a block has ([MXFP4.md](MXFP4.md)).

What runs MXFP4 now:

| Path | Where | File |
| --- | --- | --- |
| Decode experts (CUDA) | `k_mmvq_kernel<39, 20>` | `src/kernels/cuda/iq_kernels.cu:652` |
| Prefill, fused int8 experts | `native_w11_kernel<T_MXFP4>` | `src/prefill/moe_fused_iq.cu:1276-1298` |
| Prefill, ggml MMQ int8 | `run_mxfp4_w4a8` | `src/prefill/mmq_mxfp4_w4a8.cu` |
| Prefill, FP16 fallback | `group32<39>` / `dequantize_mxfp4` | `src/kernels/cuda/dequant_bf16.cu:94`, `include/strata/artifact/dequant.hpp:143` |
| CPU experts (AVX-512 / AVX2) | `mxfp4_512_*`, `mxfp4_256_*` | `src/kernels/cpu/mxfp4_avx512.cpp`, `src/kernels/cpu/mxfp4_avx2.cpp` |
| n-gram / PLE | `PleFormat::MXFP4` | `include/strata/kernels/ngram.hpp:52,59` |

## 2. The format, as the code reads it

```c++
typedef struct { uint8_t e; uint8_t qs[16]; } block_mxfp4;   // 17 bytes, 32 values
#define QK_MXFP4 32
#define QR_MXFP4 2
```

- Value `j` is the **low** nibble of `qs[j]`, value `j+16` the **high** nibble of the same byte.
- Nibble to value: the 16-entry e2m1 table `0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12`. Strata uses the doubled
  table, so the sign is the 4th bit and the magnitude is `table[nibble]`.
- `e` is E8M0: scale `2^(e-127)`; with the doubled table the kernel uses `2^(e-128)`. `e = 0xFF` is infinity,
  `e = 0` and `e = 1` are the two FP32 subnormals.
- The table is `kvalues_fp4` in `ggml-common.h`, and `ggml-common.h` spells `#define kvalues_mxfp4 kvalues_fp4`.
  A translation unit that includes `ggml-common.h` must use that name; declaring a private table there is a
  redefinition. `include/strata/artifact/dequant.hpp:143-163` has no such include and spells its own copy.

The device code is a header (`src/kernels/cuda/mxfp4_kernels.cuh`, 97 lines) included by
`src/kernels/cuda/iq_kernels.cu:1716-1720`, inside the anonymous namespace and after `#include "cvt.cuh"` -
`dq_mxfp4` calls `dq_iq4nl`. There is no separate `.cu`: the kernels are `__device__` functions called from
`k_mmvq_kernel`, `k_w11_kernel` and the prefill kernels, all of them in that one translation unit, which is also
the unit that carries the `__HIPCC__` guards the HIP build needs.

## 3. What AtomicChat's file needs

| AtomicChat AD-3.84bpw-IQ4_XS-M64 | |
| --- | --- |
| `n_embd` | **2880** |
| gate/up | **IQ2_S** |
| down | **MXFP4** |
| dense weights | **Q8_0**, stored BF16 by `--compat-bf16` (`setup.py:432-442`) |
| shards | 28 files, 84.9 GB |

Three consequences run through the whole implementation:

1. The 2560-wide assumption. `k_w11_kernel`'s IQ2_S gate is `n_embd / QK * ipb == 80`
   (`src/kernels/cuda/iq_kernels.cu:1381`); at 2880 the answer is 90, so the gate must not be a literal 80.
2. IQ2_S is not in the fused coverage list as it stands, so `gu_covered` gains it
   (`src/prefill/moe_fused_iq.cu:1138-1144`).
3. MXFP4's row arithmetic is 32-wide where every other expert format in the file is 256-wide.

## 4. The implementation

### 4.1 Decode: format registration and the matvec

`src/kernels/cuda/iq_kernels.cu`:

- `644`, `646` - `X(39)` in `STRATA_GU_FMTS` and `STRATA_D_FMTS`, which generate the `k_mmvq_kernel` and
  `k_w11_kernel` instantiations; `650`, `652`, `654` - `X(39)` in `STRATA_MMVQ_FMTS`, the expert-source table.
- `2047` - `case 39: dq_mxfp4(...)` in the matvec's decode switch.
- `2103`, `2105` - `is_iq` accepts 39, so MXFP4 rows go through the IQ4-style codebook path.
- `2905` - `xb_of`: `case 39: return (n / 32) * sizeof(block_mxfp4)`.
- `2986-2989` - the expert-row alignment check takes 32 when gate or down is MXFP4 (`const int align = (gu_type
  == 39 || d_type == 39) ? 32 : 256;`), used in both modulo checks.
- `2117-2124` - new `gu_ipb(int t)`, read from `STRATA_IPB`. `gu_rows` (`3717-3718`) and `d_rows` (`3807-3808`)
  use it, so the 80-vs-90 gate and the 32-vs-256 row stride follow the format instead of a literal.
- `1494`, `1514` - the `k_w11_kernel` K/`SUB_DOWN` comments and the `TD == 20 || TD == 39 || TD == 40` case.

### 4.2 Fused int8 prefill (`src/prefill/moe_fused_iq.cu`)

`T_MXFP4` (77-81) joins `T_IQ2_XS/T_IQ3_XXS/T_IQ2_S/T_IQ4_NL`. The unit is 17 bytes (`85-86`, `92-93`), one
32-value block, one E8M0 byte per unit, so a unit has no second scale row and no FP32 tail read.

- `187-191` `e8m0_half` - the E8M0 byte as the FP16 half of `2^(e-128)`, built from the mantissa bits, `0xFF` to
  infinity, `0` and `1` to the FP32 subnormals.
- `257-261` `ld32b` - one 32-byte unit as four aligned words when the row is aligned, byte loads otherwise.
- `310-319` `load_unit<T_MXFP4>` - `w[4] = bp[0]` carries the scale byte in the fifth word.
- `357-362` `convert` - `table16` over the two nibbles, `s0 = s1 = e8m0_half(w[4])`: gate and up share the block
  scale, which is what the format means.
- `492-495` - the codebook table is `kvalues_mxfp4`.
- `521`, `805-806` - `(2 * s + uj) * BS` is the MXFP4 row stride in the tile and in the HIP `native_w11_kernel`;
  `789-792` is its table, `805` the `T_MXFP4` branch.
- `1112` - `setup_one<T_MXFP4, false>`; `1146` - `d_covered` includes MXFP4, so a model whose down is MXFP4 is
  covered; `1138-1144` - `gu_covered` includes IQ2_S.
- `1276`, `1283`, `1298` - `STRATA_NW_D(T_MXFP4)`, `pick_ww(...)`, `launch<T_MXFP4, false>`.

Two levers came out of sm_120 tuning and are part of this tree:

- `60-65` `astages(int ww)` - four shared stages, two when the unit is 16 bytes (`ww == 2`), used at `106`,
  `471`, `481-482`, `543-586`. On a 48-SM RTX 5070 four stages kept one block per SM for MXFP4 at `ww == 4`;
  two stages at `ww == 2` is 44 KiB of shared memory and 126 registers per thread, which fits two blocks per SM.
- `1167-1192` `pick_ww(n, n_expert, gu_type, d_type)` - `ww = 2` for the measured pairs, `ww = 4` otherwise,
  with the measured pairs named in the comment above it.
- `1028-1039` `info_on()` and `1114-1123` - `STRATA_PF_INFO=1` prints the SM count, the blocks per SM (the min
  over the formats) and, for `native_kernel<T_MXFP4, false, 2>`, the shared-memory size, the register count and
  the blocks per SM.

On this tree the fused path on CUDA is opt-in: `native_supported()` (`1197-1216`) returns false when
`STRATA_PF_FUSED` is unset, and the default covers gfx12 IQ3_S/IQ3_XXS only. Running MXFP4 experts fused needs
`STRATA_PF_FUSED=1`; `STRATA_PF_FUSED_NATIVE=0` is the A/B switch back to the MMQ path.

### 4.3 MMQ: Strata's own int8 unit

`src/prefill/mmq_mxfp4_w4a8.cu` (29 lines) instantiates the shared MMQ template for MXFP4 weights and int8
activations - `mma.sync.m16n8k32.row.col.s32.s8.s8.s32`, sm_80 and up. `src/prefill/moe_mmq.cu` declares
`run_mxfp4_w4a8` (`185-187`), accepts the type in `strata_mmq_supported` (`221-222`) and dispatches it
(`383-385`). `CMakeLists.txt:1427` compiles it inside `strata_mmq`.

The unit is deliberately not ggml's `mmq-instance-mxfp4.cu`. That instance uses Blackwell's
`mma....kind::mxf4.block_scale`, which ptxas accepts only in `sm_100a/101a/103a` images, and Strata's MMQ object
is built for the architectures the user's card has. Adding MXFP4 to `_strata_mmq_types`
(`CMakeLists.txt:1409` - `q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0`) would compile that
instance for sm_120 and break the build. It is not in the list, and it stays out.

The MMQ branch is CUDA-only: the type case sits inside `#if !defined(GGML_USE_HIP)` (`moe_mmq.cu:221`), the HIP
list (`CMakeLists.txt:1554-1561`) compiles `_strata_mmq_types` only, and `STRATA_PREFILL_MMQ` is defined for
HIP only (`72`). On AMD, MXFP4 experts run fused; `docs/STRIX_HALO.md:108` names the formats.

### 4.4 FP16 fallback

`include/strata/artifact/dequant.hpp:143-163` - `dequantize_mxfp4` (its own `kvalues_mxfp4` copy,
`e8m0_to_fp32_half`), for the dequantize-to-FP16 path. `src/kernels/cuda/dequant_bf16.cu:35` (`kv_mxfp4`) and
`94-100` (`group32<39>`) are the BF16 side.

### 4.5 CPU experts

`src/kernels/cpu/mxfp4_avx512.cpp` (197 lines) and `src/kernels/cpu/mxfp4_avx2.cpp` (188 lines), declared in
`include/strata/kernels/cpu/mxfp4_avx512.hpp` / `mxfp4_avx2.hpp`, wired in
`src/kernels/cpu/native_expert.cpp:220-231` (gate/up) and `284-294` (down), with `kMxfp4Type = 39` in
`include/strata/kernels/cpu/native_expert.hpp:38-40`.

- AVX-512 works on pairs of blocks - 64 nibbles - with `_mm512_dpbusd_epi32` (16 int32 lanes, 4 values each), the
  sign moved onto the activation, `kMaxPairs = 64` (so `n <= 4096`), at most 7 tokens.
- AVX2 works on one block with `_mm256_maddubs_epi16` and `_mm256_madd_epi16`, `kMaxBlocks = 128`, `kMaxW = 8`.
- Selection is by ISA probe (`cpu_avx512_ok()`, `cpu_avx2_ok()`) plus the width fit; there are no runtime
  constructors in wide-ISA units (#2324). `STRATA_MXFP4_PREFETCH` is read lazily, default 2048.
- Knobs: `STRATA_NO_MXFP4_512`, `STRATA_NO_MXFP4_256`, `STRATA_MXFP4_PREFETCH`.
- AtomicChat's 2880 fits both: 45 pairs of 64, 90 blocks of 32.

### 4.6 n-gram / PLE

`include/strata/kernels/ngram.hpp:52` (`PLE_ROW_BYTES_MXFP4 = 85`) and `59` (`PleFormat::MXFP4`);
`src/kernels/ngram.cpp:178-179`; `src/ngram/ple_reader_test.cpp:108-109`.

### 4.7 GGUF geometry

No change: the GGUF reader already knows the format - `include/strata/artifact/gguf_reader.hpp:111` names it,
`:220` has `case 39` in the row arithmetic.

### 4.8 The prefill scratch sizes for a 2880-wide model

`src/prefill/prefill.cpp:1307-1309` and `1916` size the dequant scratch from the geometry:

```c++
o.take<uint16_t>((size_t) 2 * g.n_ff * g.n_embd, ok);   // gate/up halves
o.take<uint16_t>((size_t) g.n_embd * g.n_ff, ok);       // down half
```

The fixed forms `1280 * 2560` and `2560 * 640` are the same number as the geometry-sized forms for every
2560-wide model, so sizing from the geometry costs nothing there. For AtomicChat (`n_embd = 2880`, `n_ff = 640`)
the gate/up halves need 3,686,400 `uint16` - 409,600 more than the fixed form reserves - and the down half
1,843,200, 204,800 more. On AtomicChat this is the difference between scratch that fits and out-of-bounds writes
in the FP16 fallback.

### 4.9 Tests

- `src/kernels/iq_parity.cpp:29` (format list), `85-92` (the `mmvq` lambda, with the note that MXFP4 exists only
  as an expert format), `96`, `100`.
- `src/kernels/native_grouped_parity.cpp:66`, `70`, `84-85`, `351-352`, `355` - `check(39, 39, 2560, 640, ...)`.
- `src/kernels/cpu/mxfp4_avx512_parity.cpp` (377 lines) - the CPU parity binary: synthetic parity, `--expert` on
  the real AtomicChat layout at 2880, `--layers`, and `--bw THREADS TOKENS [ggml|avx2]` (modes at `338-344`).
  Registered at `CMakeLists.txt:1617-1621` as `mxfp4_avx512_parity` plus the CTest `mxfp4_cpu_parity` with a 120 s
  timeout.
- `tools/iq_fixture.py` - `"MXFP4": (32, 17)` (64), the E8M0 exponent range 124..130 plus the subnormals
  (79-82), `"MXFP4": 39` (91). `ref/load.py:38`.
- Setup side: `tools/test_setup_choices.py:125-126`, `tools/test_setup_menu.py:23` and `32-34` (the family list
  now includes `atomic`; MXFP4 has HIP decode kernels, so AMD keeps the family).

### 4.10 Setup, MCP and docs

- `setup.py:72` - the AtomicChat revision `142262902a46f7daed19c79d0771534c8106ad59`; `293-299` - the `IQ4_XS`
  entry with the measured numbers; `432-442` - the `atomic` family (`pack_args ["--compat-bf16"]`);
  `1759-1761` - `SUPPORTED_GGUFS`.
- `tools/strata_mcp.py:77-79` and `101` - the same model and family in the MCP fallback lists.
- New [MXFP4.md](MXFP4.md) (165 lines); `docs/MODELS.md:147-164` - the AtomicChat entry; `README.md:147-149` - the
  pointer.

## 5. Building this branch

**No architecture change is needed.** MXFP4 does not need an architecture-specific image, so the default
architecture list works as it is:

- `_strata_mmq_types` (`CMakeLists.txt:1409`) is unchanged - no `mxfp4`. That list names the ggml MMQ instances
  that get compiled, and ggml's MXFP4 instance - the one with `.kind::mxf4`, which ptxas refuses outside
  `sm_100a/101a/103a` - is therefore never compiled here.
- Strata's own `src/prefill/mmq_mxfp4_w4a8.cu` is a plain int8 `mma.sync` unit (sm_80+), compiled with the
  architecture list as given (`CMakeLists.txt:1427`).
- The only architecture-specific target in the CUDA build is `strata_mmq_w4a4` (`1446-1461`), which rewrites
  `12x` to `12xa` for its FP4 MMA unit. MXFP4 has no w4a4 unit, so nothing is added there.

So the default architecture list works: `CMAKE_CUDA_ARCHITECTURES` defaults to `120` (`161-162`), and RTX 20/30/40
(75/86/89) build from the same list.

One click (the normal path; `setup.py` detects the card and passes its architecture - `cuda_archs` at `3969`,
used at `3976-3981`):

```
START-HERE.bat --setup --family atomic --model IQ4_XS
```

Manual, Ninja (what `setup.py` does in `cmake_build`, `3847-3862`):

```
cmake -S . -B build -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target strata mxfp4_avx512_parity
```

Manual, Visual Studio (no Ninja in PATH):

```
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build --config Release --target strata mxfp4_avx512_parity
```

Two cards in one engine (this PC has an RTX 4060 and an RTX 5070): `-DCMAKE_CUDA_ARCHITECTURES="89;120"`.

ggml source: `-DSTRATA_GGML_DIR=<llama.cpp checkout>` (`CMakeLists.txt:1263`); without it the build fetches the
pinned commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (`1306-1310`, `third_party/ggml/VERSION.txt`).
`mmq.cuh` is not vendored, so the MMQ unit can only be compiled against a real ggml tree.

Tests: `mxfp4_avx512_parity` is built whenever `STRATA_NATIVE_EXPERTS` is on (default on, `107`, target at
`1617-1621`). Its CTest registration needs `enable_testing()`, which this tree calls under `STRATA_BUILD_TESTS`
(`1780-1781`); `STRATA_BUILD_TESTS` defaults on only when `tests/CMakeLists.txt` exists (`98-102`), and this
checkout has no `tests/CMakeLists.txt`, so pass `-DSTRATA_BUILD_TESTS=ON` and then:

```
ctest --test-dir build -R mxfp4_cpu_parity --output-on-failure
./build/mxfp4_avx512_parity --expert 16 128 4
```

CPU rows need no flags: `CMakeLists.txt:1322-1346` adds the two units to `strata_kernels_cpu` and sets
`/arch:AVX512` (`1327`) and `/arch:AVX2` (`1331`) per file for MSVC, `-mavx512...` (`1340-1342`) and
`-mavx2;-mfma;-mf16c` (`1346`) for GNU, and the unit is chosen at run time by the ISA probe.

### 5.1 The build as it ran on this PC (2026-10-14)

`cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON
-DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=D:/Strata/Strata/third_party/llama.cpp`
(CMake 4.4.1, CUDA 13.4.92, MSVC 14.51.36231, RTX 5070 = sm_120) produced `build\Release\strata.exe` and
`build\Release\mxfp4_avx512_parity.exe`. The architecture list as given needed no change, and the build shows why
in its own objects:

- `build\strata_mmq.dir\Release\` holds `mmq-instance-q2_0/iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq4_nl/iq4_xs/q8_0.obj`
  and **no** `mmq-instance-mxfp4.obj` - the `.kind::mxf4` instance is not in the build, so sm_120 is a legal
  architecture for this tree.
- `mmq_mxfp4_w4a8.obj` is there: Strata's int8 MXFP4 unit compiled against the user's ggml checkout
  (`mmq-config-ampere.cuh:348-363` carries the MXFP4 tile configs).
- `moe_fused_iq.obj`, `moe_mmq.obj` (prefill), `iq_kernels.obj`, `dequant_bf16.obj` (decode and the FP16/BF16
  fallback), `mxfp4_avx512.obj`, `mxfp4_avx2.obj` (CPU rows) all compiled.
- `build\CTestTestfile.cmake` registers `mxfp4_cpu_parity` -> `build/Release/mxfp4_avx512_parity.exe`, TIMEOUT 120,
  from `CMakeLists.txt:1620`.

Two MSVC fixes the first build needed, both in the new CPU files (they are part of this tree now):

- `src/kernels/cpu/mxfp4_avx512.cpp:73` - `_mm_setr_ps` takes four arguments; the two unused lanes are `0.0f`.
- `src/kernels/cpu/mxfp4_avx512_parity.cpp:298-303` - `from_float` takes `const float *` (the reference hidden is
  `double`, so it is copied to `hf` first), and `to_float` is in the base traits (`ggml.h:2976-2986`), not in
  `ggml_type_traits_cpu` (`ggml-cpu.h:117-120`: `from_float`, `vec_dot`, `vec_dot_type`).

## 6. Measurements

No speed number in this section is measured on the current build. The ones below were measured on this PC - an
RTX 5070 (12 GB), a Ryzen 7 5700X and 64 GB of RAM, PCIe 4.0, 204,800 context - on an earlier build of the same
MXFP4 work. Section 6.3 lists the runs that measure the current build.

### 6.1 The headline

Build: RTX 5070 (sm_120), CUDA 13.4.2, MSVC 14.51.36231, 64 GB RAM, `--spec 2`, int8 KV, random-word prompts
read at 8k/32k/128k, decode measured on a 128-token prompt.

| | atomic (MXFP4 down) | swift-1.5 (IQ3_XXS down) |
| --- | ---: | ---: |
| Prefill tok/s | 1494 | 2250 |
| Decode tok/s | 62,4 | 77,5 |
| TTFT at 128k | 121 s | 80 s |
| n-gram hit rate | ~77% | ~80% |

The formats are not comparable, so the ratio is not a verdict on MXFP4: atomic is 3.84 bpw with IQ2_S gate/up and
a 38.4 GB n-gram table in shard 2, swift-1.5 is a smaller model. The MXFP4 share of the expert work is the part
this report is about, and it is not separated in those runs.

### 6.2 Diagnostics

```
STRATA_PF_INFO=1: 48 SMs, 1 blocks a SM; MXFP4 down ww=2: 44 KiB shared, 126 registers a thread
loaded 38.23 GiB at 0.50 GiB/s
```

`STRATA_PF_INFO` is the print at `moe_fused_iq.cu:1114-1123`, added by this work; the text above is the line it
prints, so a run on the current build is directly comparable with it.

### 6.3 What is not measured on the current build, and how to fill it

1. Build for `120` (section 5) and record the compile output - the MXFP4 MMQ unit is the item that had not been
   compiled here before, because `mmq.cuh` is not vendored. **Done 2026-10-14**, see 5.1: `strata.exe` and
   `mxfp4_avx512_parity.exe` linked, `mmq_mxfp4_w4a8.obj` compiled, no `mmq-instance-mxfp4.obj`.
2. `ctest --test-dir build -R mxfp4_cpu_parity`, then `mxfp4_avx512_parity --expert 16 128 4` and
   `--bw 8 4 avx2`; record the parity counts and the GiB/s. The binary is built; the numbers are not recorded.
3. Load atomic with `STRATA_PF_INFO=1 STRATA_PF_FUSED=1` and record the format/`ww`/occupancy line, and the same
   run with `STRATA_PF_FUSED_NATIVE=0` for the MMQ path.
4. Prefill/decode/TTFT at 8k/32k/128k with `--spec 2`, beside a swift-1.5 run of the same build.
5. `tools/quant_quality.py` on the MXFP4 down tensors of one expert block, to keep the quantization error
   measured rather than assumed.

## 7. Known limits

- The fused path on CUDA is opt-in (`STRATA_PF_FUSED=1`); with the variable unset, MXFP4 experts go through MMQ
  (int8) or the FP16 fallback.
- MMQ for MXFP4 is CUDA-only; AMD runs it fused.
- Blackwell's native `mma...kind::mxf4` is not used. Strata's unit is int8 weights times int8 activations, which
  is what sm_80 through sm_120 have.
- AVX-512 rows cover `n <= 4096` and at most 7 tokens; wider rows fall back to AVX2, then to the scalar path.
- CPU MXFP4 rows are bit-exact against the ggml reference in the parity binary; they are not measured against a
  model run here.
- No performance number in this report is measured on the current build.
