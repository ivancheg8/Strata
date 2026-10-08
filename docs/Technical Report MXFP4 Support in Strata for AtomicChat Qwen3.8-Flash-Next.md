# Technical Report: MXFP4 Support in Strata for AtomicChat Qwen3.8-Flash-Next

## Executive Summary

This document details the work required to run the **AtomicChat Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64** model in Strata. The model uses an experimental quantization format (**MXFP4**, ggml type 39) which was not supported by Strata. The work resulted in:

- Full support for MXFP4 (type 39) in the Strata engine (CUDA sm_80+ and gfx11 paths)
- A bug fix in `prefill.cpp` for models with `n_embd != 2560`
- Automatic fused-path selection based on tensor types (no manual env vars; NVIDIA sm_80+ — gfx11 keeps `STRATA_PF_FUSED=1`)
- A new diagnostic flag `STRATA_PF_INFO=1` for measuring per-format occupancy
- Integration into `setup.py` for one-click installation
- Benchmark results on RTX 5070 (consumer Blackwell, sm_120)

The model achieves **~50 tok/s decode** and **~1390 tok/s prefill** on a consumer RTX 5070 with 64 GB RAM. No quality benchmark (KL, perplexity) was run as part of this work.

---

## 1. Background: What Makes AtomicChat Special

The AtomicChat quantization differs from the official Unsloth/ISTA-DASLab quants in several key ways:

| Aspect | Unsloth/ISTA-DASLab | AtomicChat |
|--------|---------------------|------------|
| **Expert down** | IQ4_NL / IQ4_XS | **MXFP4** (type 39) |
| **Expert gate/up** | IQ2_S / IQ3_S | IQ2_S |
| **Dense layers (hc_*)** | BF16 | **Q8_0** |
| **Shards** | 2-3 | **28** |
| **PLE table** | Included | **Separate 38 GB shard** |

### Key challenges for Strata integration:

1. **MXFP4 is a new format** (OCP standard, not an i-quant). It uses E8M0 scale + E2M1 values with a 16-entry LUT. 17-byte blocks (1 byte scale + 16 bytes packed nibbles for 32 values). Not supported anywhere in Strata.
2. **Dense layers are Q8_0**, but Strata's `bf16_proj()` requires BF16 residents → requires `--compat-bf16` packing.
3. **28 shards** means `len(shards) > 2`, so `--ple-gguf` is automatically excluded by existing logic (the PLE tensor lives in shard 2, not shard 1).
4. **`n_embd = 2880`** (not the hardcoded 2560 from Coder models) → exposed a latent bug in `prefill.cpp` buffer allocation.
5. **Consumer Blackwell (sm_120)** lacks the server-side `mma.sync...mxf4.block_scale` instructions (sm_100a/101a/103a only). Strata's kernels never emit them — the fused path decodes MXFP4 to int8 codes and runs on `mma.sync m16n8k32 s8` (sm_80+). The instructions appear in ggml's MXFP4 MMQ instance (`mmq-instance-mxfp4.cu`, added to `_strata_mmq_types` at `CMakeLists.txt:1181` by this work), and compiling that instance for `sm_120` fails in ptxas; building with `CMAKE_CUDA_ARCHITECTURES=89` avoids it (see §3).

---

## 2. Technical Changes

### 2.1 MXFP4 device code — new file `src/kernels/cuda/mxfp4_kernels.cuh`

**Rationale:** MXFP4 is architecturally different from the i-quant family (no codebooks, no signs, no grid lookup — just E8M0 scale + 16-value LUT). Mixing it into `iq_kernels.cu` violated separation of concerns. However, due to CUDA TU (Translation Unit) rules and lack of RDC (Relocatable Device Code), the device code must be in a header included into `iq_kernels.cu` inside the anonymous namespace.

**Contents (96 lines):**
- `get_int_b1()` — byte-wise 4-byte read for unaligned 17-byte blocks
- `e8m0_half()` — E8M0 → float conversion (halved, since `kvalues_mxfp4` are doubled)
- `vec_dot_mxfp4_q8_1()` — dot product for MMVQ path using DP4A
- `Fmt<39>` — format descriptor (qk=32, ipb=2, step=2)
- `kSplit<39>` + `Split<39>` — multi-column decode specialization
- `dq_mxfp4<dst_t>()` — dequantization kernel (superblock of 256 values)

**Integration point:** `#include "mxfp4_kernels.cuh"` at line 1575 of `iq_kernels.cu`, inside the anonymous namespace, after `cvt` specializations and before the first `Fmt<39>`/`Split<39>` instantiation.

### 2.2 Dispatch integration in `src/kernels/cuda/iq_kernels.cu`

Minimal changes — only dispatch, not implementation:
- `X(39)` in `STRATA_GU_FMTS`, `STRATA_D_FMTS`, `STRATA_MMVQ_FMTS` macros
- `case 39: dq_mxfp4(...)` in `dq_dispatch()`
- `t == 39` in `is_iq()` (both `#ifdef` branches)
- `case 39: return (n/32) * sizeof(block_mxfp4);` in `iq_row_bytes()`
- `native_expert_supported()` — `required_align = 32` for type 39 (vs 256 for i-quants)
- Fail-fast guards in `iq_dequant_f16` / `iq_embed_rows` / `iq_dequant_gu_f16` requiring `n % 256 == 0` — pre-existing (`iq_kernels.cu:2757`, `:2774`, `:2788`), unchanged by this work

### 2.3 Fused prefill path — `src/prefill/moe_fused_iq.cu`

Added MXFP4 (type 39) as a supported format for **down-projections** of native expert blocks on both CUDA and gfx11 paths:

- `T_MXFP4 = GGML_TYPE_MXFP4` constant
- `static_assert(sizeof(block_mxfp4) == 17, ...)`
- `block_bytes(39) = 17`
- `e8m0_half()` helper (copy from `mxfp4_kernels.cuh`)
- `ld32b()` — byte-wise 4-byte read (analog of `get_int_b1`)
- `load_unit<T_MXFP4>` — 4 words `qs` aligned to 4 → direct read, else byte-wise; `w[4] = bp[0]` (scale byte `e`)
- `convert<T_MXFP4>` — `table16(w[k], kv, q[k], q[4+k])` ×4, `s0 = s1 = e8m0_half(w[4])`
- CUDA `native_kernel`: table `kv` built from `kvalues_mxfp4` (alongside IQ4_XS/IQ4_NL)
- HIP `native_w11_kernel`: same table construction
- `d_covered`: `t == T_Q2_0 || t == T_IQ4_NL || t == T_MXFP4`
- CUDA `unit()` indexing for down: MXFP4 joins the IQ4_NL-style `(2 * s + uj)` sub-block addressing
- `setup_one<T_MXFP4, false>` added to the `dev_info()` occupancy minimum
- Dispatchers: `launch<T_MXFP4, false>(...)` for CUDA, `STRATA_NW_D(T_MXFP4)` for HIP

**Key correctness properties verified:**
- Natural nibble order: `d[0]` = values 0-15, `d[1]` = 16-31 (matches `dq_mxfp4`)
- Sign already embedded in int8 codes via table indices (entries 8-15 are negative)
- `MAGIC` trick safe: `|Σ q·a| ≤ 32·24·127 = 97536 < 2²²` (`kvalues_mxfp4` max magnitude is 24)
- One scale per 32 values = two mma `h` half-blocks, so `s0 = s1`

### 2.4 Safe dequantization fallback — `src/kernels/cuda/dequant_bf16.cu`

Added MXFP4 support for the prefill dequant path (used when fused path is not available):

- `__constant__ int8_t kv_mxfp4[16]` — doubled E2M1 values (matches `kvalues_mxfp4` from ggml)
- `group32<39>` branch — byte-wise dequantization (safe for 17-byte blocks, no vectorized loads)
- `geometry()`: `case 39: block_elems = 32; block_bytes = 17; return true;`
- `launch()`: `case 39: STRATA_DQ(39);`

### 2.5 MMQ path — `src/prefill/moe_mmq.cu`

- `case GGML_TYPE_MXFP4:` in `supported()` whitelist
- `case GGML_TYPE_MXFP4: mul_mat_q_case<GGML_TYPE_MXFP4>(...); break;`

### 2.6 Dynamic buffer allocation — `src/prefill/prefill.cpp` (bug fix)

**Problem:** Hardcoded buffer sizes `1280*2560` and `2560*640` assumed `n_embd=2560` (Coder model). AtomicChat has `n_embd=2880`, causing heap corruption and `unspecified launch failure` on long prompts.

**Fix:** Replaced hardcoded literals with dynamic calculation based on actual model geometry:

```cpp
// Line 1003 (Prefill::carve)
for (int i = 0; i < DQ; ++i) {
    m.dq_gu[i] = o.take<uint16_t>((size_t) 2 * g.n_ff * g.n_embd, ok);
    m.dq_d[i]  = o.take<uint16_t>((size_t) g.n_embd * g.n_ff, ok);
}

// Line 1523 (Prefill::bytes_needed_impl) — mirror
for (int i = 0; i < DQ; ++i) {
    o.take<uint16_t>((size_t) 2 * g.n_ff * g.n_embd, ok);
    o.take<uint16_t>((size_t) g.n_embd * g.n_ff, ok);
}
```

Both locations must be updated together — otherwise `take` bounds-checking fails and pointers become `nullptr`.

**Impact:** +1.2 MB per prefill instance for `n_embd=2880` vs `2560`. Zero impact on existing models with `n_embd=2560` (formula reduces to the same literals).

### 2.7 GGUF geometry validation — `include/strata/artifact/gguf_reader.hpp`

```cpp
// Line 217 in block_geometry()
case 39: elems = 32; bytes = 17; return true;
```

This single line unblocks loading for all MXFP4 tensors (the experts' down matrices; the dense `hc_*` tensors are Q8_0, which the validation already covered) through the validation path in `native_dense.cpp:187`, `gguf_reader.cpp:39`, and `gguf_reader.hpp:480`.

### 2.8 Python mirrors — `ref/load.py` and `tools/gguf_reader.py`

- `ref/load.py`: Added `"MXFP4": (32, 17)` to `GEOM` dict (keyed by type name, not numeric ID)
- `tools/gguf_reader.py`: Already contained `"MXFP4": (32, 17)` and `"NVFP4": (64, 36)` — no changes needed

### 2.9 setup.py integration

Added the AtomicChat family to `setup.py` for one-click installation:

```python
# HF_REVISIONS
"AtomicChat/Qwen3.8-Flash-Next-GGUF": "142262902a46f7daed19c79d0771534c8106ad59",  # 2026-10-07

# MODELS
"IQ4_XS": {"about": "AtomicChat's 3.84 bpw build (MXFP4 experts); the 38 GB n-gram table is read from the SSD, "
                    "", "download_gb": 80, "ram_gb": 48, "arena_gb": 41.05,
           "families": ("atomic",)},

# FAMILIES
"atomic": {"title": "AtomicChat Qwen3.8-Flash-Next", "by": "AtomicChat quants",
           "about": "IQ4_XS, Q4_K_M, Q5_K_M",
           "hf": hf("AtomicChat/Qwen3.8-Flash-Next-GGUF") + "Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64/",
           "file": "Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64-{i:05d}-of-00028.gguf", "shards": 28,
           "tag": "atomic-", "mmproj_hf": hf("AtomicChat/Qwen3.8-Flash-Next-GGUF"),
           "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-atomic",
           "pack_args": ["--compat-bf16"]},

# SUPPORTED_GGUFS now names "AtomicChat's AD-3.84bpw-IQ4_XS-M64" among the runnable GGUFs
```

Mirrored in `tools/strata_mcp.py` (`FALLBACK_MODELS`, `FALLBACK_FAMILIES`) and updated assertion in `tools/test_setup_choices.py`.

### 2.10 Automatic fused-path selection

Replaced manual `STRATA_PF_FUSED=1` requirement with automatic detection:

```cpp
// src/prefill/moe_fused_iq.cu:1107-1116 (native_supported)
// native_supported() now uses enabled() instead of requested()
// enabled() returns true by default on CUDA sm_80+
// Decision: !off && enabled() && gu_covered(gu) && d_covered(d) && dev_info().ok
```

`STRATA_PF_FUSED_NATIVE=0` keeps the native packs on MMQ (the A/B switch, read inside `native_supported`). This propagates through all callers: `mmq_plan()`, `fused_ring()`, `fused_nat`, `bind_stage_helper`. The pack with a layer outside supported lists safely falls back to MMQ buffers.

### 2.11 Diagnostic flag `STRATA_PF_INFO=1`

Added per-format diagnostic output:
```
strata: fused experts (native): 48 SMs, 1 blocks a SM (the min over the formats);
        MXFP4 down ww=2: 44 KiB shared, 126 registers a thread, 1 blocks a SM
```

Uses `cudaFuncGetAttributes` for `numRegs` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for occupancy (`moe_fused_iq.cu:1023-1033`). A second line is printed once per format pair by `pick_ww` (`moe_fused_iq.cu:1093-1101`): the tile shape that pair runs with — `strata: fused experts (native): gu=%d d=%d, %d weight rows a tile, %.1f rows an expert, %d/%d KiB shared`. Essential for diagnosing whether a bottleneck is shared memory, registers, or DRAM.

### 2.12 ASTAGES optimization (shared memory reduction)

Changed `constexpr int ASTAGES = 4;` to `constexpr int astages(int ww) { return ww == 2 ? 2 : 4; }`.

**Effect:** MXFP4 down ww=2: 64.5 → 44.5 KiB shared; IQ2_S gate/up ww=2: 72.5 → 52.5 KiB.

**Result on RTX 5070:** Occupancy remained at 1 block/SM because register pressure (126 regs/thread) is the actual bottleneck, not shared memory. The shared-memory saving is computed from `smem_bytes`, but its effect on speed on other GPUs and tile widths is not measured.

---

## 3. Build Instructions

```bash
# Build with the sm_89 target (workaround for the ggml MXFP4 MMQ instance; see below)
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DGGML_CUDA=ON \
  -DSTRATA_GGML_DIR="<path-to-llama.cpp-master>" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  .

cmake --build build --config Release --parallel 8
```

**Why sm_89 here, when the project's default is 120 (`CMakeLists.txt:152-153`):** Strata's own kernels contain no `.kind::mxf4` instructions — the fused path decodes MXFP4 to int8 codes and runs on `mma.sync m16n8k32 s8` (sm_80+), the MMVQ path on DP4A. The failure `Feature '.kind::mxf4' not supported on .target 'sm_120'` comes from ptxas compiling ggml's `mmq-instance-mxfp4.cu`, added to `_strata_mmq_types` by this work (`CMakeLists.txt:1181`): its block-scale `mma` is written for server Blackwell (sm_100a/101a/103a) and the ggml checkout used here does not exclude sm_120 from it. Building for `89` compiles ggml's DP4A path for MXFP4; CMake emits `compute_89` PTX for the last architecture, which the driver JITs on the RTX 5070. All benchmark numbers below were measured on such a build.

---

## 4. Benchmark Results (RTX 5070, 64 GB RAM, PCIe 4.0)

### Comparison: swift-1.5-iq3_xxs vs atomic-iq4_xs

| Metric | swift-1.5-iq3_xxs | atomic-iq4_xs |
|--------|-------------------|---------------|
| **Prefill** | 2336 tok/s | 1390 tok/s |
| **Decode** | 62 tok/s | 50 tok/s |
| **TTFT (181K ctx)** | 78 s | 133 s |
| **Expert cache hit** | ~80% | ~77% |

### Why prefill is slower for AtomicChat

The fused path **is** active for AtomicChat's experts — IQ2_S gate/up with MXFP4 down is covered (§2.3, §2.10), and the first diagnostic line below prints only when the native fused setup succeeds. The slowdown is not a fallback to `dequant_bf16 → cuBLAS`. It is the expert streaming and the wider model:

1. The experts do not fit resident: 38.23 GiB of expert arena are read from RAM at a measured 0.50 GiB/s (unbuffered) during generation; prefill streams the same rows through the fused kernels' L2 prefetch.
2. Occupancy is 1 block/SM (126 registers a thread; two 512-thread blocks would need ≤64), so the fused kernels run at half their thread budget.
3. AtomicChat is wider (n_embd 2880 against 2560) and its 38 GB n-gram table is read from the SSD.

The 0.50 GiB/s is the measured read rate of the expert arena, not the PCIe limit: PCIe 4.0 x16 carries ~25 GiB/s, so the bound is the read path (DRAM access pattern of the unbuffered arena), not the bus.

### Diagnostic output confirming the bottleneck

```
strata: fused experts (native): 48 SMs, 1 blocks a SM
        MXFP4 down ww=2: 44 KiB shared, 126 registers a thread, 1 blocks a SM
strata generate: expert arena read unbuffered ... loaded 38.23 GiB at 0.50 GiB/s
```

The 126 registers/thread (vs 64 required for 2 blocks/SM) is the actual occupancy limiter, not shared memory.

---

## 5. Known Limitations

1. **Expert streaming rate:** the measured 0.50 GiB/s arena read (38.23 GiB, unbuffered) dominates prefill and TTFT. It is far below the PCIe 4.0 x16 ceiling (~25 GiB/s), so the bound is the read path, not the bus; a pack that fits fully in VRAM removes it.

2. **Register pressure prevents 2 blocks/SM:** 126 registers/thread vs 64 needed. Reducing this would require manual rework of `mma.sync` instructions and risks register spilling.

3. **Q8_0 dense layers require `--compat-bf16`:** Strata's `bf16_proj()` requires BF16 residents. Adding Q8_0 support to the CUDA fused path would require increasing `raw[5]` to `raw[9]` (since `raw_words(Q8_0) == 9`), with unmeasured register pressure implications. Currently Q8_0 is only supported on gfx11 behind `STRATA_PF_FUSED_KQ=1`.

4. **Server-side MXFP4 MMA unavailable on consumer Blackwell:** The `mma.sync...mxf4.block_scale` instructions exist only on sm_100/101/103 (B200/GB200). Consumer sm_120 lacks them.

5. **No parity tests for MXFP4:** Unlike i-quant formats, there are no `*_parity.cpp` tests for MXFP4, and `tests/cuda/prefill_fused_iq_test.cpp` does not include MXFP4 layers. Correctness was verified against `ggml`'s `dequantize_row_mxfp4` reference implementation. This work added `mxfp4` to `_strata_mmq_types` (`CMakeLists.txt:1181`), so ggml's MMQ instance is compiled and available as the parity reference.

---