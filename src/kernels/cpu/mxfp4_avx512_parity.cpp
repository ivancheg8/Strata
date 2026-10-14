// src/kernels/cpu/mxfp4_avx512_parity.cpp - the AVX-512 and AVX2 MXFP4 rows against ggml-cpu's own dot product.
//
//     mxfp4_avx512_parity [pack/experts.bin]
//     mxfp4_avx512_parity --layers pack/experts.bin [expert]
//     mxfp4_avx512_parity --expert pack/experts.bin [layer expert]
//     mxfp4_avx512_parity --bw THREADS TOKENS [ggml|avx2]
//
// Parity: every row x every token of the multi-token kernel against ggml_vec_dot_mxfp4_q8_0 on the same Q8_0
// activation. Both sum the products in integers; ggml-cpu's dot sums a whole 32-value block in integers before the
// scale and these sum four values per lane, so only the float additions are ordered differently and the check is a
// relative tolerance, not bit equality. Random blocks cover every code and exponents 120..135 (2^-8 .. 2^7); a real
// expert covers the checkpoint's own distribution - in AtomicChat's pack the MXFP4 is the down matrix (gate/up is
// IQ2_S), so the real rows checked are the down rows, 2560 of 640 values.
// Also the gate/up rows (n 2560) through silu: MXFP4 has no per-expert global scale, so no blob tail is read.
//
// --layers: one expert's real down matrix (default 7) of every layer, 64 rows.
// --expert: one whole AtomicChat expert as the CPU pool runs it - IQ2_S gate/up through ggml-cpu's dot, the hidden
// through Q8_0, the MXFP4 down through these kernels - against a double-precision reference.
// Speed: gate+up rows of one expert for 1..8 tokens, each way, as the CPU pool runs them.
//
// DRAM rate: THREADS threads each run whole experts (gate/up through silu, then down) out of 3.8 GB of blobs -
// 40x the L3 - for TOKENS tokens; GB/s of weights read. STRATA_MXFP4_PREFETCH sets the kernel's prefetch distance.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/mxfp4_avx2.hpp"
#include "strata/kernels/cpu/mxfp4_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

struct Kern {
    const char* name;
    void (*gu)(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int);
    void (*rows)(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);
};
const Kern k512{"avx512", cpu::mxfp4_512_gu_rows, cpu::mxfp4_512_rows};
const Kern k256{"avx2", cpu::mxfp4_256_gu_rows, cpu::mxfp4_256_rows};

constexpr int N = 2560, ROWS = 1280, BLK = 17;   // an expert's gate+up: 1280 rows of 2560 values, 17 bytes per 32
constexpr int FF = 640;                          // its down: 2560 rows of 640
constexpr size_t ROW_BYTES = (size_t) N / 32 * BLK, D_ROW = (size_t) FF / 32 * BLK;
constexpr size_t BLOB = (size_t) ROWS * ROW_BYTES + (size_t) N * D_ROW;
constexpr int E = 512;                           // experts per layer; IQ2_S gate/up is 17 bytes per 32 values too,
constexpr size_t LAYER = (size_t) E * BLOB;      // so an AtomicChat blob is exactly BLOB bytes, down at 2 * up_off

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Acts {
    std::vector<std::vector<uint8_t>> q;
    std::vector<const void*> p;
    Acts(int n, int nt, std::mt19937& rng) : q(nt), p(nt) {
        const ggml_type_traits_cpu* ta = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x(n);
        for (int t = 0; t < nt; ++t) {
            for (float& v : x) v = nd(rng) * (t + 1);                 // several magnitudes
            q[t].resize(ggml_row_size(GGML_TYPE_Q8_0, n));
            ta->from_float(x.data(), q[t].data(), n);
            p[t] = q[t].data();
        }
    }
};

float ref_dot(const uint8_t* row, int n, const void* a) {
    float s = 0.f;
    ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->vec_dot(n, &s, 0, row, 0, a, 0, 1);
    return s;
}

int report(const std::vector<std::vector<float>>& mine, const std::vector<std::vector<float>>& ref,
           int nt, int rows, double& worst, int& differ) {
    int bad = 0;
    for (int t = 0; t < nt; ++t) {
        double scale = 0;
        for (int r = 0; r < rows; ++r) scale = std::fmax(scale, std::fabs(ref[t][r]));
        for (int r = 0; r < rows; ++r) {
            const double e = std::fabs((double) mine[t][r] - ref[t][r]) / (scale > 0 ? scale : 1);
            worst = std::fmax(worst, e);
            if (!(e <= 1e-5)) ++bad;
            differ += std::memcmp(&mine[t][r], &ref[t][r], sizeof(float)) != 0;
        }
    }
    return bad;
}

int verdict(const Kern& k, const char* what, double worst, int rows, int nt, int bad, int differ) {
    std::printf("%-7s %-36s worst |mine - ggml| / max|ggml| = %.2e over %d rows x 1..%d tokens, %d beyond 1e-5, "
                "%d not bit-equal\n", k.name, what, worst, rows, nt, bad, differ);
    return bad;                                                 // bit equality is not claimed for MXFP4
}

// plain rows: `rows` rows of `n` values, `row_bytes` apart
int check_rows(const Kern& k, const uint8_t* w, int n, size_t row_bytes, int rows, const char* what,
               std::mt19937& rng) {
    constexpr int NT = 9;                                              // past one slice: 9 = 5 + 4
    Acts a(n, NT, rng);
    std::vector<std::vector<float>> mine(NT, std::vector<float>(rows)), ref(NT, std::vector<float>(rows));
    std::vector<float*> op(NT);
    for (int t = 0; t < NT; ++t) op[t] = mine[t].data();
    int bad = 0, differ = 0;
    double worst = 0;
    for (int nt = 1; nt <= NT; ++nt) {
        k.rows(w, row_bytes, n, a.p.data(), nt, op.data(), 0, rows);
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < rows; ++r) ref[t][r] = ref_dot(w + (size_t) r * row_bytes, n, a.p[t]);
        bad += report(mine, ref, nt, rows, worst, differ);
    }
    return verdict(k, what, worst, rows, NT, bad, differ);
}

// gate/up through silu: ff[t][r] = silu(g) * u, as native_gu_rows' ggml path computes it with both global scales
// at 1.0 - which is what MXFP4 has: the E8M0 byte is the only scale a block has, and there is no blob tail
int check_gu(const Kern& k, const uint8_t* blob, std::mt19937& rng) {
    constexpr int NT = 9, F = ROWS / 2;
    const size_t up_off = (size_t) F * ROW_BYTES;
    Acts a(N, NT, rng);
    std::vector<std::vector<float>> mine(NT, std::vector<float>(F)), ref(NT, std::vector<float>(F));
    std::vector<float*> op(NT);
    for (int t = 0; t < NT; ++t) op[t] = mine[t].data();
    int bad = 0, differ = 0;
    double worst = 0;
    for (int nt = 1; nt <= NT; ++nt) {
        k.gu(blob, ROW_BYTES, up_off, N, a.p.data(), nt, op.data(), 0, F);
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < F; ++r) {
                const float g = ref_dot(blob + (size_t) r * ROW_BYTES, N, a.p[t]);
                const float u = ref_dot(blob + up_off + (size_t) r * ROW_BYTES, N, a.p[t]);
                ref[t][r] = (g / (1.f + std::exp(-g))) * u;
            }
        bad += report(mine, ref, nt, F, worst, differ);
    }
    return verdict(k, "expert 0: silu(gate) * up", worst, F, NT, bad, differ);
}

void speed(const std::vector<uint8_t>& w, std::mt19937& rng, bool has512, bool has256) {
    const ggml_type_traits_cpu* tw = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    const ggml_type_traits_cpu* ta = ggml_get_type_traits_cpu(tw->vec_dot_type);
    const size_t act_bytes = ggml_row_size(tw->vec_dot_type, N);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> act(8, std::vector<uint8_t>(act_bytes));
    std::vector<float> x(N), sink(8 * ROWS);
    const void* ap[8];
    float* op[8];
    for (int t = 0; t < 8; ++t) {
        for (float& v : x) v = nd(rng);
        ta->from_float(x.data(), act[t].data(), N);
        ap[t] = act[t].data();
        op[t] = sink.data() + (size_t) t * ROWS;
    }
    std::printf("\n%-6s %10s %10s %8s %10s %8s   (one expert's gate+up, %d rows; best of 20)\n", "tokens",
                "ggml ms", "avx512 ms", "speedup", "avx2 ms", "speedup", ROWS);
    for (int nt = 1; nt <= 8; ++nt) {
        double tg = 1e30, tm = 1e30, t2 = 1e30;
        for (int rep = 0; rep < 20; ++rep) {
            double t0 = now_ms();
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < ROWS; ++r) tw->vec_dot(N, &op[t][r], 0, w.data() + (size_t) r * ROW_BYTES, 0, ap[t], 0, 1);
            tg = std::fmin(tg, now_ms() - t0);
            if (has512) {
                t0 = now_ms();
                k512.rows(w.data(), ROW_BYTES, N, ap, nt, op, 0, ROWS);
                tm = std::fmin(tm, now_ms() - t0);
            }
            if (has256) {
                t0 = now_ms();
                k256.rows(w.data(), ROW_BYTES, N, ap, nt, op, 0, ROWS);
                t2 = std::fmin(t2, now_ms() - t0);
            }
        }
        std::printf("%-6d %10.3f %10.3f %7.2fx %10.3f %7.2fx\n", nt, tg, has512 ? tm : 0.0, has512 ? tg / tm : 0.0,
                    has256 ? t2 : 0.0, has256 ? tg / t2 : 0.0);
    }
}

int bandwidth(int threads, int nt, bool use_ggml, const Kern& k) {
    constexpr int E_BLOBS = 1536;
    std::vector<uint8_t> blobs((size_t) E_BLOBS * BLOB);
    {
        std::vector<std::thread> fill;
        for (int t = 0; t < 16; ++t)
            fill.emplace_back([&, t] {
                uint64_t x = 0x9E3779B97F4A7C15ull * (t + 1);
                const size_t n8 = blobs.size() / 8, a = n8 * t / 16, b = n8 * (t + 1) / 16;
                for (size_t i = a; i < b; ++i) {
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    uint64_t v = x & 0x7777777777777777ull;       // exponents stay in 120..127, codes are any nibble
                    std::memcpy(blobs.data() + 8 * i, &v, 8);
                }
            });
        for (auto& f : fill) f.join();
    }
    std::mt19937 rng(7);
    Acts ax(N, nt, rng), ah(FF, nt, rng);
    const ggml_type_traits_cpu* tw = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    double best = 1e30;
    for (int pass = 0; pass < 3; ++pass) {
        const double t0 = now_ms();
        std::vector<std::thread> pool;
        for (int w = 0; w < threads; ++w)
            pool.emplace_back([&, w] {
                std::vector<float> ff((size_t) nt * (ROWS / 2)), out((size_t) nt * N);
                std::vector<float*> fp(nt), op(nt);
                for (int t = 0; t < nt; ++t) { fp[t] = ff.data() + (size_t) t * (ROWS / 2); op[t] = out.data() + (size_t) t * N; }
                for (int e = w; e < E_BLOBS; e += threads) {
                    const uint8_t* b = blobs.data() + (size_t) e * BLOB;
                    const size_t up_off = (size_t) (ROWS / 2) * ROW_BYTES, d_off = (size_t) ROWS * ROW_BYTES;
                    if (!use_ggml) {
                        k.gu(b, ROW_BYTES, up_off, N, ax.p.data(), nt, fp.data(), 0, ROWS / 2);
                        k.rows(b + d_off, D_ROW, FF, ah.p.data(), nt, op.data(), 0, N);
                    } else {
                        for (int r = 0; r < ROWS / 2; ++r)
                            for (int t = 0; t < nt; ++t) {
                                float g = 0.f, u = 0.f;
                                tw->vec_dot(N, &g, 0, b + (size_t) r * ROW_BYTES, 0, ax.p[t], 0, 1);
                                tw->vec_dot(N, &u, 0, b + up_off + (size_t) r * ROW_BYTES, 0, ax.p[t], 0, 1);
                                fp[t][r] = (g / (1.f + std::exp(-g))) * u;
                            }
                        for (int r = 0; r < N; ++r)
                            for (int t = 0; t < nt; ++t) tw->vec_dot(FF, &op[t][r], 0, b + d_off + (size_t) r * D_ROW, 0, ah.p[t], 0, 1);
                    }
                }
            });
        for (auto& p : pool) p.join();
        best = std::fmin(best, now_ms() - t0);
    }
    const char* pf = std::getenv("STRATA_MXFP4_PREFETCH");
    std::printf("%s  threads %d  tokens %d  prefetch %s: %.1f ms for %d experts -> %.1f GB/s, %.1f us/expert/thread\n",
                use_ggml ? "ggml  " : k.name, threads, nt, use_ggml ? "-" : (pf ? pf : "2048"), best, E_BLOBS,
                (double) E_BLOBS * BLOB / best / 1e6, best * 1000.0 * threads / E_BLOBS);
    return 0;
}

// one expert of a pack, 64 rows of its real MXFP4 down matrix
int layers(const char* path, int e) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> blob(BLOB);
    std::mt19937 rng(3);
    int bad = 0;
    const Kern& k = cpu::cpu_avx512_ok() ? k512 : k256;
    for (int l = 0; l < 48; ++l) {
        f.seekg((std::streamoff) ((size_t) l * E + (size_t) e) * BLOB);
        if (!f.read((char*) blob.data(), (std::streamsize) blob.size())) { std::fprintf(stderr, "cannot read layer %d\n", l); return 2; }
        char what[96];
        std::snprintf(what, sizeof what, "L%02d expert %d: down rows", l, e);
        bad += check_rows(k, blob.data() + (size_t) ROWS * ROW_BYTES, FF, D_ROW, 64, what, rng);
    }
    std::printf("RESULT: %s\n", bad ? "MISMATCH" : "parity OK");
    return bad ? 1 : 0;
}

// one whole AtomicChat expert as the CPU pool runs it: IQ2_S gate/up through ggml-cpu's dot, the hidden through
// Q8_0, the MXFP4 down through these rows - against a double-precision reference
int expert_e2e(const char* path, int layer, int e) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> b(BLOB);
    f.seekg((std::streamoff) ((size_t) layer * E + (size_t) e) * BLOB);
    if (!f.read((char*) b.data(), (std::streamsize) b.size())) { std::fprintf(stderr, "cannot read %s\n", path); return 2; }
    cpu::NativeFmt fmt;
    std::string err;
    if (!cpu::native_fmt(22, 39, N, FF, fmt, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    const int F = FF;
    std::vector<float> wg((size_t) F * N), wu((size_t) F * N), wd((size_t) N * FF);
    const ggml_type_traits* ti = ggml_get_type_traits(GGML_TYPE_IQ2_S), *tm = ggml_get_type_traits(GGML_TYPE_MXFP4);
    ti->to_float(b.data(), wg.data(), (int64_t) F * N);
    ti->to_float(b.data() + fmt.up_off, wu.data(), (int64_t) F * N);
    tm->to_float(b.data() + fmt.down_off, wd.data(), (int64_t) N * FF);

    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 1.f);
    const ggml_type_traits_cpu* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    double num = 0, den = 0, worst = 0;
    for (int trial = 0; trial < 8; ++trial) {
        std::vector<float> x(N);
        for (float& v : x) v = nd(rng);
        std::vector<double> h(F);
        for (int r = 0; r < F; ++r) {
            double g = 0, u = 0;
            for (int k = 0; k < N; ++k) { g += (double) wg[(size_t) r * N + k] * x[k]; u += (double) wu[(size_t) r * N + k] * x[k]; }
            h[r] = g / (1 + std::exp(-g)) * u;
        }
        std::vector<float> hf(F), hq(F);                          // what the down sees: the hidden as Q8_0
        {
            for (int r = 0; r < F; ++r) hf[r] = (float) h[r];     // from_float takes float, the reference is double
            std::vector<uint8_t> hqb(ggml_row_size(GGML_TYPE_Q8_0, F));
            q8->from_float(hf.data(), hqb.data(), F);
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(hqb.data(), hq.data(), F);   // to_float: base traits
        }
        std::vector<double> y(N, 0.0);
        for (int r = 0; r < N; ++r) {
            double s = 0;
            for (int k = 0; k < F; ++k) s += (double) wd[(size_t) r * FF + k] * hq[k];
            y[r] = s;
        }
        std::vector<uint8_t> xq(fmt.act_bytes);
        cpu::native_quant_act(fmt, x.data(), xq.data());
        const void* ap[1] = {xq.data()};
        std::vector<float> ff(F), out(N);
        float* fp[1] = {ff.data()};
        float* op[1] = {out.data()};
        cpu::native_gu_rows(fmt, b.data(), ap, 1, fp, 0, F);
        std::vector<uint8_t> hqb(fmt.h_bytes);
        cpu::native_quant_h(fmt, ff.data(), hqb.data());
        const void* hp[1] = {hqb.data()};
        cpu::native_down_rows(fmt, b.data(), hp, 1, op, 0, N);
        for (int r = 0; r < N; ++r) {
            const double d = (double) out[r] - y[r];
            num += d * d;
            den += y[r] * y[r];
            worst = std::fmax(worst, std::fabs(d) / (std::fabs(y[r]) > 1e-30 ? std::fabs(y[r]) : 1e-30));
        }
    }
    std::printf("layer %d expert %d: IQ2_S gate/up + MXFP4 down, %d rows of %d: output rel. error %.3f%%, worst "
                "element %.3f%% (IQ2_S gate/up through ggml-cpu, the down through %s)\n", layer, e, N, FF,
                100 * std::sqrt(num / den), 100 * worst, cpu::cpu_avx512_ok() ? "avx512" : "avx2");
    return num / den > 1e-3 ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    if (argc >= 3 && std::string(argv[1]) == "--expert")
        return expert_e2e(argv[2], argc >= 4 ? std::atoi(argv[3]) : 0, argc >= 5 ? std::atoi(argv[4]) : 0);
    const bool has512 = cpu::cpu_avx512_ok(), has256 = cpu::cpu_avx2_ok();
    if (argc >= 3 && std::string(argv[1]) == "--layers") return layers(argv[2], argc >= 4 ? std::atoi(argv[3]) : 7);
    if (argc >= 4 && std::string(argv[1]) == "--bw") {
        const std::string how = argc >= 5 ? argv[4] : "";
        return bandwidth(std::atoi(argv[2]), std::atoi(argv[3]), how == "ggml", how == "avx2" || !has512 ? k256 : k512);
    }
    std::vector<const Kern*> ks;
    if (has512) ks.push_back(&k512);
    if (has256) ks.push_back(&k256);
    std::mt19937 rng(20261014);
    std::vector<uint8_t> w(BLOB);
    // random blocks: every code, exponents 120..135 (2^-8 .. 2^7), as a quantizer writes them
    std::uniform_int_distribution<int> byte(0, 255), ex(120, 135);
    for (size_t b = 0; b + BLK <= BLOB; b += BLK) {
        w[b] = (uint8_t) ex(rng);
        for (int q = 1; q < BLK; ++q) w[b + q] = (uint8_t) byte(rng);
    }
    int bad = 0;
    for (const Kern* k : ks) {
        bad += check_rows(*k, w.data(), N, ROW_BYTES, ROWS, "random blocks, n 2560", rng);
        bad += check_rows(*k, w.data(), FF, D_ROW, N, "random blocks, n 640", rng);
        bad += check_gu(*k, w.data(), rng);
    }
    if (argc >= 2) {                                                   // layer 0, expert 0 of the pack: its MXFP4 rows
        std::ifstream f(argv[1], std::ios::binary);
        if (!f.read((char*) w.data(), (std::streamsize) w.size())) {
            std::fprintf(stderr, "cannot read %zu bytes of %s\n", w.size(), argv[1]);
            return 2;
        }
        for (const Kern* k : ks)
            bad += check_rows(*k, w.data() + (size_t) ROWS * ROW_BYTES, FF, D_ROW, N,
                              "AtomicChat expert 0: down rows", rng);
    }
    speed(w, rng, has512, has256);
    std::printf("\nRESULT: %s\n", bad ? "MISMATCH" : "parity OK");
    return bad ? 1 : 0;
}
