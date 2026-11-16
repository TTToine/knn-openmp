// OpenMP variants. Each function is ONE optimisation idea, so that the report
// can compare them one against the other. Register new variants in variants.cpp.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

#include "knn.hpp"
#include "knn_common.hpp"

// v1 - parallel loop over test samples.
// Each test sample is independent (embarrassingly parallel): no synchronisation
// is needed, every thread writes a distinct out[i]. schedule(runtime) lets the
// benchmark script select static/dynamic/guided via the OMP_SCHEDULE variable.
void knn_omp_naive(const Dataset& train, const Dataset& test,
                   const KnnParams& params, std::uint8_t* out) {
    const std::size_t d = train.d;

    std::vector<float> train_norm, test_norm;
    if (params.metric == Metric::Cosine) {
        train_norm.resize(train.n);
        test_norm.resize(test.n);
        row_norms(train, train_norm.data());   // serial on purpose: see omp_blocked4_pn
        row_norms(test, test_norm.data());
    }

    const long n_test = static_cast<long>(test.n);
    // default(none): every variable used in the region must be listed. All of
    // them are shared and only READ, except out[] (each out[i] written by exactly
    // one thread). `top`, `q`, `t`, `dist` are declared inside -> private.
#pragma omp parallel for schedule(runtime) default(none) \
    shared(train, test, params, out, train_norm, test_norm, d, n_test)
    for (long i = 0; i < n_test; ++i) {
        const float* q = test.row(i);
        TopK top(params.k);
        for (std::size_t j = 0; j < train.n; ++j) {
            const float* t = train.row(j);
            const float dist = (params.metric == Metric::Euclidean)
                                   ? sq_euclidean(q, t, d)
                                   : cosine_distance(dot(q, t, d), test_norm[i], train_norm[j]);
            top.push(dist, train.y[j]);
        }
        out[i] = top.vote();
    }
}

// v2 - explicit SIMD on the distance loop (otherwise identical to v1).
// Without -ffast-math GCC must keep the scalar summation order, so the
// reductions in knn_common.hpp stay scalar (one dependent add chain).
// `omp simd reduction(+:s)` gives the compiler permission to keep one partial
// sum per SIMD lane (8 floats with AVX2) and combine them at the end.
// The summation order changes -> results may differ in the last bits.
static inline float sq_euclidean_simd(const float* a, const float* b, std::size_t d) {
    float s = 0.0f;
#pragma omp simd reduction(+:s)
    for (std::size_t i = 0; i < d; ++i) {
        const float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

static inline float dot_simd(const float* a, const float* b, std::size_t d) {
    float s = 0.0f;
#pragma omp simd reduction(+:s)
    for (std::size_t i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
}

void knn_omp_simd(const Dataset& train, const Dataset& test,
                  const KnnParams& params, std::uint8_t* out) {
    const std::size_t d = train.d;

    std::vector<float> train_norm, test_norm;
    if (params.metric == Metric::Cosine) {
        train_norm.resize(train.n);
        test_norm.resize(test.n);
        row_norms(train, train_norm.data());   // same (scalar) norms as v1
        row_norms(test, test_norm.data());
    }

    const long n_test = static_cast<long>(test.n);
#pragma omp parallel for schedule(runtime) default(none) \
    shared(train, test, params, out, train_norm, test_norm, d, n_test)
    for (long i = 0; i < n_test; ++i) {
        const float* q = test.row(i);
        TopK top(params.k);
        for (std::size_t j = 0; j < train.n; ++j) {
            const float* t = train.row(j);
            const float dist = (params.metric == Metric::Euclidean)
                                   ? sq_euclidean_simd(q, t, d)
                                   : cosine_distance(dot_simd(q, t, d), test_norm[i], train_norm[j]);
            top.push(dist, train.y[j]);
        }
        out[i] = top.vote();
    }
}

// v3 - cache blocking (tiling) on top of v2's SIMD kernel.
// In v2 every query streams the whole training set (188 MB for MNIST) from
// DRAM, so the kernel is memory-bound. Here each thread takes a block of QB
// queries and walks the training set in tiles of TB rows: a tile is loaded
// from DRAM once and then reused QB times from L2, cutting DRAM traffic by ~QB.
// Every query still sees the training rows in increasing j, so TopK receives
// the distances in the same order as in v1/v2 (same tie-breaking).
// Tile sizes: KNN_TB / KNN_QB environment variables (defaults below).
static std::size_t env_size(const char* name, std::size_t def) {
    const char* s = std::getenv(name);
    const long v = s ? std::atol(s) : 0;
    return v > 0 ? static_cast<std::size_t>(v) : def;
}

void knn_omp_blocked(const Dataset& train, const Dataset& test,
                     const KnnParams& params, std::uint8_t* out) {
    const std::size_t d = train.d;
    // Cache budget per thread: tile 32 rows x 3136 B = 100 KB + query block
    // 16 x 3136 B = 50 KB = 150 KB, inside the 256 KB L2 of a core. With 2
    // hyperthreads per core (> 6 threads) it becomes 2 x 150 = 300 KB > 256 KB:
    // part of the working set (likely the queries, re-read once per tile) is
    // then served by L3. The tile sweep (run_benchmarks.py tiles) measures this.
    // The query row being processed (3 KB) stays in L1.
    const std::size_t TB = env_size("KNN_TB", 32);
    const std::size_t QB = env_size("KNN_QB", 16);

    std::vector<float> train_norm, test_norm;
    if (params.metric == Metric::Cosine) {
        train_norm.resize(train.n);
        test_norm.resize(test.n);
        row_norms(train, train_norm.data());   // same (scalar) norms as v1
        row_norms(test, test_norm.data());
    }

    const long n_qblocks = static_cast<long>((test.n + QB - 1) / QB);
#pragma omp parallel default(none) \
    shared(train, test, params, out, train_norm, test_norm, d, TB, QB, n_qblocks)
    {
        // One TopK per query of the block, private to the thread (declared
        // inside the parallel region). Allocated once per thread, not per block.
        std::vector<TopK> tops(QB, TopK(params.k));

#pragma omp for schedule(runtime)
        for (long b = 0; b < n_qblocks; ++b) {
            const std::size_t i0 = static_cast<std::size_t>(b) * QB;
            const std::size_t i1 = std::min(i0 + QB, test.n);   // last block may be partial
            for (std::size_t i = i0; i < i1; ++i) tops[i - i0] = TopK(params.k);

            for (std::size_t j0 = 0; j0 < train.n; j0 += TB) {
                const std::size_t j1 = std::min(j0 + TB, train.n);   // last tile may be partial
                for (std::size_t i = i0; i < i1; ++i) {   // tile j0..j1 reused (i1-i0) times
                    const float* q = test.row(i);
                    TopK& top = tops[i - i0];
                    for (std::size_t j = j0; j < j1; ++j) {
                        const float* t = train.row(j);
                        const float dist = (params.metric == Metric::Euclidean)
                                               ? sq_euclidean_simd(q, t, d)
                                               : cosine_distance(dot_simd(q, t, d), test_norm[i], train_norm[j]);
                        top.push(dist, train.y[j]);
                    }
                }
            }
            for (std::size_t i = i0; i < i1; ++i) out[i] = tops[i - i0].vote();
        }
    }
}

// v4 - SoA layout: training set transposed to feature-major (xt[f][j]).
// Same parallel structure as v2 (parallel for over queries, no blocking), so
// the only change vs omp_simd is the data layout (AoS vs SoA).
// AoS (v2) vectorizes over the 784 features of ONE training sample and needs a
// horizontal reduction at the end of every distance. SoA vectorizes over JB
// different training samples: lane jj accumulates the distance of sample j0+jj,
// features are added in increasing f exactly like the scalar reference (same
// summation order, no reduction clause), and the JB accumulators are
// independent -> no single dependency chain.
// The transposition is done inside the call (it is part of the cost of this
// layout, the input is row-major); KNN_VERBOSE=1 prints its time on stderr.
namespace {
struct AlignedBuffer {   // uninitialised 64-byte aligned floats (no serial zeroing)
    float* p;
    explicit AlignedBuffer(std::size_t n)
        : p(static_cast<float*>(::operator new(n * sizeof(float), std::align_val_t(64)))) {}
    ~AlignedBuffer() { ::operator delete(p, std::align_val_t(64)); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};
}  // namespace

void knn_omp_soa(const Dataset& train, const Dataset& test,
                 const KnnParams& params, std::uint8_t* out) {
    const std::size_t d = train.d, n = train.n;
    constexpr std::size_t JB = 64;   // samples per register block = 8 AVX2 vectors
    const std::size_t n_pad = (n + JB - 1) / JB * JB;   // rows of xt padded to JB

    const double t_start = omp_get_wtime();
    AlignedBuffer xt(d * n_pad);   // xt.p[f * n_pad + j] = feature f of sample j
    const long n_jblocks = static_cast<long>(n_pad / JB);
    // Blocked transpose: JB rows (200 KB) are read while they are in L2. Each
    // thread first-touches the pages it writes (parallel page faults).
#pragma omp parallel for schedule(static) default(none) shared(train, xt, d, n, n_pad, n_jblocks)
    for (long b = 0; b < n_jblocks; ++b) {
        const std::size_t j0 = static_cast<std::size_t>(b) * JB;
        for (std::size_t f = 0; f < d; ++f) {
            float* dst = xt.p + f * n_pad + j0;
            for (std::size_t jj = 0; jj < JB; ++jj)
                dst[jj] = (j0 + jj < n) ? train.x[(j0 + jj) * d + f] : 0.0f;   // padding = 0
        }
    }
    knn_prep_seconds = omp_get_wtime() - t_start;   // layout conversion, reported apart in the CSV
    if (std::getenv("KNN_VERBOSE"))
        std::fprintf(stderr, "omp_soa: transpose %.4f s\n", knn_prep_seconds);

    std::vector<float> train_norm, test_norm;
    if (params.metric == Metric::Cosine) {
        train_norm.resize(n);
        test_norm.resize(test.n);
        row_norms(train, train_norm.data());   // same (scalar) norms as v1
        row_norms(test, test_norm.data());
    }
    const bool euclid = params.metric == Metric::Euclidean;

    const long n_test = static_cast<long>(test.n);
#pragma omp parallel for schedule(runtime) default(none) \
    shared(train, test, params, out, train_norm, test_norm, xt, d, n, n_pad, euclid, n_test, JB)
    for (long i = 0; i < n_test; ++i) {
        const float* q = test.row(i);
        TopK top(params.k);
        for (std::size_t j0 = 0; j0 < n; j0 += JB) {
            alignas(64) float acc[JB] = {};   // one partial distance per training sample
            for (std::size_t f = 0; f < d; ++f) {
                const float qf = q[f];                      // broadcast to all lanes
                const float* col = xt.p + f * n_pad + j0;   // JB contiguous samples
                if (euclid) {
#pragma omp simd aligned(col : 64)
                    for (std::size_t jj = 0; jj < JB; ++jj) {
                        const float t = qf - col[jj];
                        acc[jj] += t * t;
                    }
                } else {
#pragma omp simd aligned(col : 64)
                    for (std::size_t jj = 0; jj < JB; ++jj) acc[jj] += qf * col[jj];
                }
            }
            const std::size_t jn = std::min(JB, n - j0);   // skip the padding samples
            for (std::size_t jj = 0; jj < jn; ++jj) {
                const std::size_t j = j0 + jj;
                const float dist = euclid ? acc[jj]
                                          : cosine_distance(acc[jj], test_norm[i], train_norm[j]);
                top.push(dist, train.y[j]);   // j increasing: same order as the reference
            }
        }
        out[i] = top.vote();
    }
}

// v4b - AoSoA ("blocked SoA"): same kernel as v4, different index only.
// Pure SoA puts feature f of all samples in one long row: for a block of JB
// samples the kernel touches 784 rows that are n_pad*4 B (240 KB) apart ->
// 784 concurrent streams, a new page at every step: the hardware prefetcher
// cannot follow them and the TLB misses. Measured: SoA wins 2.1x when the
// training fits in L2 but loses on full MNIST. AoSoA stores each block of JB
// samples as a contiguous [d][JB] matrix (784 x 64 floats = 200 KB): the
// kernel reads it sequentially and the block fits in L2.
void knn_omp_aosoa(const Dataset& train, const Dataset& test,
                   const KnnParams& params, std::uint8_t* out) {
    const std::size_t d = train.d, n = train.n;
    constexpr std::size_t JB = 64;
    const std::size_t n_pad = (n + JB - 1) / JB * JB;

    const double t_start = omp_get_wtime();
    AlignedBuffer xb(d * n_pad);   // xb.p[(b * d + f) * JB + jj] = feature f of sample b*JB+jj
    const long n_jblocks = static_cast<long>(n_pad / JB);
#pragma omp parallel for schedule(static) default(none) shared(train, xb, d, n, n_jblocks)
    for (long b = 0; b < n_jblocks; ++b) {
        const std::size_t j0 = static_cast<std::size_t>(b) * JB;
        float* blk = xb.p + static_cast<std::size_t>(b) * d * JB;
        for (std::size_t f = 0; f < d; ++f)
            for (std::size_t jj = 0; jj < JB; ++jj)
                blk[f * JB + jj] = (j0 + jj < n) ? train.x[(j0 + jj) * d + f] : 0.0f;
    }
    knn_prep_seconds = omp_get_wtime() - t_start;   // layout conversion, reported apart in the CSV
    if (std::getenv("KNN_VERBOSE"))
        std::fprintf(stderr, "omp_aosoa: transpose %.4f s\n", knn_prep_seconds);

    std::vector<float> train_norm, test_norm;
    if (params.metric == Metric::Cosine) {
        train_norm.resize(n);
        test_norm.resize(test.n);
        row_norms(train, train_norm.data());
        row_norms(test, test_norm.data());
    }
    const bool euclid = params.metric == Metric::Euclidean;

    const long n_test = static_cast<long>(test.n);
#pragma omp parallel for schedule(runtime) default(none) \
    shared(train, test, params, out, train_norm, test_norm, xb, d, n, euclid, n_test, JB)
    for (long i = 0; i < n_test; ++i) {
        const float* q = test.row(i);
        TopK top(params.k);
        for (std::size_t j0 = 0; j0 < n; j0 += JB) {
            const float* blk = xb.p + (j0 / JB) * d * JB;   // contiguous 200 KB block
            alignas(64) float acc[JB] = {};
            for (std::size_t f = 0; f < d; ++f) {
                const float qf = q[f];
                const float* col = blk + f * JB;   // next 256 B: sequential access
                if (euclid) {
#pragma omp simd aligned(col : 64)
                    for (std::size_t jj = 0; jj < JB; ++jj) {
                        const float t = qf - col[jj];
                        acc[jj] += t * t;
                    }
                } else {
#pragma omp simd aligned(col : 64)
                    for (std::size_t jj = 0; jj < JB; ++jj) acc[jj] += qf * col[jj];
                }
            }
            const std::size_t jn = std::min(JB, n - j0);
            for (std::size_t jj = 0; jj < jn; ++jj) {
                const std::size_t j = j0 + jj;
                const float dist = euclid ? acc[jj]
                                          : cosine_distance(acc[jj], test_norm[i], train_norm[j]);
                top.push(dist, train.y[j]);
            }
        }
        out[i] = top.vote();
    }
}

// v5 - register blocking on top of v3: a 4 queries x 1 training row micro-kernel.
// In v3 each distance is one `omp simd` reduction with ONE vector accumulator:
// the loop is bound by the FMA latency (~4 cycles) and loads 2 vectors per FMA.
// Here one pass over a training row computes the distances to 4 queries:
//   * the row is loaded once and reused 4 times from registers (data reuse);
//   * 4 independent accumulators -> 4 FMA chains in flight (latency hiding).
// Each s_k is the same reduction as in v2/v3, so the distances are the same.
// It is a small "matrix product" tile: the bridge to the GEMM formulation.
static inline void sq_euclidean_x4(const float* q0, const float* q1, const float* q2,
                                   const float* q3, const float* t, std::size_t d, float* out) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma omp simd reduction(+ : s0, s1, s2, s3)
    for (std::size_t f = 0; f < d; ++f) {
        const float tf = t[f];   // loaded once, used 4 times
        const float a = q0[f] - tf, b = q1[f] - tf, c = q2[f] - tf, e = q3[f] - tf;
        s0 += a * a; s1 += b * b; s2 += c * c; s3 += e * e;
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

static inline void dot_x4(const float* q0, const float* q1, const float* q2,
                          const float* q3, const float* t, std::size_t d, float* out) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma omp simd reduction(+ : s0, s1, s2, s3)
    for (std::size_t f = 0; f < d; ++f) {
        const float tf = t[f];
        s0 += q0[f] * tf; s1 += q1[f] * tf; s2 += q2[f] * tf; s3 += q3[f] * tf;
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

// Row norms with a parallel loop over the rows (used by omp_blocked4_pn only).
// Each row is still summed by ONE thread in increasing feature order, like
// row_norms() in knn_common.hpp: the only change is which thread does which row.
// schedule(static): every row costs the same, no need for dynamic balancing.
static void row_norms_parallel(const Dataset& ds, float* out) {
    const long n = static_cast<long>(ds.n);
#pragma omp parallel for schedule(static) default(none) shared(ds, out, n)
    for (long i = 0; i < n; ++i) {
        const float* r = ds.row(i);
        float s = 0.0f;
        for (std::size_t j = 0; j < ds.d; ++j) s += r[j] * r[j];
        out[i] = std::sqrt(s);
    }
}

// Body shared by omp_blocked4 and omp_blocked4_pn. `parallel_norms` is a
// compile-time constant in both callers and the function is always inlined, so
// omp_blocked4 compiles to the same code as before this helper existed.
[[gnu::always_inline]] static inline void blocked4_impl(const Dataset& train, const Dataset& test,
                                                        const KnnParams& params, std::uint8_t* out,
                                                        const bool parallel_norms) {
    const std::size_t d = train.d;
    const std::size_t TB = env_size("KNN_TB", 32);   // same tiles as omp_blocked
    const std::size_t QB = env_size("KNN_QB", 16);
    constexpr std::size_t R = 4;                      // queries per micro-kernel

    std::vector<float> train_norm, test_norm;
    if (params.metric == Metric::Cosine) {
        train_norm.resize(train.n);
        test_norm.resize(test.n);
        if (parallel_norms) {
            row_norms_parallel(train, train_norm.data());
            row_norms_parallel(test, test_norm.data());
        } else {
            row_norms(train, train_norm.data());
            row_norms(test, test_norm.data());
        }
    }
    const bool euclid = params.metric == Metric::Euclidean;
    auto to_dist = [&](float v, std::size_t i, std::size_t j) {
        return euclid ? v : cosine_distance(v, test_norm[i], train_norm[j]);
    };

    const long n_qblocks = static_cast<long>((test.n + QB - 1) / QB);
#pragma omp parallel default(none) \
    shared(train, test, params, out, d, TB, QB, n_qblocks, euclid, to_dist)
    {
        std::vector<TopK> tops(QB, TopK(params.k));

#pragma omp for schedule(runtime)
        for (long b = 0; b < n_qblocks; ++b) {
            const std::size_t i0 = static_cast<std::size_t>(b) * QB;
            const std::size_t i1 = std::min(i0 + QB, test.n);
            for (std::size_t i = i0; i < i1; ++i) tops[i - i0] = TopK(params.k);

            for (std::size_t j0 = 0; j0 < train.n; j0 += TB) {
                const std::size_t j1 = std::min(j0 + TB, train.n);
                std::size_t i = i0;
                for (; i + R <= i1; i += R) {   // groups of 4 queries
                    const float *q0 = test.row(i), *q1 = test.row(i + 1),
                                *q2 = test.row(i + 2), *q3 = test.row(i + 3);
                    for (std::size_t j = j0; j < j1; ++j) {
                        float s[R];
                        if (euclid) sq_euclidean_x4(q0, q1, q2, q3, train.row(j), d, s);
                        else        dot_x4(q0, q1, q2, q3, train.row(j), d, s);
                        for (std::size_t r = 0; r < R; ++r)
                            tops[i + r - i0].push(to_dist(s[r], i + r, j), train.y[j]);
                    }
                }
                for (; i < i1; ++i) {           // leftover queries (QB or n_test not multiple of 4)
                    const float* q = test.row(i);
                    for (std::size_t j = j0; j < j1; ++j) {
                        const float* t = train.row(j);
                        const float v = euclid ? sq_euclidean_simd(q, t, d) : dot_simd(q, t, d);
                        tops[i - i0].push(to_dist(v, i, j), train.y[j]);
                    }
                }
            }
            for (std::size_t i = i0; i < i1; ++i) out[i] = tops[i - i0].vote();
        }
    }
}

void knn_omp_blocked4(const Dataset& train, const Dataset& test,
                      const KnnParams& params, std::uint8_t* out) {
    blocked4_impl(train, test, params, out, /*parallel_norms=*/false);
}

// v5b - omp_blocked4 with the row norms of the cosine metric computed in
// parallel. One idea: remove the serial fraction (Amdahl). In every other
// variant row_norms() runs on one thread INSIDE the timed call: ~47 M dependent
// scalar adds for MNIST. It is invisible next to the 19 s of seq, but it is a
// fixed cost that does not shrink with the threads, so it weighs more and more
// as the parallel part gets faster. Identical to omp_blocked4 for euclidean.
void knn_omp_blocked4_pn(const Dataset& train, const Dataset& test,
                         const KnnParams& params, std::uint8_t* out) {
    blocked4_impl(train, test, params, out, /*parallel_norms=*/true);
}

// DIAGNOSTIC variant (not an optimisation): omp_simd where thread t starts
// t * KNN_STAGGER_MS milliseconds late (default 20 ms).
// Why: with schedule(static) all threads scan the SAME training rows at almost
// the same time, so one thread pulls a row from DRAM into the shared L3 and the
// others find it there ("convoy"): the apparent bandwidth exceeds the DRAM peak.
// A 20 ms head start is ~260 MB of training at 13 GB/s, far more than the 12 MB
// of L3: the threads can no longer share rows. If the run time jumps to the
// DRAM-bound value, the sharing is demonstrated by intervention, not inferred.
// The predictions are identical to omp_simd (same kernel, same order).
// NB: the delays are inside the timed call: (threads-1) * KNN_STAGGER_MS of the
// measured time is just waiting. Always static: with dynamic the late threads
// would simply get less work.
void knn_omp_simd_stagger(const Dataset& train, const Dataset& test,
                          const KnnParams& params, std::uint8_t* out) {
    const std::size_t d = train.d;
    const long stagger_ms = static_cast<long>(env_size("KNN_STAGGER_MS", 20));

    std::vector<float> train_norm, test_norm;
    if (params.metric == Metric::Cosine) {
        train_norm.resize(train.n);
        test_norm.resize(test.n);
        row_norms(train, train_norm.data());
        row_norms(test, test_norm.data());
    }

    const long n_test = static_cast<long>(test.n);
#pragma omp parallel default(none) \
    shared(train, test, params, out, train_norm, test_norm, d, n_test, stagger_ms)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(stagger_ms * omp_get_thread_num()));
#pragma omp for schedule(static)
        for (long i = 0; i < n_test; ++i) {
            const float* q = test.row(i);
            TopK top(params.k);
            for (std::size_t j = 0; j < train.n; ++j) {
                const float* t = train.row(j);
                const float dist = (params.metric == Metric::Euclidean)
                                       ? sq_euclidean_simd(q, t, d)
                                       : cosine_distance(dot_simd(q, t, d), test_norm[i], train_norm[j]);
                top.push(dist, train.y[j]);
            }
            out[i] = top.vote();
        }
    }
}

// ---------------------------------------------------------------------------
// Next variants (see the roadmap in CLAUDE.md). Suggested names:
//   knn_omp_gemm      - |a|^2 + |b|^2 - 2 a.b with a blocked matrix product
// ---------------------------------------------------------------------------
