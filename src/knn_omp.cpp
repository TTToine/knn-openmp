// OpenMP variants. Each function is ONE optimisation idea, so that the report
// can compare them one against the other. Register new variants in variants.cpp.
#include <omp.h>

#include <algorithm>
#include <cstdlib>
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
        row_norms(train, train_norm.data());   // TODO: parallelise too (small cost)
        row_norms(test, test_norm.data());
    }

    const long n_test = static_cast<long>(test.n);
#pragma omp parallel for schedule(runtime)
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
#pragma omp parallel for schedule(runtime)
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
    // 32 rows x 3136 B = 100 KB: fits in the 256 KB L2 even when the two
    // hyperthreads of a core share it. One query row (3 KB) stays in L1.
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
#pragma omp parallel
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

// ---------------------------------------------------------------------------
// Next variants (see the roadmap in CLAUDE.md). Suggested names:
//   knn_omp_gemm      - |a|^2 + |b|^2 - 2 a.b with a blocked matrix product
// ---------------------------------------------------------------------------
