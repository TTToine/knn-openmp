// OpenMP variants. Each function is ONE optimisation idea, so that the report
// can compare them one against the other. Register new variants in variants.cpp.
#include <omp.h>

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

// ---------------------------------------------------------------------------
// Next variants (see the roadmap in CLAUDE.md). Suggested names:
//   knn_omp_simd      - #pragma omp simd reduction on the distance loop
//   knn_omp_blocked   - tile test x train to reuse train rows from cache
//   knn_omp_gemm      - |a|^2 + |b|^2 - 2 a.b with a blocked matrix product
// ---------------------------------------------------------------------------
