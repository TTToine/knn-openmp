// Sequential reference implementation.
//
// DO NOT optimise this file: it is the correctness oracle for every parallel
// variant and the baseline T_sequential used for speedup. It is compiled with
// the same flags as the other variants, so the comparison is fair.
#include <vector>

#include "knn.hpp"
#include "knn_common.hpp"

void knn_sequential(const Dataset& train, const Dataset& test,
                    const KnnParams& params, std::uint8_t* out) {
    const std::size_t d = train.d;

    std::vector<float> train_norm, test_norm;
    if (params.metric == Metric::Cosine) {
        train_norm.resize(train.n);
        test_norm.resize(test.n);
        row_norms(train, train_norm.data());
        row_norms(test, test_norm.data());
    }

    for (std::size_t i = 0; i < test.n; ++i) {
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
