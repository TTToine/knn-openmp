#pragma once
// Building blocks shared by all k-NN variants. Header-only so the compiler can
// inline them into each variant's hot loop.
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "knn.hpp"

// Plain scalar loops: summation order is fixed, so without -ffast-math GCC will
// NOT vectorize these reductions. Variants that use `omp simd reduction` change
// the summation order and can produce tiny floating-point differences.
inline float sq_euclidean(const float* a, const float* b, std::size_t d) {
    float s = 0.0f;
    for (std::size_t i = 0; i < d; ++i) {
        const float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

inline float dot(const float* a, const float* b, std::size_t d) {
    float s = 0.0f;
    for (std::size_t i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
}

// For cosine: 1 - <a,b> / (|a||b|), norms precomputed by the caller.
inline float cosine_distance(float dot_ab, float norm_a, float norm_b) {
    const float denom = norm_a * norm_b;
    return denom > 0.0f ? 1.0f - dot_ab / denom : 1.0f;
}

// Keeps the k smallest distances seen so far, sorted ascending.
// Insertion sort is a good fit for the small k used here (1, 5, 10).
// Ties: strict '<' keeps the earlier training index -> deterministic result.
struct TopK {
    int k;
    int size = 0;
    float dist[kMaxK];
    std::uint8_t label[kMaxK];

    explicit TopK(int k_) : k(k_) {}

    inline void push(float d, std::uint8_t lab) {
        if (size == k && !(d < dist[k - 1])) return;
        int pos = (size < k) ? size++ : k - 1;
        while (pos > 0 && d < dist[pos - 1]) {
            dist[pos] = dist[pos - 1];
            label[pos] = label[pos - 1];
            --pos;
        }
        dist[pos] = d;
        label[pos] = lab;
    }

    // Majority vote; ties broken in favour of the class of the nearest neighbour.
    inline std::uint8_t vote() const {
        int counts[kMaxClasses] = {0};
        int best = 0;
        for (int j = 0; j < size; ++j) {
            const int c = ++counts[label[j]];
            if (c > best) best = c;
        }
        for (int j = 0; j < size; ++j)
            if (counts[label[j]] == best) return label[j];
        return 0;
    }
};

// Euclidean norms of every row (used by the cosine metric).
inline void row_norms(const Dataset& ds, float* out) {
    for (std::size_t i = 0; i < ds.n; ++i) {
        const float* r = ds.row(i);
        float s = 0.0f;
        for (std::size_t j = 0; j < ds.d; ++j) s += r[j] * r[j];
        out[i] = std::sqrt(s);
    }
}
