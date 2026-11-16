#include "knn.hpp"

// Registry of all implementations. To add a variant:
//   1. implement it in knn_omp.cpp (declare it in knn.hpp)
//   2. add one line below
//   3. run `ctest` (the smoke test validates every registered variant)
const std::vector<Variant>& all_variants() {
    static const std::vector<Variant> v = {
        {"seq", "sequential reference (oracle + baseline)", knn_sequential, false},
        {"omp_naive", "parallel for over test samples, schedule(runtime)", knn_omp_naive, true},
        {"omp_simd", "omp_naive + omp simd reduction on the distance loop", knn_omp_simd, true},
        {"omp_blocked", "omp_simd + query-block x train-tile cache blocking", knn_omp_blocked, true},
        {"omp_soa", "omp_simd with feature-major (SoA) training set", knn_omp_soa, true},
        {"omp_aosoa", "omp_soa with the training set in contiguous 64-sample blocks", knn_omp_aosoa, true},
        {"omp_blocked4", "omp_blocked + 4 queries x 1 row micro-kernel", knn_omp_blocked4, true},
        {"omp_blocked4_pn", "omp_blocked4 + parallel row norms (cosine)", knn_omp_blocked4_pn, true},
        {"omp_blocked4_heap", "omp_blocked4 + top-k in a binary max-heap", knn_omp_blocked4_heap, true},
        {"omp_blocked_avx2", "omp_blocked with a hand-written AVX2 kernel (1 accumulator)", knn_omp_blocked_avx2, true},
        {"omp_blocked_avx2u4", "omp_blocked_avx2 with 4 accumulators per distance", knn_omp_blocked_avx2u4, true},
        {"omp_mk4x1", "AVX2 micro-kernel 4 queries x 1 row (twin of omp_blocked4)", knn_omp_mk4x1, true},
        {"omp_mk4x2", "AVX2 micro-kernel 4 queries x 2 rows (8 accumulators)", knn_omp_mk4x2, true},
        {"omp_mk6x2", "AVX2 micro-kernel 6 queries x 2 rows (12 accumulators)", knn_omp_mk6x2, true},
        {"omp_mk2x4", "AVX2 micro-kernel 2 queries x 4 rows (8 accumulators)", knn_omp_mk2x4, true},
        {"omp_blocked_ea", "omp_blocked_avx2 + exact early abandoning (euclidean)", knn_omp_blocked_ea, true},
        {"omp_simd_stagger", "DIAGNOSTIC: omp_simd with staggered thread start (always static)", knn_omp_simd_stagger, true},
    };
    return v;
}

const Variant* find_variant(const std::string& name) {
    for (const auto& v : all_variants())
        if (v.name == name) return &v;
    return nullptr;
}

std::string metric_name(Metric m) { return m == Metric::Euclidean ? "euclidean" : "cosine"; }

bool parse_metric(const std::string& s, Metric& out) {
    if (s == "euclidean" || s == "l2") { out = Metric::Euclidean; return true; }
    if (s == "cosine") { out = Metric::Cosine; return true; }
    return false;
}
