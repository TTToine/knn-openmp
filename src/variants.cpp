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
