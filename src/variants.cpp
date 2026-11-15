#include "knn.hpp"

// Registry of all implementations. To add a variant:
//   1. implement it in knn_omp.cpp (declare it in knn.hpp)
//   2. add one line below
//   3. run `ctest` (the smoke test validates every registered variant)
const std::vector<Variant>& all_variants() {
    static const std::vector<Variant> v = {
        {"seq", "sequential reference (oracle + baseline)", knn_sequential, false},
        {"omp_naive", "parallel for over test samples, schedule(runtime)", knn_omp_naive, true},
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
