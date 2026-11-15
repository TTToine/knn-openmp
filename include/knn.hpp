#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "dataset.hpp"

enum class Metric { Euclidean, Cosine };

struct KnnParams {
    int k = 5;
    Metric metric = Metric::Euclidean;
};

constexpr int kMaxK = 128;       // upper bound on k (fixed-size top-k buffers)
constexpr int kMaxClasses = 256;

// Every implementation has this signature: classify each test sample and
// write its predicted label into out[i]. Only this call is timed.
using KnnFn = void (*)(const Dataset& train, const Dataset& test,
                       const KnnParams& params, std::uint8_t* out);

// ---- Implementations ----
// Sequential reference. Correctness oracle and baseline for speedup.
void knn_sequential(const Dataset& train, const Dataset& test,
                    const KnnParams& params, std::uint8_t* out);

// First parallel version: parallel loop over test samples, same kernel.
// Uses schedule(runtime) so the schedule is chosen via OMP_SCHEDULE.
void knn_omp_naive(const Dataset& train, const Dataset& test,
                   const KnnParams& params, std::uint8_t* out);

// Same as omp_naive, plus `#pragma omp simd reduction` on the distance loop.
// Reorders the float sums: predictions may differ on near-exact ties.
void knn_omp_simd(const Dataset& train, const Dataset& test,
                  const KnnParams& params, std::uint8_t* out);

// omp_simd + cache blocking: blocks of QB queries x tiles of TB training rows,
// so each tile is reused from L2. Tile sizes from KNN_QB / KNN_TB env vars.
void knn_omp_blocked(const Dataset& train, const Dataset& test,
                     const KnnParams& params, std::uint8_t* out);

// ---- Variant registry (see src/variants.cpp) ----
struct Variant {
    std::string name;
    std::string description;
    KnnFn fn;
    bool parallel;
};

const std::vector<Variant>& all_variants();
const Variant* find_variant(const std::string& name);

std::string metric_name(Metric m);
bool parse_metric(const std::string& s, Metric& out);
