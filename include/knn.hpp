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

// omp_simd with the training set transposed to feature-major (SoA): vectorizes
// over 64 training samples instead of over the features. Transpose is timed.
void knn_omp_soa(const Dataset& train, const Dataset& test,
                 const KnnParams& params, std::uint8_t* out);

// Same as omp_soa but the transposed training set is stored in contiguous
// blocks of 64 samples ([d][64] each, "AoSoA"): sequential, prefetch-friendly.
void knn_omp_aosoa(const Dataset& train, const Dataset& test,
                   const KnnParams& params, std::uint8_t* out);

// omp_blocked + 4 queries x 1 training row micro-kernel (register blocking).
void knn_omp_blocked4(const Dataset& train, const Dataset& test,
                      const KnnParams& params, std::uint8_t* out);

// omp_blocked4 with the cosine row norms computed in parallel (Amdahl example).
void knn_omp_blocked4_pn(const Dataset& train, const Dataset& test,
                         const KnnParams& params, std::uint8_t* out);

// DIAGNOSTIC: omp_simd where thread t starts t * KNN_STAGGER_MS ms late, to
// break the sharing of training rows in L3 between synchronous threads.
void knn_omp_simd_stagger(const Dataset& train, const Dataset& test,
                          const KnnParams& params, std::uint8_t* out);

// Seconds spent by the LAST call in one-off data preparation that is part of
// the timed call but not of the k-NN kernel (layout conversion of omp_soa /
// omp_aosoa: allocation + transposition). 0 for the other variants. The driver
// resets it before every call and writes it in the `prep_s` CSV column, so the
// report can show kernel and conversion separately. Written by the calling
// thread only, outside any parallel region.
inline double knn_prep_seconds = 0.0;

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
