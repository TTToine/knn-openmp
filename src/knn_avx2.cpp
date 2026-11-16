// Variants written with AVX2/FMA INTRINSICS (this file is compiled with
// -mavx2 -mfma, see CMakeLists.txt). Everything else in the project lets the
// compiler vectorize (`#pragma omp simd`); here the vector instructions are
// chosen by hand, which makes three things possible:
//   1. omp_blocked_avx2    - the same kernel as omp_blocked, by hand: does the
//                            compiler do as well as intrinsics?        (1 idea)
//   2. omp_blocked_avx2u4  - 4 accumulators on ONE distance: breaks the FMA
//                            dependency chain without touching the data flow;
//   3. omp_mkMRxNR         - register-blocked micro-kernel, MR queries x NR
//                            training rows per pass (generalises omp_blocked4);
//   4. omp_blocked_ea      - exact early abandoning of the euclidean distance.
//
// All the kernels in this file compute the distance of a (query, row) pair
// with the SAME sequence of floating-point operations (one 8-lane accumulator
// updated with FMA in increasing feature order, the fixed-order horizontal sum
// hsum256, a scalar FMA tail) - except avx2u4, which uses 4 accumulators by
// design. So a pair gets bit-identical distances from every MRxNR kernel and
// from the early-abandoning kernel, whatever the tile sizes and remainders.
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "knn.hpp"
#include "knn_common.hpp"

namespace {

std::size_t env_size(const char* name, std::size_t def) {
    const char* s = std::getenv(name);
    const long v = s ? std::atol(s) : 0;
    return v > 0 ? static_cast<std::size_t>(v) : def;
}

// Horizontal sum of the 8 lanes in a FIXED order:
// ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7)). Every step adds two values, so if all
// lanes are >= 0 the result is monotone in each lane (needed by omp_blocked_ea).
inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));   // 4 sums
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));                                           // 2 sums
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));                                       // 1 sum
    return _mm_cvtss_f32(s);
}

// Features beyond the last multiple of 8, added one by one with an explicit
// FMA (std::fma, never left to the compiler's contraction choices).
template <bool EUCLID>
inline float scalar_tail(float s, const float* a, const float* b, std::size_t from, std::size_t d) {
    for (std::size_t f = from; f < d; ++f) {
        if (EUCLID) { const float e = a[f] - b[f]; s = std::fma(e, e, s); }
        else        s = std::fma(a[f], b[f], s);
    }
    return s;
}

// ---------------------------------------------------------------------------
// MR x NR micro-kernel. One pass over the features computes MR*NR values
// (squared distances if EUCLID, dot products otherwise):
//   out[m * NR + n] = f(q[m], t[n]).
// Per 8 features: NR loads of training vectors, MR loads of query vectors,
// MR*NR (vsubps +) vfmadd231ps on MR*NR independent accumulators.
//   * every loaded training vector is reused MR times, every query vector NR
//     times: (MR + NR) loads feed MR*NR FMAs (4x1: 1.25 loads per FMA,
//     4x2: 0.75, 6x2: 0.67);
//   * MR*NR independent dependency chains hide the 4-cycle FMA latency.
// Register budget (16 ymm): MR*NR accumulators + NR training vectors + 1-2
// temporaries. 4x2 = 8 + 2 + 2 = 12; 6x2 = 12 + 2 + 2 = 16: the limit.
// MR and NR are template parameters: the loops over m and n have constant
// bounds, the compiler unrolls them and keeps acc[][] in registers (checked
// in the assembly, see report/notes.md).
// ---------------------------------------------------------------------------
template <int MR, int NR, bool EUCLID>
[[gnu::always_inline]] inline void kernel_mrnr(const float* const* q, const float* const* t,
                                               std::size_t d, float* out) {
    __m256 acc[MR][NR];
    for (int m = 0; m < MR; ++m)
        for (int n = 0; n < NR; ++n) acc[m][n] = _mm256_setzero_ps();
    const std::size_t dv = d & ~static_cast<std::size_t>(7);
    for (std::size_t f = 0; f < dv; f += 8) {
        __m256 tv[NR];
        for (int n = 0; n < NR; ++n) tv[n] = _mm256_loadu_ps(t[n] + f);   // loaded once, used MR times
        for (int m = 0; m < MR; ++m) {
            const __m256 qv = _mm256_loadu_ps(q[m] + f);                  // loaded once, used NR times
            for (int n = 0; n < NR; ++n) {
                if (EUCLID) {
                    const __m256 e = _mm256_sub_ps(qv, tv[n]);
                    acc[m][n] = _mm256_fmadd_ps(e, e, acc[m][n]);         // acc += e * e
                } else {
                    acc[m][n] = _mm256_fmadd_ps(qv, tv[n], acc[m][n]);    // acc += q * t
                }
            }
        }
    }
    for (int m = 0; m < MR; ++m)
        for (int n = 0; n < NR; ++n)
            out[m * NR + n] = scalar_tail<EUCLID>(hsum256(acc[m][n]), q[m], t[n], dv, d);
}

// The M queries i .. i+M-1 against the tile rows j0 .. j1-1, NR rows at a
// time (then one at a time for the last j1-j0 mod NR rows). Each query still
// receives the rows in increasing j: same order of TopK pushes as everywhere.
template <int M, int NR, bool EUCLID, class ToDist>
[[gnu::always_inline]] inline void tile_group(const Dataset& train, const Dataset& test, std::size_t i,
                                              std::size_t j0, std::size_t j1, TopK* tops,
                                              const ToDist& to_dist) {
    const std::size_t d = train.d;
    const float* q[M];
    for (int m = 0; m < M; ++m) q[m] = test.row(i + m);
    std::size_t j = j0;
    for (; j + NR <= j1; j += NR) {
        const float* t[NR];
        for (int n = 0; n < NR; ++n) t[n] = train.row(j + n);
        float s[M * NR];
        kernel_mrnr<M, NR, EUCLID>(q, t, d, s);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < NR; ++n)
                tops[m].push(to_dist(s[m * NR + n], i + m, j + n), train.y[j + n]);
    }
    for (; j < j1; ++j) {   // tile size not a multiple of NR
        const float* t[1] = {train.row(j)};
        float s[M];
        kernel_mrnr<M, 1, EUCLID>(q, t, d, s);
        for (int m = 0; m < M; ++m) tops[m].push(to_dist(s[m], i + m, j), train.y[j]);
    }
}

// Cache-blocked driver (same tiling as omp_blocked: blocks of QB queries x
// tiles of TB training rows, KNN_QB / KNN_TB) around the MR x NR micro-kernel.
// Queries left over when the block is not a multiple of MR go through the
// 2 x NR and then the 1 x NR kernel (QB = 16, MR = 6: 6 + 6 + 2 + 2).
template <int MR, int NR>
[[gnu::always_inline]] inline void blocked_mrnr(const Dataset& train, const Dataset& test,
                                                const KnnParams& params, std::uint8_t* out) {
    const std::size_t TB = env_size("KNN_TB", 32);
    const std::size_t QB = env_size("KNN_QB", 16);

    std::vector<float> train_norm, test_norm;
    if (params.metric == Metric::Cosine) {
        train_norm.resize(train.n);
        test_norm.resize(test.n);
        row_norms(train, train_norm.data());   // serial, like omp_blocked (see omp_blocked4_pn)
        row_norms(test, test_norm.data());
    }
    const bool euclid = params.metric == Metric::Euclidean;
    auto to_dist_e = [](float v, std::size_t, std::size_t) { return v; };
    auto to_dist_c = [&](float v, std::size_t i, std::size_t j) {
        return cosine_distance(v, test_norm[i], train_norm[j]);
    };

    const long n_qblocks = static_cast<long>((test.n + QB - 1) / QB);
    // Shared and read-only: train, test, params, norms (through the lambdas),
    // sizes. out[]: each element written by one thread. tops: private.
#pragma omp parallel default(none) \
    shared(train, test, params, out, TB, QB, n_qblocks, euclid, to_dist_e, to_dist_c)
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
                for (; i + MR <= i1; i += MR) {
                    if (euclid) tile_group<MR, NR, true>(train, test, i, j0, j1, &tops[i - i0], to_dist_e);
                    else        tile_group<MR, NR, false>(train, test, i, j0, j1, &tops[i - i0], to_dist_c);
                }
                if (MR > 2)
                    for (; i + 2 <= i1; i += 2) {
                        if (euclid) tile_group<2, NR, true>(train, test, i, j0, j1, &tops[i - i0], to_dist_e);
                        else        tile_group<2, NR, false>(train, test, i, j0, j1, &tops[i - i0], to_dist_c);
                    }
                for (; i < i1; ++i) {
                    if (euclid) tile_group<1, NR, true>(train, test, i, j0, j1, &tops[i - i0], to_dist_e);
                    else        tile_group<1, NR, false>(train, test, i, j0, j1, &tops[i - i0], to_dist_c);
                }
            }
            for (std::size_t i = i0; i < i1; ++i) out[i] = tops[i - i0].vote();
        }
    }
}

// ---------------------------------------------------------------------------
// 1 x 1 kernel with 4 accumulators on the SAME distance (features unrolled by
// 32): 4 independent FMA chains, so 4 FMAs are in flight instead of 1. This is
// what `omp simd reduction` cannot do by itself: it gives one vector
// accumulator (8 lanes) and the loop stays bound by the FMA latency.
// Different summation order from the 1-accumulator kernels (32 partial sums).
// ---------------------------------------------------------------------------
template <bool EUCLID>
[[gnu::always_inline]] inline float pair_u4(const float* a, const float* b, std::size_t d) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    auto step = [&](__m256& acc, std::size_t f) {
        const __m256 x = _mm256_loadu_ps(a + f), y = _mm256_loadu_ps(b + f);
        if (EUCLID) { const __m256 e = _mm256_sub_ps(x, y); acc = _mm256_fmadd_ps(e, e, acc); }
        else        acc = _mm256_fmadd_ps(x, y, acc);
    };
    std::size_t f = 0;
    for (; f + 32 <= d; f += 32) { step(a0, f); step(a1, f + 8); step(a2, f + 16); step(a3, f + 24); }
    for (; f + 8 <= d; f += 8) step(a0, f);
    const __m256 sum = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    return scalar_tail<EUCLID>(hsum256(sum), a, b, f, d);
}

// ---------------------------------------------------------------------------
// Early abandoning (euclidean only). Same accumulation as kernel_mrnr<1,1>,
// but every `step` features the partial sum is compared with `bound`, the
// distance of the current k-th neighbour of this query: if partial >= bound
// the candidate cannot enter the top-k and the rest of the row is skipped.
//
// EXACT, not approximate. Claim: if the kernel abandons, TopK would have
// rejected the full distance D (TopK rejects when !(D < bound)).
//   1. Every lane of `acc` only grows: acc' = fl(e*e + acc), with e*e >= 0
//      computed exactly inside the FMA, and rounding is monotone
//      (x >= y implies fl(x) >= fl(y)), so fl(e*e + acc) >= fl(acc) = acc.
//   2. hsum256 adds lanes that are all >= 0 in a fixed order; each of its
//      additions is monotone in both operands, so hsum256 is monotone in every
//      lane: hsum(acc now) <= hsum(acc at the end).
//   3. The scalar tail adds terms >= 0 with the same FMA: again monotone.
//   Hence partial <= D, and partial >= bound implies D >= bound, i.e. !(D < bound).
// The partial sums of squares are monotone in FLOATING POINT too, not only in
// exact arithmetic: that is the whole proof. (It would fail for a dot product,
// whose terms can be negative: the cosine metric is not abandoned.)
// When the kernel does not abandon, it returns exactly the float that
// kernel_mrnr<1,1> returns: omp_blocked_ea == omp_blocked_avx2 on every push
// that matters.
// ---------------------------------------------------------------------------
[[gnu::always_inline]] inline bool pair_ea(const float* a, const float* b, std::size_t d,
                                           std::size_t step, float bound, float& dist,
                                           std::size_t& vectors_done) {
    __m256 acc = _mm256_setzero_ps();
    const std::size_t dv = d & ~static_cast<std::size_t>(7);
    std::size_t f = 0;
    while (f < dv) {
        const std::size_t stop = std::min(f + step, dv);
        for (; f < stop; f += 8) {
            const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(a + f), _mm256_loadu_ps(b + f));
            acc = _mm256_fmadd_ps(e, e, acc);
        }
        // One data-dependent branch per check: not taken until the partial sum
        // crosses the bound, then taken once (position unpredictable).
        if (hsum256(acc) >= bound) { vectors_done += f / 8; return false; }
    }
    vectors_done += dv / 8;
    dist = scalar_tail<true>(hsum256(acc), a, b, dv, d);
    return true;
}

}  // namespace

// Fraction of the distance work actually done by the last omp_blocked_ea call
// (features evaluated / features of all pairs); 1 for the cosine metric.
double knn_ea_fraction = 1.0;

// ---- registered variants ----------------------------------------------------

// omp_blocked with the distance kernel written by hand: 1 x 1, one accumulator.
// One idea: WHO vectorizes (the programmer instead of `omp simd`).
void knn_omp_blocked_avx2(const Dataset& train, const Dataset& test,
                          const KnnParams& params, std::uint8_t* out) {
    blocked_mrnr<1, 1>(train, test, params, out);
}

// Register-blocked micro-kernels. 4x1 is the hand-written twin of omp_blocked4
// (control: must perform like it); the others add reuse of the query vectors.
void knn_omp_mk4x1(const Dataset& train, const Dataset& test, const KnnParams& params, std::uint8_t* out) {
    blocked_mrnr<4, 1>(train, test, params, out);
}
void knn_omp_mk4x2(const Dataset& train, const Dataset& test, const KnnParams& params, std::uint8_t* out) {
    blocked_mrnr<4, 2>(train, test, params, out);
}
void knn_omp_mk6x2(const Dataset& train, const Dataset& test, const KnnParams& params, std::uint8_t* out) {
    blocked_mrnr<6, 2>(train, test, params, out);
}
void knn_omp_mk2x4(const Dataset& train, const Dataset& test, const KnnParams& params, std::uint8_t* out) {
    blocked_mrnr<2, 4>(train, test, params, out);
}

// Shared body of the two 1 x 1 variants below (same tiling as omp_blocked).
// EA = false: 4-accumulator kernel. EA = true: early abandoning (euclidean).
template <bool EA>
[[gnu::always_inline]] static inline void blocked_pair(const Dataset& train, const Dataset& test,
                                                       const KnnParams& params, std::uint8_t* out) {
    const std::size_t d = train.d;
    const std::size_t TB = env_size("KNN_TB", 32);
    const std::size_t QB = env_size("KNN_QB", 16);
    // Features between two checks of the bound (multiple of 8, the vector width).
    const std::size_t ea_step = std::max<std::size_t>(8, env_size("KNN_EA_STEP", 64) / 8 * 8);

    std::vector<float> train_norm, test_norm;
    if (params.metric == Metric::Cosine) {
        train_norm.resize(train.n);
        test_norm.resize(test.n);
        row_norms(train, train_norm.data());
        row_norms(test, test_norm.data());
    }
    const bool euclid = params.metric == Metric::Euclidean;
    const float inf = std::numeric_limits<float>::infinity();

    const long n_qblocks = static_cast<long>((test.n + QB - 1) / QB);
    unsigned long long vectors = 0;   // 8-feature steps executed (statistics of the EA variant)
#pragma omp parallel default(none) reduction(+ : vectors) \
    shared(train, test, params, out, train_norm, test_norm, d, TB, QB, ea_step, euclid, inf, n_qblocks)
    {
        std::vector<TopK> tops(QB, TopK(params.k));
        std::size_t done = 0;   // private counter, added to the reduction once per thread

#pragma omp for schedule(runtime)
        for (long b = 0; b < n_qblocks; ++b) {
            const std::size_t i0 = static_cast<std::size_t>(b) * QB;
            const std::size_t i1 = std::min(i0 + QB, test.n);
            for (std::size_t i = i0; i < i1; ++i) tops[i - i0] = TopK(params.k);

            for (std::size_t j0 = 0; j0 < train.n; j0 += TB) {
                const std::size_t j1 = std::min(j0 + TB, train.n);
                for (std::size_t i = i0; i < i1; ++i) {
                    const float* q = test.row(i);
                    TopK& top = tops[i - i0];
                    for (std::size_t j = j0; j < j1; ++j) {
                        const float* t = train.row(j);
                        if (!euclid) {
                            float s[1];
                            const float* qq[1] = {q};
                            const float* tt[1] = {t};
                            if (EA) kernel_mrnr<1, 1, false>(qq, tt, d, s);
                            else    s[0] = pair_u4<false>(q, t, d);
                            top.push(cosine_distance(s[0], test_norm[i], train_norm[j]), train.y[j]);
                        } else if (EA) {
                            // bound = distance of the current k-th neighbour (+inf until k are found)
                            const float bound = top.size == top.k ? top.dist[top.k - 1] : inf;
                            float dist;
                            if (pair_ea(q, t, d, ea_step, bound, dist, done)) top.push(dist, train.y[j]);
                        } else {
                            top.push(pair_u4<true>(q, t, d), train.y[j]);
                        }
                    }
                }
            }
            for (std::size_t i = i0; i < i1; ++i) out[i] = tops[i - i0].vote();
        }
        vectors += done;
    }
    if (EA) {
        const double total = static_cast<double>(train.n) * static_cast<double>(test.n) * static_cast<double>(d / 8);
        knn_ea_fraction = (euclid && total > 0) ? static_cast<double>(vectors) / total : 1.0;
        if (std::getenv("KNN_VERBOSE"))
            std::fprintf(stderr, "omp_blocked_ea: step=%zu features, evaluated %.4f of the features\n",
                         ea_step, knn_ea_fraction);
    }
}

// omp_blocked_avx2 with 4 accumulators per distance (latency hiding only).
void knn_omp_blocked_avx2u4(const Dataset& train, const Dataset& test,
                            const KnnParams& params, std::uint8_t* out) {
    blocked_pair<false>(train, test, params, out);
}

// omp_blocked_avx2 + exact early abandoning of the euclidean distance
// (check every KNN_EA_STEP features, default 64). Cosine: plain 1x1 kernel.
void knn_omp_blocked_ea(const Dataset& train, const Dataset& test,
                        const KnnParams& params, std::uint8_t* out) {
    blocked_pair<true>(train, test, params, out);
}
