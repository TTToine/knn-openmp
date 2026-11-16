#!/usr/bin/env bash
# Experiments added on 2026-11-16 (written and validated on battery, never
# timed). Run AFTER run_rest.sh and run_extra.sh, same conditions: laptop
# plugged in, "High performance" plan, nothing else running, no builds.
#   bash scripts/run_new.sh 2>&1 | tee results/raw/run_new_$(date +%Y%m%d_%H%M%S).log
# Each part states the prediction it tests (report/notes.md, 2026-11-16).
set -euo pipefail
cd "$(dirname "$0")/.."
B="python3 scripts/run_benchmarks.py"
echo "### commit $(git rev-parse --short HEAD) $(git diff --quiet || echo "(dirty)")"

# N1. top-k policy: insertion sort vs heap, k = 1..128 (~6 min, most of it seq)
#     prediction: no difference within noise for every k (T1)
$B topk --variants omp_blocked4,omp_blocked4_heap --threads 6

# N2. weak scaling, queries grow with the threads (~4 min)
#     prediction: omp_blocked4 weak efficiency >= 0.9 up to 4 threads, then DRAM-limited (W1)
$B weak --variants omp_simd,omp_blocked,omp_blocked4
#     same with the training set growing instead (memory footprint grows: p * 188 MB) (~6 min)
$B weak --variants omp_simd,omp_blocked4 --weak-mode train --n-test 96

# N3. synchronisation strategies on the confusion matrix (~1 min)
#     predictions S1-S4
bash scripts/sync_strategies.sh
# N4. kernels with the training set in L2 (64 rows), 1 and 6 threads, both metrics (~4 min)
#     predictions A (intrinsics = omp simd), B (4 accumulators), C (MRxNR euclidean: no gain),
#     D (MRxNR cosine: 4x2 up to 2x faster than 4x1; 6x2 slower than 4x2 because of the spill)
K="omp_blocked,omp_blocked_avx2,omp_blocked_avx2u4,omp_blocked4,omp_mk4x1,omp_mk4x2,omp_mk6x2,omp_mk2x4"
$B threads --variants $K --thread-list 1,6 --n-test 9984 --synthetic --n-train 64 --no-seq --tag l2_euclid
$B threads --variants $K --thread-list 1,6 --n-test 9984 --synthetic --n-train 64 --no-seq --tag l2_cosine --metric cosine

# N5. the same kernels on full MNIST: strong scaling (~5 min)
#     prediction: at >= 4 threads all the 4-accumulator-or-more kernels meet the same DRAM limit
$B threads --variants $K
$B threads --variants omp_blocked4,omp_mk4x1,omp_mk4x2,omp_mk6x2,omp_mk2x4 --thread-list 1,6,12 --metric cosine --tag cosine

# N6. does more reuse (QB = 32) let the 8-accumulator kernel run? tile sweep at 6 threads (~2 min)
$B tiles --variants omp_mk4x1,omp_mk4x2 --threads 6

# N7. early abandoning: check-interval sweep at 1 and 6 threads + strong scaling (~4 min)
#     prediction E: 41% of the features evaluated (counter, already measured);
#     1.7-2.2x faster than omp_blocked_avx2 at 1 thread; best step 32-64; step 16 slower than 64
$B eastep --threads 6
$B threads --variants omp_blocked_avx2,omp_blocked_ea,omp_blocked4 --tag ea
# N8. matrix formulation |q|^2+|t|^2-2q.t (~4 min)
#     prediction F: in L2 omp_gemm up to 2x omp_mk4x1 (1 FP uop per element instead of 2);
#     omp_gemm_exact <= 10% slower than omp_gemm; on MNIST from 4 threads: same DRAM limit
G="omp_mk4x1,omp_mk4x2,omp_gemm,omp_gemm_exact"
$B threads --variants $G --thread-list 1,6 --n-test 9984 --synthetic --n-train 64 --no-seq --tag l2_gemm
$B threads --variants $G --tag gemm
echo "### done"
