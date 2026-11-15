#!/usr/bin/env bash
# Official measurement campaign (~45 min). Run it with the laptop plugged in,
# "High performance" power plan, no other heavy application, and do NOT build
# anything while it runs. All CSVs + system info go to results/raw/.
#   bash scripts/run_all.sh 2>&1 | tee results/raw/run_all_$(date +%Y%m%d_%H%M%S).log
set -euo pipefail
cd "$(dirname "$0")/.."
B="python3 scripts/run_benchmarks.py"
ALL=omp_naive,omp_simd,omp_soa,omp_aosoa,omp_blocked,omp_blocked4
FAST=omp_blocked4

echo "### commit $(git rev-parse --short HEAD) $(git diff --quiet || echo '(dirty)')"
# 1. strong scaling 1-12 threads, every variant, 384 queries
$B threads  --variants "$ALL"
# 2. scheduling at 12 threads (no seq: the comparison is between schedules)
$B schedule --variants omp_naive,omp_simd,omp_blocked,omp_blocked4 --threads 12 --no-seq
# 3. tile sweep of the blocked variants at 6 and 12 threads
$B tiles    --variants omp_blocked,omp_blocked4 --threads 6
$B tiles    --variants omp_blocked,omp_blocked4 --threads 12
# 4. training in L2 / L3 / DRAM, 1 and 6 threads
$B cachefit --variants omp_naive,omp_simd,omp_soa,omp_aosoa --threads 6
# 5. with vs without auto-vectorization (build-novec), 1 thread
$B vec      --variants omp_naive,omp_simd
# 6. k = 1, 5, 10 x euclidean / cosine: time (384 queries) ...
$B kmetric  --variants "$FAST" --threads 12
# ... and accuracy on the whole MNIST test set (10000 queries). Not a timing:
# 1 run, no validation (the fast variant matched the reference with 0
# mismatches on every validated configuration above).
OUT=results/raw/accuracy_full_$(date +%Y%m%d_%H%M%S).csv
for m in euclidean cosine; do
  for k in 1 5 10; do
    OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./build/knn --data data/mnist \
      --variant "$FAST" --threads 12 --metric $m --k $k --runs 1 --warmup 0 --no-validate \
      --tag full --csv "$OUT" | tail -1
  done
done
echo "### done"
