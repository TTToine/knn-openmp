#!/usr/bin/env bash
# Is omp_simd memory-bound? Compare naive vs simd with a training set that fits
# in L2 (64 rows = 200 KB), in L3 (2000 rows = 6.3 MB) and that does not (60000
# rows = 188 MB). If the SIMD gain grows when the data fits in cache, the full
# MNIST run is limited by memory bandwidth, not by arithmetic.
# Usage: bash scripts/cachefit.sh [out.csv]
set -euo pipefail
KNN=./build/knn
OUT="${1:-results/raw/cachefit_$(date +%Y%m%d_%H%M%S).csv}"
export OMP_SCHEDULE=static

# n_train:n_test  (n_test chosen to keep each run around 0.5-5 s)
for cfg in 64:10000 2000:2000 60000:100; do
  ntr="${cfg%%:*}"; nte="${cfg##*:}"
  for t in 1 6; do
    for v in omp_naive omp_simd; do
      "$KNN" --data data/mnist --n-train "$ntr" --n-test "$nte" --variant "$v" \
        --threads "$t" --runs 3 --warmup 1 --tag cachefit --csv "$OUT" 2>&1 | tail -1 \
        | sed "s/^/n_train=$ntr  /"
    done
  done
done
echo "csv: $OUT"
