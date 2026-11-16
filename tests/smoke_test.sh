#!/usr/bin/env bash
# Validates EVERY registered variant against the sequential reference on small
# synthetic inputs (both metrics, several k, several thread counts).
# Usage: tests/smoke_test.sh path/to/knn
# MAX_MISMATCH=0.01 tolerates 1% differing predictions (only for variants that
# reorder floating-point sums, e.g. omp simd: explain it in the report!).
set -euo pipefail
KNN="${1:-./build/knn}"
CACHE="$(mktemp -d)"
trap 'rm -rf "$CACHE"' EXIT

export KNN_STAGGER_MS=1   # omp_simd_stagger: keep the test fast (default 20 ms per thread)
fail=0
for variant in $("$KNN" --list | cut -f1); do
  for metric in euclidean cosine; do
    for k in 1 5 10; do
      for threads in 1 3 4; do
        for sched in static dynamic,7 guided; do
          if ! OMP_SCHEDULE="$sched" "$KNN" --synthetic --n-train 1500 --n-test 97 --dim 50 \
               --variant "$variant" --metric "$metric" --k "$k" --threads "$threads" \
               --runs 1 --warmup 0 --cache-dir "$CACHE" \
               --max-mismatch "${MAX_MISMATCH:-0}" > /dev/null 2>&1; then
            echo "FAIL: variant=$variant metric=$metric k=$k threads=$threads schedule=$sched"
            fail=1
          fi
        done
      done
    done
  done
done
# Odd sizes (97 test samples, 50 dims) are intentional: they expose bugs in
# chunking, remainders of SIMD loops and blocking.

# Blocked variants: odd tile / query-block sizes (partial tiles, partial blocks,
# query groups that are not a multiple of 4, tiles larger than the data).
for variant in $("$KNN" --list | cut -f1 | grep blocked); do
  for tbqb in "1 1" "7 5" "13 97" "100000 1000" "32 6"; do
    set -- $tbqb
    for metric in euclidean cosine; do
      if ! KNN_TB="$1" KNN_QB="$2" OMP_SCHEDULE=dynamic,2 "$KNN" --synthetic --n-train 1500 --n-test 97 \
           --dim 50 --variant "$variant" --metric "$metric" --k 5 --threads 3 --runs 1 --warmup 0 \
           --cache-dir "$CACHE" --max-mismatch "${MAX_MISMATCH:-0}" > /dev/null 2>&1; then
        echo "FAIL: variant=$variant metric=$metric KNN_TB=$1 KNN_QB=$2"
        fail=1
      fi
    done
  done
done

# k larger than the training set: the vote must use the n_train neighbours found.
for variant in $("$KNN" --list | cut -f1); do
  if ! OMP_SCHEDULE=static "$KNN" --synthetic --n-train 3 --n-test 20 --dim 50 --variant "$variant" \
       --k 10 --threads 3 --runs 1 --warmup 0 --cache-dir "$CACHE" > /dev/null 2>&1; then
    echo "FAIL: variant=$variant k > n_train"
    fail=1
  fi
done
if [ "$fail" -eq 0 ]; then echo "all variants OK"; fi
exit "$fail"
