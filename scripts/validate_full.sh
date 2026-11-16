#!/usr/bin/env bash
# Correctness on the WHOLE MNIST test set (10000 queries): every registered
# variant against the sequential reference, k = 1, 5, 10 x euclidean / cosine.
#
# This is a TEST, not a measurement: the times printed in the log are taken in
# uncontrolled conditions (6 reference processes in parallel, any power state)
# and must never be used in the report. What matters is the `mismatches=` field.
# Takes ~30-40 min: the references are computed by `seq` (~8 min each on one
# core), one process per configuration, each pinned to its own physical core.
#   bash scripts/validate_full.sh            # THREADS=8 by default
set -uo pipefail
cd "$(dirname "$0")/.."
THREADS="${THREADS:-8}"
OUT="results/raw/validation_full_$(date +%Y%m%d_%H%M%S).log"
{
  echo "# full-test-set validation - NOT A TIMING EXPERIMENT (see scripts/validate_full.sh)"
  echo "# commit $(git rev-parse --short HEAD) $(git diff --quiet -- src include CMakeLists.txt || echo '(dirty sources)')  $(date -Iseconds)"
} | tee "$OUT"

# 1. references (cached in .cache/, reused if already there)
cpu=0
for m in euclidean cosine; do
  for k in 1 5 10; do
    ref=".cache/ref_mnist_tr60000_te10000_k${k}_${m}.bin"
    if [ ! -s "$ref" ]; then
      taskset -c "$cpu" ./build/knn --data data/mnist --variant seq --metric "$m" --k "$k" \
        --runs 1 --warmup 0 > /dev/null 2>&1 &
    fi
    cpu=$((cpu + 2))   # CPUs 0,2,4,... = one per physical core (siblings are 0-1, 2-3, ...)
  done
done
wait
echo "# references ready $(date -Iseconds)" | tee -a "$OUT"

# 2. every parallel variant against them
fail=0
for m in euclidean cosine; do
  for k in 1 5 10; do
    for v in $(./build/knn --list | cut -f1 | grep -v '^seq$'); do
      line=$(OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./build/knn --data data/mnist \
               --variant "$v" --threads "$THREADS" --metric "$m" --k "$k" --runs 1 --warmup 0 2>/dev/null)
      rc=$?
      [ "$rc" -ne 0 ] && fail=1
      echo "metric=$m k=$k rc=$rc $line" | sed -E 's/ (median|mean|std|min|max|cpu)=[0-9.]+s//g' | tee -a "$OUT"
    done
  done
done
echo "# done $(date -Iseconds) - $([ "$fail" -eq 0 ] && echo 'ALL VARIANTS MATCH THE REFERENCE' || echo 'SOME VARIANT DIFFERS (rc=2 lines above)')" | tee -a "$OUT"
exit "$fail"
