#!/usr/bin/env bash
# Numerical behaviour of the matrix formulation |q|^2 + |t|^2 - 2 q.t
# (omp_gemm) against the direct kernel, as the data are shifted by a constant.
# COUNTERS ONLY (errors, negative distances, mismatches, recomputed candidates):
# no timing is taken, so it can run in any condition (~3 min; the references are
# computed by seq on 64 queries and cached).
#   bash scripts/gemm_numerics.sh            # writes results/raw/gemm_numerics_<stamp>.log
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="results/raw/gemm_numerics_$(date +%Y%m%d_%H%M%S).log"
N_TEST="${N_TEST:-64}"
{
  echo "# omp_gemm numerics - counters, NOT a timing experiment"
  echo "# commit $(git rev-parse --short HEAD)  MNIST 60000 x $N_TEST, k=5, euclidean  $(date -Iseconds)"
  echo "# columns: offset variant | counters printed by the variant | mismatches vs seq on the same shifted data"
  for off in 0 1 4 16 64 256 1024; do
    for v in omp_blocked_avx2 omp_gemm omp_gemm_exact; do
      line=$(KNN_GEMM_STATS=1 KNN_VERBOSE=1 OMP_SCHEDULE=static ./build/knn --data data/mnist \
               --n-test "$N_TEST" --offset "$off" --variant "$v" --k 5 --threads 6 --runs 1 --warmup 0 \
               --max-mismatch 1 2>&1 |
             grep -o 'negative=[0-9]* rechecked=[0-9]* ([0-9.]*%)\|max_abs_err=[0-9.e+-]* max_err/scale=[0-9.e+-]*\|acc=[0-9.]* mismatches=[0-9]*' |
             tr '\n' ' ')
      echo "offset=$off $v | $line"
    done
  done
} | tee "$OUT"
echo "log: $OUT"
