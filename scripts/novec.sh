#!/usr/bin/env bash
# With vs without compiler auto-vectorization (-fno-tree-vectorize, build-novec).
# 1 thread: vectorization acts inside one core. Two training sizes: in L2
# (compute-bound regime) and full MNIST (DRAM, memory-bound for omp_simd).
# Needs: cmake -S . -B build-novec -DKNN_NO_VEC=ON && cmake --build build-novec -j
# Usage: bash scripts/novec.sh [out.csv]
set -euo pipefail
OUT="${1:-results/raw/novec_$(date +%Y%m%d_%H%M%S).csv}"
export OMP_SCHEDULE=static

for cfg in 64:10000 60000:100; do
  ntr="${cfg%%:*}"; nte="${cfg##*:}"
  for b in build build-novec; do
    tag=$([ "$b" = build ] && echo vec || echo novec)
    for v in omp_naive omp_simd; do
      "./$b/knn" --data data/mnist --n-train "$ntr" --n-test "$nte" --variant "$v" \
        --threads 1 --runs 3 --warmup 1 --tag "$tag" --csv "$OUT" 2>&1 | tail -1 \
        | sed "s/^/n_train=$ntr  $tag  /"
    done
  done
done
echo "csv: $OUT"
