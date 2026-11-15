#!/usr/bin/env bash
# With vs without compiler auto-vectorization (build vs build-novec), 1 thread,
# training in L2 and in DRAM. Thin wrapper around run_benchmarks.py (pinning,
# 5 runs, rotated order of builds and variants, CSV in results/raw/).
# Needs: cmake -S . -B build-novec -DKNN_NO_VEC=ON && cmake --build build-novec -j
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/run_benchmarks.py vec --variants omp_naive,omp_simd "$@"
