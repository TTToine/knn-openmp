#!/usr/bin/env bash
# Is omp_simd memory-bound? omp_naive vs omp_simd with a training set that fits
# in L2 (64 rows = 200 KB), in L3 (2000 rows = 6.3 MB) and in DRAM (60000 rows).
# Thin wrapper around run_benchmarks.py (pinning, 5 runs, rotated order, CSV +
# system info in results/raw/). Extra arguments are passed through.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/run_benchmarks.py cachefit --variants omp_naive,omp_simd --threads 6 "$@"
