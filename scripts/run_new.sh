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
echo "### done"
