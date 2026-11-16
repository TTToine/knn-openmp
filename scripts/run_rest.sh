#!/usr/bin/env bash
# Second half of the campaign (redo after the battery incident, ~30 min):
# tile sweep, cache-fit, vec/novec, k x metric timings. Laptop plugged in!
# run_benchmarks.py aborts by itself if the charger is unplugged.
#   bash scripts/run_rest.sh 2>&1 | tee results/raw/run_rest_$(date +%Y%m%d_%H%M%S).log
set -euo pipefail
cd "$(dirname "$0")/.."
B="python3 scripts/run_benchmarks.py"
echo "### commit $(git rev-parse --short HEAD) $(git diff --quiet || echo '(dirty)')"
cat /sys/class/power_supply/*/online 2>/dev/null | tr '\n' ' '; echo "(AC online)"
# drift check BEFORE spending 30 minutes (~2.5 min): seq must be 19.1 s +- 3% and
# omp_blocked4 at 12 threads 0.171 s +- 5% (threads_20261115_184752). If not, stop
# (Ctrl-C) and fix the conditions: power plan, charger, background applications.
$B threads  --variants omp_blocked4 --thread-list 12
$B tiles   --variants omp_blocked,omp_blocked4 --threads 6
$B tiles    --variants omp_blocked,omp_blocked4 --threads 12
$B cachefit --variants omp_naive,omp_simd,omp_soa,omp_aosoa --threads 6
$B vec      --variants omp_naive,omp_simd
$B kmetric  --variants omp_blocked4 --threads 12
# drift check: the same configuration as the start of threads_20261115_184752
# (seq 19.08 s, omp_blocked4 12 thr 0.171 s); a big difference = something changed
$B threads  --variants omp_blocked4 --runs 5
echo "### done"
