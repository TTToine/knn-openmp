#!/usr/bin/env bash
# Why does omp_simd at 6 threads vary so much (0.35-1.10 s)? Hypothesis: without
# pinning the OS may place two threads on the two hyperthreads of the same core
# (and migrate them), so the run time depends on placement. Compare 10 runs:
#   unbound            : OMP_PROC_BIND unset (OS scheduler decides)
#   close/cores        : one thread per physical core, pinned
#   spread/threads     : pinned to hardware threads, spread over the cores
# Usage: bash scripts/variance_probe.sh [out.csv]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/raw/variance_$(date +%Y%m%d_%H%M%S).csv}"
export OMP_SCHEDULE=static
lscpu -e=CPU,CORE | tr '\n' ' '; echo   # which logical CPUs are hyperthread siblings
run() {  # $1 = tag, rest = env assignments
  local tag=$1; shift
  env "$@" ./build/knn --data data/mnist --n-test 384 --variant omp_simd --threads 6 \
    --runs 10 --warmup 1 --tag "$tag" --csv "$OUT" 2>&1 | tail -1 | sed "s/^/$tag  /"
}
run unbound      -u OMP_PROC_BIND -u OMP_PLACES
run close_cores  OMP_PROC_BIND=close  OMP_PLACES=cores
run spread_thr   OMP_PROC_BIND=spread OMP_PLACES=threads
run close_thr    OMP_PROC_BIND=close  OMP_PLACES=threads
run unbound2     -u OMP_PROC_BIND -u OMP_PLACES
echo "csv: $OUT"
