#!/usr/bin/env bash
# Extra experiments from report/review.md (the numbers in brackets are the
# experiment numbers there). Run AFTER run_rest.sh, same conditions: laptop
# plugged in, "High performance" plan, nothing else running, no builds.
# ~25 min for parts 1-9, + ~15 min for the input-size sweep (part 10).
#   bash scripts/run_extra.sh 2>&1 | tee results/raw/run_extra_$(date +%Y%m%d_%H%M%S).log
#   SKIP_SIZES=1 bash scripts/run_extra.sh ...      # without part 10
set -euo pipefail
cd "$(dirname "$0")/.."
B="python3 scripts/run_benchmarks.py"
echo "### commit $(git rev-parse --short HEAD) $(git diff --quiet || echo '(dirty)')"

# 1. [5, 8] real clock frequency and DRAM bandwidth (~1.5 min)
bash scripts/hwprobe.sh

# 2. [1] where does the 12-thread collapse start? (~4 min)
$B threads  --variants omp_simd,omp_naive --thread-list 6,7,8,9,10,11,12 --no-seq --tag cliff

# 3. [2] staggered start: is the L3 sharing real? (~1.5 min)
$B threads  --variants omp_simd,omp_simd_stagger --thread-list 2,6 --no-seq --tag stagger

# 4. [3] binding probe again, right after a reference point (~2 min)
bash scripts/variance_probe.sh

# 5. [7] tile sweep at 1 thread: T(QB) = a + b/QB, b = DRAM time (~8 min)
$B tiles    --variants omp_blocked,omp_blocked4 --threads 1

# 6. [4, H] cache-fit at 2 and 12 threads: efficiency > 1 of naive, clean SMT gain (~2 min)
$B cachefit --variants omp_naive,omp_simd --thread-list 2,12 --tag ht

# 7. [J] scheduling where it can matter: 6 threads; block-sized chunks (~4 min)
$B schedule --variants omp_naive,omp_simd --threads 6 --no-seq --tag s6
$B schedule --variants omp_blocked,omp_blocked4 --threads 12 --no-seq --tag chunk \
            --schedules static static,1 static,2 static,4 dynamic,1 dynamic,2 dynamic,4 guided

# 8. [P1] serial row norms = Amdahl: cosine with serial vs parallel norms (~1 min)
$B kmetric  --variants omp_blocked4,omp_blocked4_pn --threads 12 --no-seq --tag norms

# 9. [6, 10] synthetic data in cache, 1 thread (~2 min)
#    same flops, rows 10x longer: does the out-of-order overlap of short loops matter?
$B threads  --variants omp_simd --thread-list 1 --synthetic --n-train 64 --dim 784  --n-test 10000 --no-seq --tag d784
$B threads  --variants omp_simd --thread-list 1 --synthetic --n-train 6  --dim 7840 --n-test 10000 --no-seq --tag d7840
#    SoA block that fits in L1 (64 x 100 floats = 25 KB): is "SoA in L2" bound by L2 -> L1?
$B threads  --variants omp_simd,omp_soa,omp_aosoa --thread-list 1 --synthetic --n-train 64 --dim 100 \
            --n-test 400000 --no-seq --tag l1

# 10. speedup curves for several input sizes - required by the assignment (~15 min)
if [ -z "${SKIP_SIZES:-}" ]; then
  $B sizes  --variants omp_simd,omp_blocked4 --n-trains 15000,30000,60000,120000
fi
echo "### done"
