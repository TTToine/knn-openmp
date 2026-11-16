#!/usr/bin/env bash
# Clock frequency under load and DRAM read bandwidth for 1..12 threads, with the
# same pinning as the k-NN runs (~1.5 min). Part of the measurement block: laptop
# plugged in, nothing else running.   bash scripts/hwprobe.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="results/raw/hwprobe_$(date +%Y%m%d_%H%M%S).csv"
bash scripts/system_info.sh > "${OUT%.csv}_system.txt" 2>&1
echo "probe,threads,thread,run,value,unit" > "$OUT"
export OMP_PROC_BIND=close OMP_PLACES=cores
for t in 1 2 4 6 8 12; do
  OMP_NUM_THREADS=$t ./build/hwprobe freq 1.0  >> "$OUT"
  OMP_NUM_THREADS=$t ./build/hwprobe membw 1024 >> "$OUT"
done
# summary: median over runs (and over threads for the frequency)
python3 - "$OUT" <<'EOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
print(df.groupby(["probe", "threads", "unit"])["value"].agg(["median", "min", "max"]).round(2).to_string())
EOF
echo "csv: $OUT"
