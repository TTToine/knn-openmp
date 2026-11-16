#!/usr/bin/env bash
# Synchronisation strategies on the confusion matrix (tools/sync_strategies.cpp):
# critical vs atomic vs reduction vs private counters with/without false
# sharing, 1..12 threads, same pinning as the k-NN runs (~1 min).
# Part of the measurement block: laptop plugged in, nothing else running.
#   bash scripts/sync_strategies.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="results/raw/sync_$(date +%Y%m%d_%H%M%S).csv"
bash scripts/system_info.sh > "${OUT%.csv}_system.txt" 2>&1
echo "strategy,threads,run,seconds,Mupdates_per_s" > "$OUT"
export OMP_PROC_BIND=close OMP_PLACES=cores
for t in 1 2 4 6 12; do
  OMP_NUM_THREADS=$t ./build/sync_strategies 20000000 5 >> "$OUT"
done
python3 - "$OUT" <<'EOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
df = df[~((df.strategy == "serial") & (df.groupby(["strategy", "threads", "run"]).cumcount() > 0))]
t = df.groupby(["strategy", "threads"])["Mupdates_per_s"].median().unstack("threads").round(1)
print("median million updates per second (rows: strategy, columns: threads)")
print(t.to_string())
EOF
echo "csv: $OUT"
