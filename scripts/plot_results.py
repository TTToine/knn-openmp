#!/usr/bin/env python3
"""Aggregates benchmark CSVs and draws the report plots.

  python3 scripts/plot_results.py results/raw/threads_*.csv
  python3 scripts/plot_results.py results/raw/schedule_*.csv

Outputs (results/plots/):
  summary.csv          median/mean/std/min/max time + speedup + efficiency
  speedup_<cfg>.png    speedup and efficiency vs threads (one line per variant)
  schedule_<cfg>.png   median time per schedule (if several schedules present)
Speedup = median T(seq) / median T(variant), with the seq run that has the
same tag, data size, k and metric.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "plots"
CFG = ["tag", "k", "metric", "source", "n_train", "n_test", "dim"]


def main(paths):
    if not paths:
        sys.exit(__doc__)
    df = pd.concat([pd.read_csv(p, keep_default_na=False) for p in paths], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)

    keys = CFG + ["variant", "threads", "schedule"]
    agg = (df.groupby(keys)["time_s"]
             .agg(median="median", mean="mean", std="std", min="min", max="max", runs="count")
             .reset_index())
    seq = agg[agg.variant == "seq"][CFG + ["median"]].rename(columns={"median": "t_seq"})
    agg = agg.merge(seq, on=CFG, how="left")
    agg["speedup"] = agg["t_seq"] / agg["median"]
    agg["efficiency"] = agg["speedup"] / agg["threads"]
    agg.to_csv(OUT / "summary.csv", index=False)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    for cfg, g in agg[agg.variant != "seq"].groupby(CFG):
        label = "_".join(str(c) for c in cfg if c != "")
        title = f"n_train={cfg[4]}, n_test={cfg[5]}, k={cfg[1]}, {cfg[2]}" + (f" [{cfg[0]}]" if cfg[0] else "")

        # speedup vs threads (only rows of the most common schedule, to avoid mixing)
        if g.threads.nunique() > 1:
            sched = g.schedule.mode()[0]
            gs = g[g.schedule == sched]
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
            tmax = gs.threads.max()
            a1.plot([1, tmax], [1, tmax], "k--", lw=1, label="ideal")
            for v, gv in gs.groupby("variant"):
                gv = gv.sort_values("threads")
                # error bar on speedup derived from the time std (first-order approx.)
                err = gv["speedup"] * gv["std"].fillna(0) / gv["median"]
                a1.errorbar(gv.threads, gv.speedup, yerr=err, marker="o", capsize=3, label=v)
                a2.plot(gv.threads, gv.efficiency, marker="o", label=v)
            a1.axvline(6, color="grey", lw=0.8, ls=":")          # physical cores
            a1.text(6.1, 0.5, "6 physical cores", color="grey", fontsize=8)
            a1.set(xlabel="threads", ylabel="speedup", title=f"Speedup ({sched})")
            a2.set(xlabel="threads", ylabel="efficiency", title="Efficiency", ylim=(0, 1.15))
            for a in (a1, a2):
                a.grid(alpha=0.3); a.legend(); a.set_xticks(sorted(gs.threads.unique()))
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(OUT / f"speedup_{label}.png", dpi=150)
            plt.close(fig)

        # time per schedule
        if g.schedule.nunique() > 1:
            fig, ax = plt.subplots(figsize=(9, 4))
            for i, (v, gv) in enumerate(g.groupby("variant")):
                gv = gv.sort_values("median")
                ax.bar([f"{s}\n{v}" for s in gv.schedule], gv["median"], yerr=gv["std"].fillna(0), capsize=3)
            ax.set(ylabel="median time [s]", title=f"Scheduling - {title}")
            ax.grid(axis="y", alpha=0.3)
            plt.setp(ax.get_xticklabels(), fontsize=8)
            fig.tight_layout()
            fig.savefig(OUT / f"schedule_{label}.png", dpi=150)
            plt.close(fig)

    print(f"\nplots and summary in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
