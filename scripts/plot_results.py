#!/usr/bin/env python3
"""Aggregates benchmark CSVs and draws the report plots.

  python3 scripts/plot_results.py results/raw/threads_*.csv
  python3 scripts/plot_results.py results/raw/schedule_*.csv
  python3 scripts/plot_results.py results/raw/tiles_*.csv

Outputs (results/plots/):
  summary.csv          median/mean/std/min/max time, CPU time, GFLOP/s, speedup,
                       scalability, efficiency, accuracy
  speedup_<cfg>.png    speedup vs seq | scalability | efficiency vs threads
  schedule_<cfg>.png   median time per schedule (if several schedules present)
  tiles_<cfg>.png      GFLOP/s heatmap over (TB, QB) for omp_blocked tile sweeps

Two different metrics (do not mix them):
  speedup     = median T(seq) / median T(variant, p)
                -> the definition of the assignment: total gain, SIMD + threads.
  scalability = median T(variant, 1) / median T(variant, p)
                -> gain from threads only, with the same kernel.
  efficiency  = scalability / p
For SIMD variants the speedup is already > 1 at p = 1, so speedup / p is NOT an
efficiency (it would exceed 1). The seq run used for the speedup is the one with
the same tag, data size, k and metric.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "plots"
CFG = ["tag", "k", "metric", "source", "n_train", "n_test", "dim"]
TILE_TAG = re.compile(r"tb(\d+)_qb(\d+)")


def flops(row):
    # Distance work only: euclidean = sub + mul + add, cosine (dot) = mul + add.
    per_elem = 3 if row["metric"] == "euclidean" else 2
    return per_elem * row["n_train"] * row["n_test"] * row["dim"]


def main(paths):
    if not paths:
        sys.exit(__doc__)
    df = pd.concat([pd.read_csv(p, keep_default_na=False) for p in paths], ignore_index=True)
    if "cpu_s" not in df.columns:          # CSVs written before the CPU-time column
        df["cpu_s"] = float("nan")
    df["cpu_s"] = pd.to_numeric(df["cpu_s"], errors="coerce")
    OUT.mkdir(parents=True, exist_ok=True)

    keys = CFG + ["variant", "threads", "schedule"]
    g = df.groupby(keys)
    agg = g["time_s"].agg(median="median", mean="mean", std="std", min="min", max="max",
                          runs="count").reset_index()
    agg = agg.merge(g["cpu_s"].median().rename("cpu_median").reset_index(), on=keys)
    agg = agg.merge(g["accuracy"].first().reset_index(), on=keys)
    agg = agg.merge(g["mismatches"].max().reset_index(), on=keys)

    # speedup vs the sequential reference (definition of the assignment)
    seq = agg[agg.variant == "seq"][CFG + ["median"]].rename(columns={"median": "t_seq"})
    agg = agg.merge(seq, on=CFG, how="left")
    agg["speedup"] = agg["t_seq"] / agg["median"]

    # scalability vs the same variant at 1 thread (same config and schedule)
    t1 = (agg[agg.threads == 1][CFG + ["variant", "schedule", "median"]]
          .rename(columns={"median": "t_1"}))
    agg = agg.merge(t1, on=CFG + ["variant", "schedule"], how="left")
    agg["scalability"] = agg["t_1"] / agg["median"]
    agg["efficiency"] = agg["scalability"] / agg["threads"]

    agg["gflops"] = agg.apply(flops, axis=1) / agg["median"] / 1e9
    # average number of busy cores: CPU time / wall time (ideal = threads)
    agg["cpu_util"] = agg["cpu_median"] / agg["median"]

    agg.to_csv(OUT / "summary.csv", index=False)
    cols = keys + ["runs", "median", "std", "cpu_median", "cpu_util", "gflops",
                   "speedup", "scalability", "efficiency", "accuracy", "mismatches"]
    print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    is_tile = agg.tag.astype(str).str.fullmatch(r"tb\d+_qb\d+")
    for cfg, gc in agg[(agg.variant != "seq") & ~is_tile].groupby(CFG):
        label = "_".join(str(c) for c in cfg if c != "")
        title = f"n_train={cfg[4]}, n_test={cfg[5]}, k={cfg[1]}, {cfg[2]}" + (f" [{cfg[0]}]" if cfg[0] else "")

        # speedup / scalability / efficiency vs threads (most common schedule only)
        if gc.threads.nunique() > 1:
            sched = gc.schedule.mode()[0]
            gs = gc[gc.schedule == sched]
            fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4))
            tmax = gs.threads.max()
            a2.plot([1, tmax], [1, tmax], "k--", lw=1, label="ideal")
            for v, gv in gs.groupby("variant"):
                gv = gv.sort_values("threads")
                # Error bars = fastest and slowest run (T_x / T_max .. T_x / T_min around
                # the median): with 5 runs and sporadic outliers the range is more
                # honest than a std, which one slow run inflates symmetrically.
                rel = [1 - gv["median"] / gv["max"], gv["median"] / gv["min"] - 1]
                a1.errorbar(gv.threads, gv.speedup, yerr=[gv.speedup * r for r in rel],
                            marker="o", capsize=3, label=v)
                a2.errorbar(gv.threads, gv.scalability, yerr=[gv.scalability * r for r in rel],
                            marker="o", capsize=3, label=v)
                a3.plot(gv.threads, gv.efficiency, marker="o", label=v)
            for a in (a1, a2, a3):
                a.axvline(6, color="grey", lw=0.8, ls=":")   # physical cores
                a.grid(alpha=0.3, which="both"); a.set_xticks(sorted(gs.threads.unique())); a.set_xlabel("threads")
            # log axis: the variants span two orders of magnitude (1x .. 110x)
            a1.set_yscale("log")
            a1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            a3.text(6.15, 0.02, "6 physical cores", color="grey", fontsize=8, va="bottom")
            a1.set(ylabel="T_seq / T_p (log scale)", title=f"Speedup vs seq ({sched}), bars = min-max")
            a2.set(ylabel="T_1 / T_p", title="Scalability (same variant)")
            top = max(1.15, float(gs.efficiency.max()) + 0.05)
            a3.set(ylabel="scalability / p", title="Efficiency", ylim=(0, top))
            a3.axhline(1, color="k", lw=0.8, ls="--")
            for a in (a1, a2, a3):
                a.legend(fontsize=8)
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(OUT / f"speedup_{label}.png", dpi=150)
            plt.close(fig)

        # time per schedule
        if gc.schedule.nunique() > 1:
            # One panel per variant (own y scale: the variants differ by 15x), schedules
            # in the same order everywhere so the panels can be compared by eye.
            order = list(dict.fromkeys(df.schedule[df.schedule != "none"]))   # order of execution
            names = sorted(gc.variant.unique())
            fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 4), squeeze=False)
            for ax, v in zip(axes[0], names):
                gv = gc[gc.variant == v].set_index("schedule").reindex(order).dropna(subset=["median"])
                err = [gv["median"] - gv["min"], gv["max"] - gv["median"]]
                ax.bar(gv.index, gv["median"], yerr=err, capsize=3, color="tab:blue")
                ax.set_title(f"{v}, {int(gv.threads.iloc[0])} threads", fontsize=10)
                ax.grid(axis="y", alpha=0.3)
                ax.tick_params(axis="x", labelrotation=60, labelsize=8)
            axes[0][0].set_ylabel("median time [s], bars = min-max")
            fig.suptitle(f"Scheduling (OMP_SCHEDULE kind-chunk) - {title}")
            fig.tight_layout()
            fig.savefig(OUT / f"schedule_{label}.png", dpi=150)
            plt.close(fig)

    # tile sweep: tags like "tb32_qb16" (written by run_benchmarks.py tiles)
    tiles = agg[is_tile].copy()
    if not tiles.empty:
        tiles[["tb", "qb"]] = tiles.tag.str.extract(TILE_TAG).astype(int)
        for (thr, v), gt in tiles.groupby(["threads", "variant"]):
            piv = gt.pivot_table(index="qb", columns="tb", values="gflops")
            fig, ax = plt.subplots(figsize=(8, 3.8))
            im = ax.imshow(piv.values, cmap="viridis", aspect="auto", origin="lower")
            ax.set_xticks(range(len(piv.columns)), [f"{c}\n{c * 3136 // 1024} KB" for c in piv.columns])
            ax.set_yticks(range(len(piv.index)), piv.index)
            for (r, c), val in pd.DataFrame(piv.values).stack().items():
                ax.text(c, r, f"{val:.0f}", ha="center", va="center", color="w", fontsize=8)
            ax.set(xlabel="TB (training rows per tile, tile size)", ylabel="QB (queries per block)",
                   title=f"{v}, {thr} threads: GFLOP/s")
            fig.colorbar(im, ax=ax, label="GFLOP/s")
            fig.tight_layout()
            fig.savefig(OUT / f"tiles_{v}_t{thr}.png", dpi=150)
            plt.close(fig)

    print(f"\nplots and summary in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
