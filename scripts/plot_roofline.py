#!/usr/bin/env python3
"""Roofline plot of the k-NN variants (see report/perf_model.md).

  python3 scripts/plot_roofline.py results/raw/threads_*.csv
  python3 scripts/plot_roofline.py results/raw/threads_*.csv --hwprobe results/raw/hwprobe_*.csv

x = arithmetic intensity with respect to DRAM [flop / byte read from DRAM]
    0.75      variants where every query streams the whole training set
              (3 flop per 4-byte float)
    0.75 * QB blocked variants: a tile is read from DRAM once per block of QB queries
y = GFLOP/s (median of the measured runs)

Roofs:
  diagonals   = memory bandwidth x intensity: one core (B1), all cores (measured
                if --hwprobe is given, else the theoretical DDR4 peak)
  horizontals = compute: flop/cycle x frequency x cores, for the scalar ordered
                sum (0.75), one ymm accumulator (6) and the FMA-port peak (24)
A point ABOVE a memory diagonal is not an error: it means the bytes did not come
from DRAM (threads sharing the same rows through L3, the "convoy").

Without --hwprobe the frequency (--ghz) and the bandwidths are ASSUMPTIONS and the
figure says so in its title. Only measured CSVs go in: never predictions.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "plots"
BLOCKED = ("omp_blocked", "omp_blocked4", "omp_blocked4_pn")
FLOP_PER_CYCLE = [(0.75, "scalar ordered sum"), (6, "1 ymm accumulator"), (24, "FMA-port peak")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+")
    p.add_argument("--hwprobe", help="CSV written by scripts/hwprobe.sh (measured GHz and GB/s)")
    p.add_argument("--ghz", type=float, default=4.2, help="assumed frequency if no --hwprobe")
    p.add_argument("--b1", type=float, default=10.9, help="one-core DRAM bandwidth [GB/s] if no --hwprobe")
    p.add_argument("--bmax", type=float, default=23.5, help="all-core DRAM bandwidth [GB/s] if no --hwprobe")
    p.add_argument("--qb", type=int, default=16, help="query block of the blocked variants")
    p.add_argument("--threads", default="1,6", help="thread counts to plot")
    args = p.parse_args()
    threads = [int(t) for t in args.threads.split(",")]

    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in args.csv], ignore_index=True)
    df = df[(df.metric == "euclidean") & (df.variant != "seq") & df.threads.isin(threads)
            & ~df.tag.astype(str).str.contains("tb")]
    keys = ["variant", "threads", "n_train", "n_test", "dim"]
    agg = df.groupby(keys)["time_s"].median().reset_index()
    agg["gflops"] = 3 * agg.n_train * agg.n_test * agg.dim / agg.time_s / 1e9
    agg["ai"] = [0.75 * (args.qb if v in BLOCKED else 1) for v in agg.variant]

    ghz = {t: args.ghz for t in threads}
    b1, bmax, source = args.b1, args.bmax, f"ASSUMED: {args.ghz} GHz, DRAM {args.b1} / {args.bmax} GB/s"
    if args.hwprobe:
        hw = pd.read_csv(args.hwprobe)
        f = hw[hw.probe == "freq_avx2"].groupby("threads")["value"].median()
        ghz = {t: float(f.get(t, f.iloc[-1])) for t in threads}
        bw = hw[hw.probe == "membw_read"].groupby("threads")["value"].median()
        b1, bmax = float(bw.get(1, bw.iloc[0])), float(bw.max())
        source = f"measured (hwprobe): {ghz} GHz, DRAM {b1:.1f} / {bmax:.1f} GB/s"

    fig, ax = plt.subplots(figsize=(9.5, 6))
    xs = [0.3, 40]
    for bw_, lab, ls in [(b1, f"DRAM, 1 core ({b1:.1f} GB/s)", ":"), (bmax, f"DRAM, all cores ({bmax:.1f} GB/s)", "--")]:
        ax.plot(xs, [bw_ * x for x in xs], "k", ls=ls, lw=1.2, label=lab)
    colors = {t: c for t, c in zip(threads, ["tab:blue", "tab:red", "tab:green", "tab:purple"])}
    for t in threads:
        cores = min(t, 6)
        for fpc, lab in FLOP_PER_CYCLE:
            y = fpc * ghz[t] * cores
            ax.plot(xs, [y, y], color=colors[t], lw=0.9, alpha=0.6)
            ax.text(xs[1] * 0.98, y * 1.04, f"{lab}, {cores} core{'s' if cores > 1 else ''}: {y:.0f}",
                    color=colors[t], fontsize=7, ha="right")
    markers = dict(zip(sorted(agg.variant.unique()), "osD^vP*X<>"))
    for _, r in agg.iterrows():
        ax.scatter(r.ai * (0.93 if r.threads == threads[0] else 1.07), r.gflops, s=70, zorder=3,
                   color=colors[r.threads], marker=markers[r.variant], edgecolor="k", linewidth=0.6)
    for v, m in markers.items():
        ax.scatter([], [], color="w", edgecolor="k", marker=m, s=60, label=v)
    for t in threads:
        ax.scatter([], [], color=colors[t], marker="s", s=60, label=f"{t} thread{'s' if t > 1 else ''}")
    cfg = agg.iloc[0]
    ax.set(xscale="log", yscale="log", xlim=xs, ylim=(1.5, 1200),
           xlabel="arithmetic intensity w.r.t. DRAM [flop / byte]  (0.75 = re-read everything, "
                  f"{0.75 * args.qb:g} = blocked, QB={args.qb})",
           ylabel="GFLOP/s (median)",
           title=f"Roofline, MNIST {cfg.n_train} x {cfg.n_test} queries, euclidean\nroofs {source}")
    ax.title.set_fontsize(10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "roofline.png", dpi=150)
    print(agg[["variant", "threads", "ai", "gflops"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n{OUT.relative_to(ROOT) / 'roofline.png'}")


if __name__ == "__main__":
    main()
