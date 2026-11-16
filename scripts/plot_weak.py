#!/usr/bin/env python3
"""Weak-scaling plot from a `run_benchmarks.py weak` CSV.

  python3 scripts/plot_weak.py results/raw/weak_*.csv

Weak scaling: the problem grows WITH the threads (work per thread constant), so
the ideal is a constant time.
  weak efficiency  E_w(p) = T(1) / T(p)                 (ideal 1)
  scaled speedup   S(p)   = p * T(1) / T(p)             (ideal p)
Gustafson's law predicts S(p) = p - s (p - 1) with s the serial fraction of the
scaled run: the plot draws it for the s fitted on the points up to 6 threads.
The amount of work is taken from the CSV (n_train * n_test), so the script
checks that work / threads is really constant.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "plots"


def main(paths):
    if not paths:
        sys.exit(__doc__)
    df = pd.concat([pd.read_csv(p, keep_default_na=False) for p in paths], ignore_index=True)
    df = df[df.variant != "seq"]
    df["work"] = df.n_train * df.n_test
    g = (df.groupby(["variant", "threads"])
           .agg(median=("time_s", "median"), tmin=("time_s", "min"), tmax=("time_s", "max"),
                work=("work", "first")).reset_index())
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4))
    pmax = g.threads.max()
    a3.plot([1, pmax], [1, pmax], "k--", lw=1, label="ideal")
    for v, gv in g.groupby("variant"):
        gv = gv.sort_values("threads")
        base = gv[gv.threads == 1]
        if base.empty:
            print(f"{v}: no 1-thread point, skipped")
            continue
        t1, w1 = float(base["median"].iloc[0]), float(base["work"].iloc[0])
        per_thread = gv.work / gv.threads / w1
        if not np.allclose(per_thread, 1.0, rtol=0.02):
            print(f"WARNING {v}: work per thread is not constant ({per_thread.round(2).tolist()})")
        eff = t1 / gv["median"]
        scaled = gv.threads * eff
        a1.errorbar(gv.threads, gv["median"], yerr=[gv["median"] - gv.tmin, gv.tmax - gv["median"]],
                    marker="o", capsize=3, label=v)
        a2.plot(gv.threads, eff, marker="o", label=v)
        line, = a3.plot(gv.threads, scaled, marker="o", label=v)
        fit = gv.threads.between(2, 6)
        if fit.any():   # S = p - s (p - 1)  ->  s = (p - S) / (p - 1), least squares through the points
            pf, sf = gv.threads[fit].to_numpy(float), scaled[fit].to_numpy(float)
            s = float(np.sum((pf - sf) * (pf - 1)) / np.sum((pf - 1) ** 2))
            pp = np.linspace(1, pmax, 50)
            a3.plot(pp, pp - s * (pp - 1), color=line.get_color(), lw=0.8, ls=":")
            print(f"{v}: Gustafson serial fraction fitted on 2..6 threads: s = {s:.3f}")
        print(pd.DataFrame({"threads": gv.threads, "time_s": gv["median"].round(4), "weak_eff": eff.round(3),
                            "scaled_speedup": scaled.round(2)}).to_string(index=False))
    for a in (a1, a2, a3):
        a.axvline(6, color="grey", lw=0.8, ls=":")
        a.grid(alpha=0.3); a.set_xlabel("threads (problem size grows with them)"); a.legend(fontsize=8)
        a.set_xticks(sorted(g.threads.unique()))
    a1.set(ylabel="median time [s], bars = min-max", title="Weak scaling: time (ideal: flat)")
    a2.set(ylabel="T(1) / T(p)", title="Weak efficiency (ideal: 1)")
    a2.axhline(1, color="k", lw=0.8, ls="--")
    a2.set_ylim(0, max(1.15, a2.get_ylim()[1]))
    a3.set(ylabel="p * T(1) / T(p)", title="Scaled speedup; dotted = Gustafson fit")
    fig.tight_layout()
    fig.savefig(OUT / "weak_scaling.png", dpi=150)
    print(f"\nplot: {(OUT / 'weak_scaling.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
