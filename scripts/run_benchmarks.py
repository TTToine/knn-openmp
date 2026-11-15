#!/usr/bin/env python3
"""Runs a benchmark experiment and appends the results to a CSV in results/raw/.

Each experiment is a sweep over one dimension (threads, schedule, k/metric,
input size). The sequential baseline is always measured with the same data, so
plot_results.py can compute the speedup.

Examples:
  python3 scripts/run_benchmarks.py threads  --variants omp_naive
  python3 scripts/run_benchmarks.py schedule --variants omp_naive --threads 12
  python3 scripts/run_benchmarks.py sizes    --variants omp_naive
  python3 scripts/run_benchmarks.py kmetric  --variants omp_naive
  python3 scripts/run_benchmarks.py threads  --variants omp_naive --binary build-novec/knn --tag novec
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

THREADS = [1, 2, 4, 6, 8, 12]            # i7-10750H: 6 cores / 12 hw threads
SCHEDULES = ["static", "static,1", "static,16", "dynamic,1", "dynamic,16",
             "dynamic,64", "guided"]
REPLICATES = [1, 2, 4]                   # training-set sizes for the size sweep
K_VALUES = [1, 5, 10]
METRICS = ["euclidean", "cosine"]


def run(binary, out_csv, variant, threads, schedule, args, extra):
    cmd = [str(binary), "--variant", variant, "--threads", str(threads),
           "--runs", str(args.runs), "--warmup", str(args.warmup),
           "--csv", str(out_csv), "--tag", args.tag] + data_args(args) + extra
    env = dict(os.environ, OMP_SCHEDULE=schedule, OMP_PROC_BIND="close", OMP_PLACES="cores"
               if threads <= args.physical_cores else "threads")
    print(f"[{dt.datetime.now():%H:%M:%S}] {variant:12s} t={threads:<3d} sched={schedule:12s} {' '.join(extra)}",
          flush=True)
    res = subprocess.run(cmd, env=env, cwd=ROOT, text=True, capture_output=True)
    print("   ", res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        if res.returncode == 2:
            sys.exit("validation failed - fix the variant before benchmarking it")
        sys.exit(f"knn exited with code {res.returncode}")


def data_args(args):
    if args.synthetic:
        return ["--synthetic", "--n-train", str(args.n_train), "--n-test", str(args.n_test)]
    return ["--data", args.data, "--n-test", str(args.n_test)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=["threads", "schedule", "sizes", "kmetric"])
    p.add_argument("--variants", default="omp_naive", help="comma-separated parallel variants")
    p.add_argument("--binary", default="build/knn")
    p.add_argument("--tag", default="", help="label stored in the CSV (e.g. novec)")
    p.add_argument("--data", default="data/mnist")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n-train", type=int, default=60000, help="synthetic only")
    p.add_argument("--n-test", type=int, default=500,
                   help="calibrate so that the sequential run takes >= 10 s")
    p.add_argument("--threads", type=int, default=12, help="thread count for non-thread sweeps")
    p.add_argument("--schedule", default="static", help="schedule for non-schedule sweeps")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--physical-cores", type=int, default=6)
    args = p.parse_args()

    binary = ROOT / args.binary
    if not binary.exists():
        sys.exit(f"{binary} not found - build first (see CLAUDE.md)")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.experiment}_{args.tag + '_' if args.tag else ''}{stamp}"
    out_csv = out_dir / f"{name}.csv"
    with open(out_dir / f"{name}_system.txt", "w") as f:
        subprocess.run(["bash", str(ROOT / "scripts" / "system_info.sh")], stdout=f, stderr=subprocess.STDOUT)

    variants = [v for v in args.variants.split(",") if v and v != "seq"]

    if args.experiment == "threads":
        run(binary, out_csv, "seq", 1, "static", args, [])
        for v in variants:
            for t in THREADS:
                run(binary, out_csv, v, t, args.schedule, args, [])
    elif args.experiment == "schedule":
        run(binary, out_csv, "seq", 1, "static", args, [])
        for v in variants:
            for s in SCHEDULES:
                run(binary, out_csv, v, args.threads, s, args, [])
    elif args.experiment == "sizes":
        for r in REPLICATES:
            run(binary, out_csv, "seq", 1, "static", args, ["--replicate", str(r)])
            for v in variants:
                for t in THREADS:
                    run(binary, out_csv, v, t, args.schedule, args, ["--replicate", str(r)])
    elif args.experiment == "kmetric":
        for m in METRICS:
            for k in K_VALUES:
                extra = ["--metric", m, "--k", str(k)]
                run(binary, out_csv, "seq", 1, "static", args, extra)
                for v in variants:
                    run(binary, out_csv, v, args.threads, args.schedule, args, extra)

    print(f"\nresults: {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
