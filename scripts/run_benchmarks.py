#!/usr/bin/env python3
"""Runs a benchmark experiment and appends the results to a CSV in results/raw/.

Each experiment is a sweep over one dimension (threads, schedule, k/metric,
input size, tile size, ...). The sequential baseline is measured with the same
data, so plot_results.py can compute the speedup.

Methodology built in:
  * threads pinned (OMP_PROC_BIND=close, OMP_PLACES=cores): one thread per
    physical core up to 6, hyperthreads used only beyond that;
  * OMP_SCHEDULE always set explicitly (libgomp default would be dynamic,1);
  * the order of the variants is rotated at every sweep point, so thermal
    drift of the laptop does not always penalise the same variant;
  * default --n-test 384: 24 query blocks of 16 for omp_blocked, divisible by
    1, 2, 4, 6, 8, 12 threads (no load-imbalance artefacts), and the sequential
    run takes > 10 s.

Examples:
  python3 scripts/run_benchmarks.py threads  --variants omp_naive,omp_simd,omp_blocked
  python3 scripts/run_benchmarks.py schedule --variants omp_naive --threads 12
  python3 scripts/run_benchmarks.py kmetric  --variants omp_blocked --threads 12
  python3 scripts/run_benchmarks.py tiles    --variants omp_blocked --threads 6
  python3 scripts/run_benchmarks.py cachefit --variants omp_naive,omp_simd
  python3 scripts/run_benchmarks.py vec      --variants omp_naive,omp_simd
  python3 scripts/run_benchmarks.py sizes    --variants omp_naive
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
TILE_TB = [8, 16, 32, 64, 128, 256]      # training rows per tile (x 3136 B)
TILE_QB = [4, 8, 16, 32]                 # queries per block (384 / QB blocks)
# n_train : n_test pairs. 64 rows = 200 KB (fits L2), 2000 = 6.3 MB (fits L3),
# 60000 = 188 MB (DRAM). n_test keeps every run around 0.5-5 s at 1 thread.
CACHEFIT = [(64, 10000), (2000, 2000), (60000, 100)]


def rotated(items, i):
    """items rotated by i positions: alternates the run order between sweep points."""
    i %= max(len(items), 1)
    return items[i:] + items[:i]


def on_ac_power():
    """True/False if WSL exposes the mains adapter (/sys/class/power_supply), else None.
    On battery Windows lowers the CPU power limits: timings become 15%-2x slower."""
    found = None
    for ps in Path("/sys/class/power_supply").glob("*"):
        try:
            if (ps / "type").read_text().strip() == "Mains":
                found = found or (ps / "online").read_text().strip() == "1"
        except OSError:
            pass
    return found


def run(binary, out_csv, variant, threads, schedule, args, extra, tag=None, env_extra=None):
    if on_ac_power() is False and not args.allow_battery:
        sys.exit("ABORT: laptop on battery - plug it in (or pass --allow-battery). "
                 f"Results so far are in {out_csv}")
    tag = args.tag if tag is None else tag
    cmd = [str(binary), "--variant", variant, "--threads", str(threads),
           "--runs", str(args.runs), "--warmup", str(args.warmup),
           "--csv", str(out_csv), "--tag", tag] + data_args(args) + extra
    # Places = physical cores. Up to 6 threads: one thread per core. Beyond, close
    # puts 2 threads on some cores (hyperthreads) and 1 on the others. NB: with
    # OMP_PLACES=threads + close, 8 threads would land on CPUs 0-7 = only 4 cores
    # (siblings are 0-1, 2-3, ... on this machine, see `lscpu -e`).
    env = dict(os.environ, OMP_SCHEDULE=schedule, OMP_PROC_BIND="close", OMP_PLACES="cores")
    env.update(env_extra or {})
    shown = " ".join(extra + [f"{k}={v}" for k, v in (env_extra or {}).items()])
    print(f"[{dt.datetime.now():%H:%M:%S}] {binary.parent.name}/{variant:12s} t={threads:<3d} "
          f"sched={schedule:11s} {shown}", flush=True)
    res = subprocess.run(cmd, env=env, cwd=ROOT, text=True, capture_output=True)
    print("   ", res.stdout.strip(), flush=True)
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
    p.add_argument("experiment",
                   choices=["threads", "schedule", "sizes", "kmetric", "tiles", "cachefit", "vec"])
    p.add_argument("--variants", default="omp_naive", help="comma-separated parallel variants")
    p.add_argument("--binary", default="build/knn")
    p.add_argument("--novec-binary", default="build-novec/knn", help="vec experiment only")
    p.add_argument("--tag", default="", help="label stored in the CSV (e.g. novec)")
    p.add_argument("--data", default="data/mnist")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n-train", type=int, default=60000, help="synthetic only")
    p.add_argument("--n-test", type=int, default=384,
                   help="384 = 24 blocks of 16 queries; sequential run > 10 s")
    p.add_argument("--threads", type=int, default=12, help="thread count for non-thread sweeps")
    p.add_argument("--schedule", default="static", help="schedule for non-schedule sweeps")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--no-seq", action="store_true", help="skip the sequential baseline run")
    p.add_argument("--allow-battery", action="store_true", help="do not abort when on battery")
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
    seq = (lambda extra=(), tag=None: None) if args.no_seq else \
        (lambda extra=(), tag=None: run(binary, out_csv, "seq", 1, "static", args, list(extra), tag))

    if args.experiment == "threads":
        seq()
        for i, t in enumerate(THREADS):
            for v in rotated(variants, i):
                run(binary, out_csv, v, t, args.schedule, args, [])
    elif args.experiment == "schedule":
        seq()
        for i, s in enumerate(SCHEDULES):
            for v in rotated(variants, i):
                run(binary, out_csv, v, args.threads, s, args, [])
    elif args.experiment == "sizes":
        for r in REPLICATES:
            seq(["--replicate", str(r)])
            for i, t in enumerate(THREADS):
                for v in rotated(variants, i):
                    run(binary, out_csv, v, t, args.schedule, args, ["--replicate", str(r)])
    elif args.experiment == "kmetric":
        i = 0
        for m in METRICS:
            for k in K_VALUES:
                extra = ["--metric", m, "--k", str(k)]
                seq(extra)
                for v in rotated(variants, i):
                    run(binary, out_csv, v, args.threads, args.schedule, args, extra)
                i += 1
    elif args.experiment == "tiles":
        # No seq run: the comparison is between tile sizes of the same variant.
        i = 0
        for qb in TILE_QB:
            for tb in rotated(TILE_TB, i):
                for v in variants:
                    run(binary, out_csv, v, args.threads, args.schedule, args, [],
                        tag=f"{args.tag}tb{tb}_qb{qb}", env_extra={"KNN_TB": str(tb), "KNN_QB": str(qb)})
                i += 1
    elif args.experiment == "cachefit":
        # 1 thread and the --threads value; training sizes from L2 to DRAM.
        i = 0
        for ntr, nte in CACHEFIT:
            for t in sorted({1, args.threads}):
                for v in rotated(variants, i):
                    run(binary, out_csv, v, t, args.schedule, args,
                        ["--n-test", str(nte), "--n-train", str(ntr)], tag=f"{args.tag}cachefit")
                i += 1
    elif args.experiment == "vec":
        # Same variants from the normal build and from build-novec (-fno-tree-vectorize),
        # 1 thread, training in L2 and in DRAM. Binary order alternates as well.
        novec = ROOT / args.novec_binary
        if not novec.exists():
            sys.exit(f"{novec} not found - cmake -S . -B build-novec -DKNN_NO_VEC=ON && cmake --build build-novec -j")
        builds = [(binary, f"{args.tag}vec"), (novec, f"{args.tag}novec")]
        i = 0
        for ntr, nte in [CACHEFIT[0], CACHEFIT[2]]:
            for b, tag in rotated(builds, i):
                for v in rotated(variants, i):
                    run(b, out_csv, v, 1, args.schedule, args,
                        ["--n-test", str(nte), "--n-train", str(ntr)], tag=tag)
                i += 1

    print(f"\nresults: {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
