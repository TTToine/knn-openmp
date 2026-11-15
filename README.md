# Parallel k-NN classifier (OpenMP + SIMD)

Mid-term assignment for the *Parallel Programming* course (Università di Firenze).
Brute-force k-nearest-neighbours classifier on MNIST: a sequential reference and
several OpenMP/vectorized variants, with a performance analysis (speedup, scaling,
scheduling, memory layout).

## Build (Linux / WSL2, GCC ≥ 9, CMake ≥ 3.16)
```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```
CMake options: `KNN_NATIVE` (default ON, `-march=native`), `KNN_NO_VEC`,
`KNN_VEC_REPORT`, `KNN_TSAN`.

## Data
```bash
bash scripts/download_mnist.sh      # -> data/mnist (IDX files)
```
MNIST: Y. LeCun, C. Cortes, C. J. C. Burges. A synthetic generator
(`--synthetic`) is also available for testing.

## Run
```bash
./build/knn --list
OMP_SCHEDULE=static ./build/knn --data data/mnist --n-test 500 \
    --variant omp_naive --threads 6 --k 5 --metric euclidean --runs 5 --csv results/raw/run.csv
```
Main parameters:

| option | meaning | default |
|---|---|---|
| `--variant` | implementation (`--list`) | `seq` |
| `--threads` | OpenMP threads | all |
| `--k` | neighbours | 5 |
| `--metric` | `euclidean` / `cosine` | euclidean |
| `--n-train`, `--n-test` | subset of MNIST (0 = all) | 0 |
| `--replicate` | replicate the training set | 1 |
| `--runs`, `--warmup` | measured / discarded runs | 5 / 1 |

Every run is validated against the sequential reference (number of differing
predictions is reported). Only the classification is timed (`omp_get_wtime`).

## Benchmarks and plots
```bash
python3 scripts/run_benchmarks.py threads  --variants omp_naive
python3 scripts/run_benchmarks.py schedule --variants omp_naive --threads 12
python3 scripts/plot_results.py results/raw/*.csv
```

## Repository layout
```
include/   headers (dataset, k-NN API, shared kernels)
src/       sequential reference, OpenMP variants, variant registry, driver
scripts/   benchmark sweeps, plotting, system info, dataset download
tests/     smoke test validating every variant
results/   raw CSVs (committed) and plots
report/    lab notes and report outline
```
