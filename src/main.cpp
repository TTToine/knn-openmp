// k-NN benchmark driver.
//
// Loads (or generates) the data, runs ONE variant `warmup + runs` times,
// validates its predictions against the sequential reference and appends one
// CSV row per measured run. Only the classification call is timed (no I/O).
#include <omp.h>
#include <time.h>   // clock_gettime(CLOCK_PROCESS_CPUTIME_ID)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "dataset.hpp"
#include "knn.hpp"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string data_dir;              // MNIST directory (IDX files)
    bool synthetic = false;
    std::size_t n_train = 0, n_test = 0;   // 0 = all (MNIST) ; required for synthetic
    std::size_t dim = 784;
    int classes = 10;
    std::uint32_t seed = 42;
    std::size_t replicate = 1;
    KnnParams params;
    std::string variant = "seq";
    int threads = 0;                   // 0 = OpenMP default
    int runs = 5, warmup = 1;
    std::string csv;
    std::string tag;
    bool validate = true;
    double max_mismatch_frac = 0.0;
    std::string cache_dir = ".cache";
};

void usage() {
    std::cout <<
R"(Usage: knn [options]
Data:
  --data DIR            MNIST directory with the 4 uncompressed IDX files
  --synthetic           use Gaussian blobs instead (needs --n-train/--n-test)
  --n-train N           training samples (MNIST: take the first N; 0 = all)
  --n-test N            test samples (MNIST: take the first N; 0 = all)
  --dim D               synthetic dimensionality (default 784)
  --classes C           synthetic classes (default 10)
  --seed S              synthetic seed (default 42)
  --replicate R         replicate the training set R times (default 1)
k-NN:
  --k K                 neighbours (default 5)
  --metric M            euclidean | cosine (default euclidean)
Run:
  --variant NAME        implementation to run (see --list)
  --threads T           OpenMP threads (default: OMP_NUM_THREADS / all)
  --runs R              measured runs (default 5)
  --warmup W            discarded warm-up runs (default 1)
  --csv FILE            append one row per measured run
  --tag STR             free label stored in the CSV (e.g. "novec")
  --no-validate         skip the check against the sequential reference
  --max-mismatch F      tolerated fraction of differing predictions (default 0)
  --cache-dir DIR       where reference predictions are cached (default .cache)
  --list                list registered variants
Scheduling for schedule(runtime) loops: set OMP_SCHEDULE, e.g. "dynamic,16".
)";
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (a == "--data") o.data_dir = next();
        else if (a == "--synthetic") o.synthetic = true;
        else if (a == "--n-train") o.n_train = std::stoul(next());
        else if (a == "--n-test") o.n_test = std::stoul(next());
        else if (a == "--dim") o.dim = std::stoul(next());
        else if (a == "--classes") o.classes = std::stoi(next());
        else if (a == "--seed") o.seed = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--replicate") o.replicate = std::stoul(next());
        else if (a == "--k") o.params.k = std::stoi(next());
        else if (a == "--metric") {
            if (!parse_metric(next(), o.params.metric)) { std::cerr << "unknown metric\n"; return false; }
        }
        else if (a == "--variant") o.variant = next();
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--runs") o.runs = std::stoi(next());
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--csv") o.csv = next();
        else if (a == "--tag") o.tag = next();
        else if (a == "--no-validate") o.validate = false;
        else if (a == "--max-mismatch") o.max_mismatch_frac = std::stod(next());
        else if (a == "--cache-dir") o.cache_dir = next();
        else if (a == "--list") {
            for (const auto& v : all_variants())
                std::cout << v.name << "\t" << v.description << "\n";
            std::exit(0);
        }
        else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        else { std::cerr << "unknown option " << a << "\n"; usage(); return false; }
    }
    if (o.params.k < 1 || o.params.k > kMaxK) { std::cerr << "k must be in [1, " << kMaxK << "]\n"; return false; }
    if (!o.synthetic && o.data_dir.empty()) { std::cerr << "use --data DIR or --synthetic\n"; return false; }
    if (o.synthetic && (o.n_train == 0 || o.n_test == 0)) { std::cerr << "--synthetic needs --n-train and --n-test\n"; return false; }
    return true;
}

std::string schedule_string() {
    omp_sched_t kind;
    int chunk;
    omp_get_schedule(&kind, &chunk);
    std::string s;
    switch (static_cast<int>(kind) & 0x7fffffff) {   // strip the monotonic modifier bit
        case omp_sched_static: s = "static"; break;
        case omp_sched_dynamic: s = "dynamic"; break;
        case omp_sched_guided: s = "guided"; break;
        case omp_sched_auto: s = "auto"; break;
        default: s = "other";
    }
    if (chunk > 0) s += "-" + std::to_string(chunk);   // '-' instead of ',' keeps the CSV simple
    return s;
}

// Key identifying a (data, k, metric) configuration for the reference cache.
std::string config_key(const Options& o, const Dataset& train, const Dataset& test) {
    std::ostringstream k;
    k << (o.synthetic ? "syn" + std::to_string(o.seed) + "d" + std::to_string(o.dim) : "mnist")
      << "_tr" << train.n << "_te" << test.n << "_k" << o.params.k << "_" << metric_name(o.params.metric);
    return k.str();
}

bool load_preds(const fs::path& p, std::vector<std::uint8_t>& v) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size()));
    return static_cast<bool>(f);
}

void save_preds(const fs::path& p, const std::vector<std::uint8_t>& v) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()));
}

double accuracy(const std::vector<std::uint8_t>& pred, const std::vector<std::uint8_t>& truth) {
    std::size_t ok = 0;
    for (std::size_t i = 0; i < pred.size(); ++i) ok += (pred[i] == truth[i]);
    return static_cast<double>(ok) / static_cast<double>(pred.size());
}

// CPU time consumed by ALL threads of the process (user + system), in seconds.
// Wall-clock says how long we waited; CPU time says how much work the cores did:
// CPU / wall = average number of busy cores (ideal = number of threads).
// NB: OpenMP threads that spin-wait at barriers also accumulate CPU time.
double process_cpu_seconds() {
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) return 1;

    const Variant* var = find_variant(o.variant);
    if (!var) { std::cerr << "unknown variant '" << o.variant << "' (try --list)\n"; return 1; }
    if (o.threads > 0) omp_set_num_threads(o.threads);
    const int threads = var->parallel ? omp_get_max_threads() : 1;

    // ---- data (not timed) ----
    Dataset train, test;
    try {
        if (o.synthetic) {
            // Same class centres, different samples for train and test.
            train = make_synthetic(o.n_train, o.dim, o.classes, o.seed, o.seed + 1);
            test = make_synthetic(o.n_test, o.dim, o.classes, o.seed, o.seed + 2);
        } else {
            const fs::path dir(o.data_dir);
            train = load_idx((dir / "train-images-idx3-ubyte").string(),
                             (dir / "train-labels-idx1-ubyte").string(), o.n_train);
            test = load_idx((dir / "t10k-images-idx3-ubyte").string(),
                            (dir / "t10k-labels-idx1-ubyte").string(), o.n_test);
        }
    } catch (const std::exception& e) {
        std::cerr << "data error: " << e.what() << "\n";
        return 1;
    }
    train = replicate(train, o.replicate);

    // schedule(runtime) only matters for parallel variants. NB: if OMP_SCHEDULE is
    // unset, libgomp defaults to dynamic,1 -> always set it explicitly.
    const std::string sched = var->parallel ? schedule_string() : "none";
    std::cerr << "variant=" << var->name << " threads=" << threads << " schedule=" << sched
              << " train=" << train.n << "x" << train.d << " test=" << test.n
              << " k=" << o.params.k << " metric=" << metric_name(o.params.metric) << "\n";

    // ---- runs (only the knn call is timed) ----
    std::vector<std::uint8_t> pred(test.n);
    std::vector<double> times, cpu_times;
    for (int r = 0; r < o.warmup + o.runs; ++r) {
        const double c0 = process_cpu_seconds();
        const double t0 = omp_get_wtime();
        var->fn(train, test, o.params, pred.data());
        const double t1 = omp_get_wtime();
        const double c1 = process_cpu_seconds();
        if (r >= o.warmup) {
            times.push_back(t1 - t0);
            cpu_times.push_back(c1 - c0);
        }
    }

    // ---- validation against the sequential reference ----
    long mismatches = -1;
    bool valid = true;
    if (o.validate) {
        const fs::path ref_path = fs::path(o.cache_dir) / ("ref_" + config_key(o, train, test) + ".bin");
        std::vector<std::uint8_t> ref(test.n);
        if (var->name == "seq") {
            ref = pred;
            save_preds(ref_path, ref);
        } else if (!load_preds(ref_path, ref)) {
            std::cerr << "computing reference predictions (cached in " << ref_path << ")...\n";
            knn_sequential(train, test, o.params, ref.data());
            save_preds(ref_path, ref);
        }
        mismatches = 0;
        for (std::size_t i = 0; i < test.n; ++i) mismatches += (pred[i] != ref[i]);
        const double frac = static_cast<double>(mismatches) / static_cast<double>(test.n);
        valid = frac <= o.max_mismatch_frac;
        if (!valid)
            std::cerr << "VALIDATION FAILED: " << mismatches << " predictions differ from the reference\n";
    }

    // ---- report ----
    const double acc = accuracy(pred, test.y);
    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double var_acc = 0.0;
    for (double t : times) var_acc += (t - mean) * (t - mean);
    const double stddev = times.size() > 1 ? std::sqrt(var_acc / (times.size() - 1)) : 0.0;
    const double median = sorted[sorted.size() / 2];

    std::vector<double> cpu_sorted = cpu_times;
    std::sort(cpu_sorted.begin(), cpu_sorted.end());
    const double cpu_median = cpu_sorted[cpu_sorted.size() / 2];

    std::printf("%-12s threads=%-3d median=%.4fs mean=%.4fs std=%.4fs min=%.4fs max=%.4fs cpu=%.4fs acc=%.4f mismatches=%ld\n",
                var->name.c_str(), threads, median, mean, stddev, sorted.front(), sorted.back(),
                cpu_median, acc, mismatches);

    if (!o.csv.empty()) {
        const bool new_file = !fs::exists(o.csv);
        if (fs::path(o.csv).has_parent_path()) fs::create_directories(fs::path(o.csv).parent_path());
        std::ofstream f(o.csv, std::ios::app);
        // cpu_s is the last column so that older CSVs (without it) still line up.
        if (new_file)
            f << "tag,variant,threads,schedule,k,metric,source,n_train,n_test,dim,replicate,run,time_s,accuracy,mismatches,cpu_s\n";
        for (std::size_t r = 0; r < times.size(); ++r)
            f << o.tag << ',' << var->name << ',' << threads << ',' << sched << ','
              << o.params.k << ',' << metric_name(o.params.metric) << ','
              << (o.synthetic ? "synthetic" : "mnist") << ',' << train.n << ',' << test.n << ','
              << train.d << ',' << o.replicate << ',' << r << ',' << times[r] << ',' << acc << ','
              << mismatches << ',' << cpu_times[r] << '\n';
    }
    return valid ? 0 : 2;
}
