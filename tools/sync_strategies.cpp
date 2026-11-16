// Stand-alone experiment on SYNCHRONISATION strategies (assignment: "critical
// vs atomic vs reduction, false sharing mitigation").
//
// The k-NN classifier itself needs no synchronisation: every query writes its
// own out[i]. The step that DOES share state between threads is the
// evaluation: the confusion matrix cm[true][predicted], a histogram that every
// thread updates. It is tiny compared with the distance computation, so here it
// is isolated and repeated on N (label, prediction) pairs to make the cost of
// each strategy visible:
//
//   serial     one thread, the reference (also validates the others);
//   critical   #pragma omp critical around the increment: one global lock,
//              the updates are serialised;
//   atomic     #pragma omp atomic: one `lock add` per update, no lock, but
//              the cache line of the cell still bounces between cores
//              (true sharing: the threads really update the same counter);
//   reduction  reduction(+ : cm[:C*C]): each thread gets a private zeroed copy
//              on its stack, summed into the shared matrix at the end;
//   packed     private counters written by hand, but stored INTERLEAVED:
//              counters[cell][thread]. The counters of different threads for
//              the same cell sit in the same cache line: no data race, every
//              thread touches only its own counters, and yet each increment
//              invalidates the line in the other cores = FALSE SHARING;
//   padded     private counters, one matrix per thread aligned to 64 bytes
//              (alignas): no cache line is written by two threads.
//
// packed and padded execute the same instructions on the same amount of data:
// the only difference is the memory layout. Their time ratio is the cost of
// false sharing.
//
// Usage: sync_strategies [N=20000000] [runs=5]     (threads: OMP_NUM_THREADS)
// Output: CSV rows  strategy,threads,run,seconds,Mupdates_per_s  + a check line.
#include <omp.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace {
constexpr int C = 10;            // classes (MNIST digits)
constexpr int CELLS = C * C;

using Matrix = std::vector<long>;   // C*C counters, row = true label, column = prediction

Matrix cm_serial(const std::uint8_t* y, const std::uint8_t* p, long n) {
    Matrix cm(CELLS, 0);
    for (long i = 0; i < n; ++i) ++cm[y[i] * C + p[i]];
    return cm;
}

Matrix cm_critical(const std::uint8_t* y, const std::uint8_t* p, long n) {
    Matrix cm(CELLS, 0);
    long* m = cm.data();
#pragma omp parallel for schedule(static) default(none) shared(y, p, n, m)
    for (long i = 0; i < n; ++i) {
#pragma omp critical(cm_update)
        ++m[y[i] * C + p[i]];
    }
    return cm;
}

Matrix cm_atomic(const std::uint8_t* y, const std::uint8_t* p, long n) {
    Matrix cm(CELLS, 0);
    long* m = cm.data();
#pragma omp parallel for schedule(static) default(none) shared(y, p, n, m)
    for (long i = 0; i < n; ++i) {
#pragma omp atomic update
        ++m[y[i] * C + p[i]];
    }
    return cm;
}

Matrix cm_reduction(const std::uint8_t* y, const std::uint8_t* p, long n) {
    Matrix cm(CELLS, 0);
    long* m = cm.data();
    // Array-section reduction (OpenMP 4.5): private copy per thread, combined at the end.
#pragma omp parallel for schedule(static) default(none) shared(y, p, n) reduction(+ : m[:CELLS])
    for (long i = 0; i < n; ++i) ++m[y[i] * C + p[i]];
    return cm;
}

// Private counters, interleaved: counter of thread t for cell c at [c * T + t].
Matrix cm_packed(const std::uint8_t* y, const std::uint8_t* p, long n) {
    const int T = omp_get_max_threads();
    std::vector<long> counters(static_cast<std::size_t>(CELLS) * T, 0);
    long* cnt = counters.data();
#pragma omp parallel default(none) shared(y, p, n, cnt, T)
    {
        const int t = omp_get_thread_num();
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i) ++cnt[(y[i] * C + p[i]) * T + t];
    }
    Matrix cm(CELLS, 0);
    for (int c = 0; c < CELLS; ++c)
        for (int t = 0; t < T; ++t) cm[c] += counters[static_cast<std::size_t>(c) * T + t];
    return cm;
}

// Private counters, one cache-line-aligned matrix per thread. sizeof(Padded)
// is rounded up to a multiple of 64, so two threads never share a line.
struct alignas(64) Padded {
    long cell[CELLS];
};

Matrix cm_padded(const std::uint8_t* y, const std::uint8_t* p, long n) {
    const int T = omp_get_max_threads();
    Padded* per_thread = new (std::align_val_t(64)) Padded[T]();   // zero-initialised
#pragma omp parallel default(none) shared(y, p, n, per_thread)
    {
        long* mine = per_thread[omp_get_thread_num()].cell;
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i) ++mine[y[i] * C + p[i]];
    }
    Matrix cm(CELLS, 0);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < CELLS; ++c) cm[c] += per_thread[t].cell[c];
    ::operator delete[](per_thread, std::align_val_t(64));
    return cm;
}

struct Strategy {
    const char* name;
    Matrix (*fn)(const std::uint8_t*, const std::uint8_t*, long);
};
}  // namespace

int main(int argc, char** argv) {
    const long n = argc > 1 ? std::atol(argv[1]) : 20000000L;
    const int runs = argc > 2 ? std::atoi(argv[2]) : 5;
    const int threads = omp_get_max_threads();

    // Synthetic labels with the statistics of a 97%-accurate classifier: the
    // 10 diagonal cells receive almost all the updates (the contended case).
    std::vector<std::uint8_t> y(n), p(n);
    std::uint64_t s = 88172645463325252ull;   // xorshift64: fixed seed, same data every run
    auto next = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (long i = 0; i < n; ++i) {
        const std::uint64_t r = next();
        y[i] = static_cast<std::uint8_t>(r % C);
        p[i] = ((r >> 20) % 100 < 97) ? y[i] : static_cast<std::uint8_t>((r >> 40) % C);
    }

    const Matrix ref = cm_serial(y.data(), p.data(), n);
    const Strategy strategies[] = {{"serial", cm_serial},   {"critical", cm_critical},
                                   {"atomic", cm_atomic},   {"reduction", cm_reduction},
                                   {"packed", cm_packed},   {"padded", cm_padded}};
    bool all_ok = true;
    for (const Strategy& st : strategies) {
        // critical is ~100x slower than the others: fewer updates, same rate metric
        const long m = std::strcmp(st.name, "critical") == 0 ? n / 20 : n;
        const Matrix expect = (m == n) ? ref : cm_serial(y.data(), p.data(), m);
        for (int r = -1; r < runs; ++r) {   // r = -1: warm-up, not printed
            const double t0 = omp_get_wtime();
            const Matrix cm = st.fn(y.data(), p.data(), m);
            const double dt = omp_get_wtime() - t0;
            if (cm != expect) all_ok = false;
            if (r >= 0)
                std::printf("%s,%d,%d,%.6f,%.2f\n", st.name, std::strcmp(st.name, "serial") == 0 ? 1 : threads,
                            r, dt, static_cast<double>(m) / dt / 1e6);
        }
    }
    std::fprintf(stderr, "threads=%d N=%ld: %s\n", threads, n,
                 all_ok ? "all strategies match the serial matrix" : "MISMATCH");
    return all_ok ? 0 : 2;
}
