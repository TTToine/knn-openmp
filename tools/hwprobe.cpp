// hwprobe - two tiny measurements the performance model needs and that WSL2 does
// not expose (no cpufreq, no MSR, no perf counters):
//
//   hwprobe freq   [seconds]   real clock frequency under load, per thread
//   hwprobe membw  [MB]        sustained read bandwidth from DRAM, all threads
//
// Thread count and pinning come from the OpenMP environment (OMP_NUM_THREADS,
// OMP_PROC_BIND, OMP_PLACES), exactly as for the k-NN runs. Output: CSV lines
//   probe,threads,thread,run,value,unit
// Like every measurement: laptop plugged in, nothing else running.
#include <immintrin.h>
#include <omp.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

// ---- frequency ---------------------------------------------------------------
// A loop whose duration in CYCLES is known, so that time -> cycles/second.
//  * scalar: `imul` has a latency of 3 cycles on every Intel core since Nehalem.
//    N dependent multiplications take 3 N cycles: the out-of-order engine cannot
//    overlap them (each needs the previous result) and nothing else is on the
//    critical path (dec/jnz is a separate 1-cycle chain).
//  * avx: `vfmadd231ps` on ymm registers, latency 4 cycles (Skylake family).
//    Same idea, but it runs the 256-bit FMA units: this is the frequency the CPU
//    really grants to our SIMD kernels (it may be lower than the scalar one).
// Written in assembly so the compiler cannot unroll, reorder or vectorize it.
static double chain_scalar(std::uint64_t n) {   // returns seconds
    std::uint64_t x = 3, c = 0x9E3779B97F4A7C15ull;
    const double t0 = omp_get_wtime();
    __asm__ volatile("1:\n\timul %[c], %[x]\n\tdec %[n]\n\tjnz 1b"
                     : [x] "+r"(x), [n] "+r"(n) : [c] "r"(c) : "cc");
    return omp_get_wtime() - t0;
}

static double chain_avx(std::uint64_t n) {
    __m256 acc = _mm256_set1_ps(1.0f);
    const __m256 a = _mm256_set1_ps(0.5f);   // acc = a*a + acc
    // acc grows by 0.25 per step until the float saturates and stops changing:
    // no overflow, no denormals, and the FMA latency does not depend on the data.
    const double t0 = omp_get_wtime();
    __asm__ volatile("1:\n\tvfmadd231ps %t[a], %t[a], %t[acc]\n\tdec %[n]\n\tjnz 1b"
                     : [acc] "+x"(acc), [n] "+r"(n) : [a] "x"(a) : "cc");
    return omp_get_wtime() - t0;
}

static void probe_freq(double seconds) {
    const int threads = omp_get_max_threads();
    // ~4.5e9 cycles/s assumed only to size the loop; the result does not depend on it.
    const auto n_scalar = static_cast<std::uint64_t>(seconds * 4.5e9 / 3);
    const auto n_avx = static_cast<std::uint64_t>(seconds * 4.5e9 / 4);
    for (int run = 0; run < 3; ++run) {
        std::vector<double> fs(threads), fa(threads);
#pragma omp parallel default(none) shared(fs, fa, n_scalar, n_avx)
        {
            const int t = omp_get_thread_num();
#pragma omp barrier
            fs[t] = 3.0 * static_cast<double>(n_scalar) / chain_scalar(n_scalar) / 1e9;
#pragma omp barrier
            fa[t] = 4.0 * static_cast<double>(n_avx) / chain_avx(n_avx) / 1e9;
        }
        for (int t = 0; t < threads; ++t) {
            std::printf("freq_scalar,%d,%d,%d,%.3f,GHz\n", threads, t, run, fs[t]);
            std::printf("freq_avx2,%d,%d,%d,%.3f,GHz\n", threads, t, run, fa[t]);
        }
    }
}

// ---- memory bandwidth ----------------------------------------------------------
// STREAM-like read test: every thread sums its own contiguous slice of a big
// float array (first touched by the same thread). 4 independent 256-bit
// accumulators: the loop needs 32 bytes/cycle from memory, far more than DRAM
// can deliver, so the measured GB/s is the memory bandwidth, not the adder.
static void probe_membw(std::size_t mb) {
    const int threads = omp_get_max_threads();
    const std::size_t n = mb * 1024 * 1024 / sizeof(float) / 32 * 32;
    float* a = static_cast<float*>(::operator new(n * sizeof(float), std::align_val_t(64)));
#pragma omp parallel for schedule(static) default(none) shared(a, n)
    for (std::size_t i = 0; i < n; ++i) a[i] = 1.0f;   // first touch = same thread that reads

    for (int run = 0; run < 7; ++run) {
        double total = 0.0;
        const double t0 = omp_get_wtime();
#pragma omp parallel default(none) shared(a, n) reduction(+ : total)
        {
            const int t = omp_get_thread_num(), p = omp_get_num_threads();
            const std::size_t lo = n / 32 * t / p * 32, hi = n / 32 * (t + 1) / p * 32;
            __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
            for (std::size_t i = lo; i < hi; i += 32) {
                s0 = _mm256_add_ps(s0, _mm256_load_ps(a + i));
                s1 = _mm256_add_ps(s1, _mm256_load_ps(a + i + 8));
                s2 = _mm256_add_ps(s2, _mm256_load_ps(a + i + 16));
                s3 = _mm256_add_ps(s3, _mm256_load_ps(a + i + 24));
            }
            alignas(32) float buf[8];
            _mm256_store_ps(buf, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
            for (float v : buf) total += v;
        }
        const double dt = omp_get_wtime() - t0;
        if (total != static_cast<double>(n)) std::fprintf(stderr, "membw: checksum %.0f != %zu\n", total, n);
        std::printf("membw_read,%d,-1,%d,%.2f,GB/s\n", threads, run,
                    static_cast<double>(n) * sizeof(float) / dt / 1e9);
    }
    ::operator delete(a, std::align_val_t(64));
}

int main(int argc, char** argv) {
    if (argc >= 2 && !std::strcmp(argv[1], "freq")) {
        probe_freq(argc >= 3 ? std::atof(argv[2]) : 1.0);
    } else if (argc >= 2 && !std::strcmp(argv[1], "membw")) {
        probe_membw(argc >= 3 ? std::strtoul(argv[2], nullptr, 10) : 1024);
    } else {
        std::fprintf(stderr, "usage: hwprobe freq [seconds] | hwprobe membw [MB]\n");
        return 1;
    }
    return 0;
}
