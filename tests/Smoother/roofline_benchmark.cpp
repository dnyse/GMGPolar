/*
 * =============================================================================
 * ROOFLINE MODEL BENCHMARK FOR SMOOTHER OPERATIONS
 * =============================================================================
 *
 * Measures wall-clock time of SmootherGive. FLOPs, memory bandwidth, and
 * arithmetic intensity are measured externally via LIKWID hardware counters.
 *
 * =============================================================================
 * BUILDING
 * =============================================================================
 *
 *   ./scripts/compile-openmp.sh Release
 *
 * LIKWID markers are enabled by default via -DLIKWID_PERFMON in compile-openmp.sh.
 *
 * =============================================================================
 * RUNNING
 * =============================================================================
 *
 * SEQUENTIAL (1 core):
 *   likwid-perfctr -C 0 -g FLOPS_DP -m \
 *       ./build/tests/gmgpolar_tests --gtest_filter=RooflineBenchmark.Grid_769x1024_Sequential
 *   likwid-perfctr -C 0 -g MEM_DP -m \
 *       ./build/tests/gmgpolar_tests --gtest_filter=RooflineBenchmark.Grid_769x1024_Sequential
 *
 * PARALLEL (thread count via SMOOTHER_THREADS env var):
 *   export OMP_PROC_BIND=spread
 *   export OMP_PLACES=cores
 *
 *   # Single socket (36 cores):
 *   SMOOTHER_THREADS=36 likwid-perfctr -C 0-35 -g FLOPS_DP -m \
 *       ./build/tests/gmgpolar_tests --gtest_filter=RooflineBenchmark.Grid_769x1024_Parallel
 *
 *   # Full node (72 cores):
 *   SMOOTHER_THREADS=72 likwid-perfctr -C 0-71 -g MEM_DP -m \
 *       ./build/tests/gmgpolar_tests --gtest_filter=RooflineBenchmark.Grid_769x1024_Parallel
 *
 * STRONG SCALING (timing only, no LIKWID needed):
 *   for T in 1 2 4 8 16 36 72; do
 *     SMOOTHER_THREADS=$T ./build/tests/gmgpolar_tests \
 *         --gtest_filter=RooflineBenchmark.StrongScaling_769x1024
 *   done
 *
 * =============================================================================
 * INTERPRETING LIKWID OUTPUT
 * =============================================================================
 *
 * FLOPS_DP group — use the Sum column across all threads:
 *   Total_FLOPs = PMC1*1 + PMC0*2 + PMC2*4 + PMC3*8
 *   FLOPs/point = Total_FLOPs / (num_iterations * grid_nodes)
 *   Performance = Total_FLOPs / RDTSC_Runtime / 1e9  [GFLOP/s]
 *
 * MEM_DP group — LIKWID computes directly:
 *   "Operational intensity [FLOP/Byte]"  -> arithmetic intensity
 *   "Memory bandwidth [MBytes/s]"        -> memory bandwidth
 *
 * =============================================================================
 * PEAK PERFORMANCE REFERENCE (Fritz, Intel Xeon Platinum 8360Y)
 * =============================================================================
 *
 *   1 core  (measured 2.74 GHz turbo): 2.74 * 2 FMA * 8 doubles * 2 =  87.7 GFLOP/s
 *   36 cores (base 2.4 GHz):           2.40 * 36 * 2 * 8 * 2        = 2764.8 GFLOP/s
 *   72 cores (base 2.4 GHz):           2.40 * 72 * 2 * 8 * 2        = 5529.6 GFLOP/s
 *
 *   Memory bandwidth (1 socket, 8x DDR4-3200): ~130 GB/s measured
 *   Ridge point (1 core):   87.7  / 130 = 0.67 FLOP/byte
 *   Ridge point (36 cores): 2764.8 / 130 = 21.3 FLOP/byte
 *
 * =============================================================================
 */

#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <utility>
#include <cstdlib>

#include "../../include/GMGPolar/gmgpolar.h"
#include "../../include/Smoother/SmootherGive/smootherGive.h"
#include "../../include/InputFunctions/domainGeometry.h"
#include "../../include/InputFunctions/densityProfileCoefficients.h"

#include "../include/InputFunctions/DomainGeometry/czarnyGeometry.h"
#include "../include/InputFunctions/DensityProfileCoefficients/zoniShiftedCoefficients.h"

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#endif

namespace RooflineBenchmark
{

/* Read thread count from SMOOTHER_THREADS env var, fall back to default. */
int get_num_threads(int default_threads)
{
    if (const char* env = std::getenv("SMOOTHER_THREADS")) {
        int t = std::atoi(env);
        if (t > 0) return t;
    }
    return default_threads;
}

/* Override iteration/warmup counts via env (SMOOTHER_ITERS / SMOOTHER_WARMUP).
 * Useful for whole-process LIKWID roofline measurement (no -m marker): a large
 * iteration count makes the smoother dominate the one-off setup cost. */
int get_env_int(const char* name, int default_value)
{
    if (const char* env = std::getenv(name)) {
        int v = std::atoi(env);
        if (v > 0) return v;
    }
    return default_value;
}

Vector<double> generate_random_data(const PolarGrid& grid, unsigned int seed)
{
    Vector<double> x("x", grid.numberOfNodes());
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < x.size(); ++i) {
        x(i) = dist(gen);
    }
    return x;
}

struct BenchmarkConfig {
    int nr;
    int ntheta;
    int num_warmup;
    int num_iterations;
    int default_threads;
    std::string marker_name;
};

/*
 * Runs the smoother benchmark. LIKWID markers wrap only the timed iterations.
 * MARKER_INIT and MARKER_CLOSE are called once per invocation — do NOT call
 * this function in a loop within a single test.
 */
void run_benchmark_with_config(const BenchmarkConfig& config)
{
    const double Rmax      = 1.3;
    const double r_min     = 1e-5;
    const double kappa_eps = 0.3;
    const double delta_e   = 1.4;

    int num_threads    = get_num_threads(config.default_threads);
    int num_warmup     = get_env_int("SMOOTHER_WARMUP", config.num_warmup);
    int num_iterations = get_env_int("SMOOTHER_ITERS", config.num_iterations);

    // Build uniform grid
    std::vector<double> radii(config.nr);
    for (int i = 0; i < config.nr; ++i) {
        radii[i] = r_min + i * (Rmax - r_min) / (config.nr - 1);
    }
    std::vector<double> angles(config.ntheta + 1);
    for (int i = 0; i <= config.ntheta; ++i) {
        angles[i] = i * 2 * M_PI / config.ntheta;
    }

    CzarnyGeometry domain_geometry(Rmax, kappa_eps, delta_e);
    double alpha_jump = 0.678 * Rmax;
    std::unique_ptr<DensityProfileCoefficients> coefficients =
        std::make_unique<ZoniShiftedCoefficients>(Rmax, alpha_jump);

    auto grid       = std::make_unique<PolarGrid>(radii, angles);
    auto levelCache = std::make_unique<LevelCache>(*grid, *coefficients, domain_geometry,
                                                   true, false);
    Level level(0, std::move(grid), std::move(levelCache), ExtrapolationType::NONE, 0);

    SmootherGive smoother_op(level.grid(), level.levelCache(), domain_geometry,
                             *coefficients, true, num_threads);

    Vector<double> solution = generate_random_data(level.grid(), 42);
    Vector<double> rhs      = generate_random_data(level.grid(), 69);
    Vector<double> temp     = generate_random_data(level.grid(), 8);

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_INIT;
    LIKWID_MARKER_THREADINIT;
    LIKWID_MARKER_REGISTER(config.marker_name.c_str());
#endif

    // Warmup (outside measurement region)
    for (int i = 0; i < num_warmup; ++i) {
        smoother_op.smoothing(solution, rhs, temp);
    }

    // Timed + measured region
    auto t_start = std::chrono::high_resolution_clock::now();

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START(config.marker_name.c_str());
#endif

    for (int i = 0; i < num_iterations; ++i) {
        smoother_op.smoothing(solution, rhs, temp);
    }

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP(config.marker_name.c_str());
#endif

    auto t_end = std::chrono::high_resolution_clock::now();

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_CLOSE;
#endif

    double elapsed_ms       = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double time_per_iter_ms = elapsed_ms / num_iterations;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Grid: " << config.nr << " x " << config.ntheta
              << "  nodes: " << level.grid().numberOfNodes()
              << "  threads: " << num_threads
              << "  iterations: " << num_iterations
              << "  time/iter: " << time_per_iter_ms << " ms\n";

    // CSV line for scaling plots: grep "TIMING_CSV" from output
    std::cout << "TIMING_CSV,"
              << config.nr << "," << config.ntheta << ","
              << level.grid().numberOfNodes() << ","
              << num_threads << ","
              << num_iterations << ","
              << time_per_iter_ms << "\n";
}

} // namespace RooflineBenchmark

using namespace RooflineBenchmark;

/* ============================================================================
 * SEQUENTIAL BENCHMARKS
 * likwid-perfctr -C 0 -g FLOPS_DP -m ./build/tests/gmgpolar_tests
 *     --gtest_filter=RooflineBenchmark.Grid_769x1024_Sequential
 * ============================================================================ */

TEST(RooflineBenchmark, Grid_769x1024_Sequential)
{
    run_benchmark_with_config({
        .nr = 769, .ntheta = 1024,
        .num_warmup = 5, .num_iterations = 50,
        .default_threads = 1,
        .marker_name = "769x1024_seq"
    });
}

TEST(RooflineBenchmark, Grid_1537x2048_Sequential)
{
    run_benchmark_with_config({
        .nr = 1537, .ntheta = 2048,
        .num_warmup = 3, .num_iterations = 20,
        .default_threads = 1,
        .marker_name = "1537x2048_seq"
    });
}

TEST(RooflineBenchmark, Grid_3073x4096_Sequential)
{
    run_benchmark_with_config({
        .nr = 3073, .ntheta = 4096,
        .num_warmup = 2, .num_iterations = 10,
        .default_threads = 1,
        .marker_name = "3073x4096_seq"
    });
}

/* ============================================================================
 * PARALLEL BENCHMARKS
 *
 *   export OMP_PROC_BIND=spread && export OMP_PLACES=cores
 *   SMOOTHER_THREADS=36 likwid-perfctr -C 0-35 -g FLOPS_DP -m \
 *       ./build/tests/gmgpolar_tests --gtest_filter=RooflineBenchmark.Grid_769x1024_Parallel
 * ============================================================================ */

TEST(RooflineBenchmark, Grid_769x1024_Parallel)
{
    run_benchmark_with_config({
        .nr = 769, .ntheta = 1024,
        .num_warmup = 5, .num_iterations = 50,
        .default_threads = 36,
        .marker_name = "769x1024_par"
    });
}

TEST(RooflineBenchmark, Grid_1537x2048_Parallel)
{
    run_benchmark_with_config({
        .nr = 1537, .ntheta = 2048,
        .num_warmup = 3, .num_iterations = 20,
        .default_threads = 36,
        .marker_name = "1537x2048_par"
    });
}

TEST(RooflineBenchmark, Grid_3073x4096_Parallel)
{
    run_benchmark_with_config({
        .nr = 3073, .ntheta = 4096,
        .num_warmup = 2, .num_iterations = 10,
        .default_threads = 36,
        .marker_name = "3073x4096_par"
    });
}

/* ============================================================================
 * STRONG SCALING — timing only, no LIKWID needed
 *
 *   for T in 1 2 4 8 16 36 72; do
 *     SMOOTHER_THREADS=$T ./build/tests/gmgpolar_tests \
 *         --gtest_filter=RooflineBenchmark.StrongScaling_769x1024
 *   done
 * ============================================================================ */

TEST(RooflineBenchmark, StrongScaling_769x1024)
{
    run_benchmark_with_config({
        .nr = 769, .ntheta = 1024,
        .num_warmup = 5, .num_iterations = 50,
        .default_threads = 1,
        .marker_name = "scaling_769x1024"
    });
}

/* ============================================================================
 * LARGE GRIDS — disabled by default, remove DISABLED_ to run
 * ============================================================================ */

TEST(RooflineBenchmark, DISABLED_Grid_6145x8192_Sequential)
{
    run_benchmark_with_config({
        .nr = 6145, .ntheta = 8192,
        .num_warmup = 1, .num_iterations = 5,
        .default_threads = 1,
        .marker_name = "6145x8192_seq"
    });
}

TEST(RooflineBenchmark, DISABLED_Grid_6145x8192_Parallel)
{
    run_benchmark_with_config({
        .nr = 6145, .ntheta = 8192,
        .num_warmup = 1, .num_iterations = 5,
        .default_threads = 36,
        .marker_name = "6145x8192_par"
    });
}
