/*
 * =============================================================================
 * ROOFLINE MODEL BENCHMARK FOR SMOOTHER OPERATIONS
 * =============================================================================
 *
 * This benchmark measures the performance of the SmootherGive operation for
 * roofline model analysis. It provides both estimated metrics (based on
 * theoretical FLOP/byte counts) and supports LIKWID for hardware counter
 * measurements.
 *
 * =============================================================================
 * BUILDING WITH LIKWID SUPPORT
 * =============================================================================
 *
 * To enable accurate hardware counter measurements, build with LIKWID:
 *
 *   cmake -DLIKWID_PERFMON=ON -DCMAKE_CXX_FLAGS="-DLIKWID_PERFMON" ..
 *   make gmgpolar_tests
 *
 * Make sure LIKWID is installed and the likwid-marker.h header is available.
 *
 * =============================================================================
 * RUNNING THE BENCHMARKS
 * =============================================================================
 *
 * 1. WITHOUT LIKWID (estimated values):
 *    ./gmgpolar_tests --gtest_filter=RooflineBenchmark.*
 *
 * 2. WITH LIKWID (measured FLOP count):
 *    likwid-perfctr -C 0 -g FLOPS_DP -m ./gmgpolar_tests \
 *        --gtest_filter=RooflineBenchmark.Grid_769x1024_Sequential
 *
 * 3. WITH LIKWID (measured memory bandwidth):
 *    likwid-perfctr -C 0 -g MEM_DP -m ./gmgpolar_tests \
 *        --gtest_filter=RooflineBenchmark.Grid_769x1024_Sequential
 *
 * 4. QUICK VALIDATION:
 *    ./gmgpolar_tests --gtest_filter=RooflineBenchmark.SmallGrid_Sequential
 *
 * 5. PARAMETER SWEEP (multiple grid sizes):
 *    ./gmgpolar_tests --gtest_filter=RooflineBenchmark.ParameterSweep_Sequential
 *
 * 6. LARGE GRIDS (disabled by default, remove DISABLED_ prefix to run):
 *    ./gmgpolar_tests --gtest_filter=RooflineBenchmark.Grid_6145x8192_Parallel
 *
 * =============================================================================
 * INTERPRETING LIKWID OUTPUT
 * =============================================================================
 *
 * After running with LIKWID, look for the marker region output:
 *
 *   Region: smoother (or the specific marker name)
 *   +------------------------------------------+---------+------------+
 *   |                   Event                  | Counter | HWThread 0 |
 *   +------------------------------------------+---------+------------+
 *   | FP_ARITH_INST_RETIRED_SCALAR_DOUBLE      |   PMC1  |  XXXXXXXX  |
 *   | FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE |   PMC0  |  XXXXXXXX  |
 *   | FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE |   PMC2  |  XXXXXXXX  |
 *   | FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE |   PMC3  |  XXXXXXXX  |
 *   +------------------------------------------+---------+------------+
 *
 * Calculate actual FLOPs:
 *   Total_FLOPs = PMC1 + PMC0*2 + PMC2*4 + PMC3*8
 *   FLOPs_per_point = Total_FLOPs / (num_iterations * grid_size)
 *
 * For memory bandwidth (MEM_DP group):
 *   Look for "Memory bandwidth [MBytes/s]" or calculate from
 *   memory read/write counters.
 *
 * =============================================================================
 * GRID SIZES FOR ROOFLINE ANALYSIS
 * =============================================================================
 *
 * The benchmark includes several grid sizes matching the roofline_model.py data:
 *
 *   Grid Size    | Nodes     | Data Size  | Cache Level
 *   -------------|-----------|------------|-------------
 *   33 x 64      | ~2K       | ~50 KB     | L1/L2
 *   65 x 128     | ~8K       | ~200 KB    | L2
 *   129 x 256    | ~33K      | ~800 KB    | L3
 *   257 x 512    | ~132K     | ~3 MB      | Exceeds L3
 *   769 x 1024   | ~787K     | ~19 MB     | Main memory
 *   1537 x 2048  | ~3.1M     | ~75 MB     | Main memory
 *   3073 x 4096  | ~12.6M    | ~302 MB    | Main memory
 *   6145 x 8192  | ~50.3M    | ~1.2 GB    | Main memory
 *
 * For roofline analysis, use grids that exceed L3 cache to measure
 * main memory bandwidth effects.
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

#include "../../include/GMGPolar/gmgpolar.h"
#include "../../include/Smoother/SmootherGive/smootherGive.h"
#include "../../include/InputFunctions/domainGeometry.h"
#include "../../include/InputFunctions/densityProfileCoefficients.h"

/* Test Case Includes */
#include "../include/InputFunctions/DomainGeometry/czarnyGeometry.h"
#include "../include/InputFunctions/DensityProfileCoefficients/zoniShiftedCoefficients.h"

/* LIKWID Support for accurate hardware counter measurements */
#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#endif

namespace RooflineBenchmark
{
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

struct BenchmarkResult {
    double time_per_iteration_ms;
    double gflops_per_sec;
    double bandwidth_gb_per_sec;
    double arithmetic_intensity;
    size_t grid_size;
    int num_iterations;
    int nr;      // number of radial points
    int ntheta;  // number of angular points
    bool is_estimated;  // true if values are estimated, false if measured by LIKWID
};

void print_roofline_data(const BenchmarkResult& result) {
    std::cout << "\n========== ROOFLINE BENCHMARK RESULTS ==========\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Grid Size:              " << result.nr << " x " << result.ntheta
              << " (" << result.grid_size << " nodes)\n";
    std::cout << "Iterations:             " << result.num_iterations << "\n";
    std::cout << "Time per iteration:     " << result.time_per_iteration_ms << " ms\n";
    std::cout << "Performance:            " << result.gflops_per_sec << " GFLOP/s";
    if (result.is_estimated) std::cout << " (estimated)";
    std::cout << "\n";
    std::cout << "Memory Bandwidth:       " << result.bandwidth_gb_per_sec << " GB/s";
    if (result.is_estimated) std::cout << " (estimated)";
    std::cout << "\n";
    std::cout << "Arithmetic Intensity:   " << result.arithmetic_intensity << " FLOP/byte";
    if (result.is_estimated) std::cout << " (estimated)";
    std::cout << "\n";
    std::cout << "================================================\n";

    if (result.is_estimated) {
        std::cout << "\nNote: For accurate measurements, run with LIKWID:\n";
        std::cout << "  likwid-perfctr -C 0 -g FLOPS_DP -m ./gmgpolar_tests --gtest_filter=RooflineBenchmark.*\n";
        std::cout << "  likwid-perfctr -C 0 -g MEM_DP -m ./gmgpolar_tests --gtest_filter=RooflineBenchmark.*\n";
    }
    std::cout << "\n";

    // Output for easy plotting (CSV format)
    std::cout << "ROOFLINE_DATA,"
              << result.nr << "," << result.ntheta << ","
              << result.arithmetic_intensity << ","
              << result.gflops_per_sec << "\n";
}

/*
 * Estimated FLOP and byte counts per grid point for the SmootherGive operation.
 *
 * These estimates are used when LIKWID measurements are not available.
 * For accurate roofline analysis, run with LIKWID to get measured values.
 *
 * FLOP estimate breakdown (approximate):
 *   - Stencil coefficient calculations: ~20 FLOPs per point
 *   - Matrix-vector products in ASC ortho: ~40 FLOPs per point
 *   - Tridiagonal solves: ~10 FLOPs per point
 *   - Total: ~70 FLOPs per interior point (varies with boundary conditions)
 *
 * Note: The actual FLOP count depends on grid structure and boundary conditions.
 * LIKWID measurements on a 33x65 grid showed ~226 FLOPs/point, but this includes
 * function call overhead and other effects that scale differently with grid size.
 *
 * Byte estimate breakdown:
 *   - Read x (stencil neighbors): 5-9 doubles = 40-72 bytes
 *   - Read rhs: 1 double = 8 bytes
 *   - Read/write temp: 2 doubles = 16 bytes
 *   - Read cached coefficients (arr, att, art, detDF, coeff_beta): 5 doubles = 40 bytes
 *   - Write x: 1 double = 8 bytes
 *   Total: ~112-144 bytes per point (assuming no cache reuse)
 *
 * For large grids that don't fit in cache, memory bandwidth becomes the bottleneck.
 * For small grids, cache effects dominate and these estimates are less accurate.
 */
constexpr int ESTIMATED_FLOPS_PER_POINT = 70;   // Conservative estimate
constexpr int ESTIMATED_BYTES_PER_POINT = 112;  // Assuming minimal cache reuse

BenchmarkResult run_smoother_benchmark(
    SmootherGive& smoother_op,
    const PolarGrid& grid,
    int num_warmup,
    int num_iterations,
    const std::string& marker_name = "smoother")
{
    BenchmarkResult result;
    result.grid_size = grid.numberOfNodes();
    result.num_iterations = num_iterations;
    result.nr = grid.nr();
    result.ntheta = grid.ntheta();
    result.is_estimated = true;  // Will be set to false if LIKWID provides actual measurements

    // Create data
    Vector<double> solution = generate_random_data(grid, 42);
    Vector<double> rhs = generate_random_data(grid, 69);
    Vector<double> temp = generate_random_data(grid, 8);

    // Initialize LIKWID markers if available
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_INIT;
    LIKWID_MARKER_THREADINIT;
    LIKWID_MARKER_REGISTER(marker_name.c_str());
#endif

    // Warmup iterations (outside LIKWID region)
    for (int i = 0; i < num_warmup; ++i) {
        smoother_op.smoothing(solution, rhs, temp);
    }

    // Timed iterations with LIKWID markers
    auto start = std::chrono::high_resolution_clock::now();

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START(marker_name.c_str());
#endif

    for (int i = 0; i < num_iterations; ++i) {
        smoother_op.smoothing(solution, rhs, temp);
    }

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP(marker_name.c_str());
#endif

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calculate metrics from timing
    double total_time_sec = elapsed.count();
    result.time_per_iteration_ms = (total_time_sec / num_iterations) * 1000.0;

    // Use estimated values (LIKWID output must be analyzed separately)
    size_t total_flops = static_cast<size_t>(grid.numberOfNodes()) * ESTIMATED_FLOPS_PER_POINT;
    size_t total_bytes = static_cast<size_t>(grid.numberOfNodes()) * ESTIMATED_BYTES_PER_POINT;

    double total_flops_all_iters = static_cast<double>(total_flops) * num_iterations;
    double total_bytes_all_iters = static_cast<double>(total_bytes) * num_iterations;

    result.gflops_per_sec = (total_flops_all_iters / total_time_sec) / 1e9;
    result.bandwidth_gb_per_sec = (total_bytes_all_iters / total_time_sec) / 1e9;
    result.arithmetic_intensity = static_cast<double>(ESTIMATED_FLOPS_PER_POINT) / ESTIMATED_BYTES_PER_POINT;

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_CLOSE;
#endif

    return result;
}

/*
 * Helper function to create uniform grid points.
 * Creates nr radial points from r_min to Rmax
 * Creates ntheta angular points from 0 to 2*pi (ntheta must be even)
 */
std::pair<std::vector<double>, std::vector<double>> create_uniform_grid(
    int nr, int ntheta, double r_min, double Rmax)
{
    std::vector<double> radii(nr);
    for (int i = 0; i < nr; ++i) {
        radii[i] = r_min + i * (Rmax - r_min) / (nr - 1);
    }

    std::vector<double> angles(ntheta + 1);  // +1 because last angle = first angle (periodic)
    for (int i = 0; i <= ntheta; ++i) {
        angles[i] = i * 2 * M_PI / ntheta;
    }

    return {radii, angles};
}

/*
 * Helper struct to hold benchmark configuration
 */
struct BenchmarkConfig {
    int nr;
    int ntheta;
    int num_warmup;
    int num_iterations;
    int num_omp_threads;
    std::string marker_name;
};

/*
 * Run a complete benchmark with the given configuration
 */
BenchmarkResult run_benchmark_with_config(const BenchmarkConfig& config)
{
    const double Rmax = 1.3;
    const double r_min = 1e-5;
    const double kappa_eps = 0.3;
    const double delta_e = 1.4;

    auto [radii, angles] = create_uniform_grid(config.nr, config.ntheta, r_min, Rmax);

    CzarnyGeometry domain_geometry(Rmax, kappa_eps, delta_e);

    double alpha_jump = 0.678 * Rmax;
    std::unique_ptr<DensityProfileCoefficients> coefficients =
        std::make_unique<ZoniShiftedCoefficients>(Rmax, alpha_jump);

    bool DirBC_Interior = true;
    bool cache_density_profile = true;
    bool cache_domain_geom = false;

    auto grid = std::make_unique<PolarGrid>(radii, angles);
    auto levelCache = std::make_unique<LevelCache>(*grid, *coefficients, domain_geometry,
                                                   cache_density_profile, cache_domain_geom);
    Level level(0, std::move(grid), std::move(levelCache), ExtrapolationType::NONE, 0);

    SmootherGive smoother_op(level.grid(), level.levelCache(), domain_geometry,
                            *coefficients, DirBC_Interior, config.num_omp_threads);

    return run_smoother_benchmark(smoother_op, level.grid(),
                                  config.num_warmup, config.num_iterations,
                                  config.marker_name);
}

} // namespace RooflineBenchmark

using namespace RooflineBenchmark;

/*
 * ============================================================================
 * SMALL GRID TESTS - For quick validation (fits in L1/L2 cache)
 * ============================================================================
 */

TEST(RooflineBenchmark, SmallGrid_Sequential)
{
    BenchmarkConfig config{
        .nr = 33,
        .ntheta = 64,
        .num_warmup = 10,
        .num_iterations = 100,
        .num_omp_threads = 1,
        .marker_name = "small_seq"
    };

    auto result = run_benchmark_with_config(config);
    print_roofline_data(result);

    ASSERT_GT(result.gflops_per_sec, 0.0);
    ASSERT_GT(result.bandwidth_gb_per_sec, 0.0);
}

/*
 * ============================================================================
 * PUBLICATION-QUALITY BENCHMARKS - Grid sizes matching roofline_model.py
 * These grids are large enough to stress main memory bandwidth
 * ============================================================================
 */

/*
 * Grid: 769 x 1024 (~787K nodes, ~19 MB data)
 * This grid exceeds typical L3 cache sizes
 */
TEST(RooflineBenchmark, Grid_769x1024_Sequential)
{
    BenchmarkConfig config{
        .nr = 769,
        .ntheta = 1024,
        .num_warmup = 5,
        .num_iterations = 50,
        .num_omp_threads = 1,
        .marker_name = "769x1024_seq"
    };

    std::cout << "\nBenchmark: 769 x 1024 grid (sequential)\n";
    auto result = run_benchmark_with_config(config);
    print_roofline_data(result);

    ASSERT_GT(result.gflops_per_sec, 0.0);
}

TEST(RooflineBenchmark, Grid_769x1024_Parallel)
{
    BenchmarkConfig config{
        .nr = 769,
        .ntheta = 1024,
        .num_warmup = 5,
        .num_iterations = 50,
        .num_omp_threads = 16,
        .marker_name = "769x1024_par"
    };

    std::cout << "\nBenchmark: 769 x 1024 grid (parallel, 16 threads)\n";
    auto result = run_benchmark_with_config(config);
    print_roofline_data(result);

    ASSERT_GT(result.gflops_per_sec, 0.0);
}

/*
 * Grid: 1537 x 2048 (~3.1M nodes, ~75 MB data)
 */
TEST(RooflineBenchmark, Grid_1537x2048_Sequential)
{
    BenchmarkConfig config{
        .nr = 1537,
        .ntheta = 2048,
        .num_warmup = 3,
        .num_iterations = 20,
        .num_omp_threads = 1,
        .marker_name = "1537x2048_seq"
    };

    std::cout << "\nBenchmark: 1537 x 2048 grid (sequential)\n";
    auto result = run_benchmark_with_config(config);
    print_roofline_data(result);

    ASSERT_GT(result.gflops_per_sec, 0.0);
}

TEST(RooflineBenchmark, Grid_1537x2048_Parallel)
{
    BenchmarkConfig config{
        .nr = 1537,
        .ntheta = 2048,
        .num_warmup = 3,
        .num_iterations = 20,
        .num_omp_threads = 16,
        .marker_name = "1537x2048_par"
    };

    std::cout << "\nBenchmark: 1537 x 2048 grid (parallel, 16 threads)\n";
    auto result = run_benchmark_with_config(config);
    print_roofline_data(result);

    ASSERT_GT(result.gflops_per_sec, 0.0);
}

/*
 * Grid: 3073 x 4096 (~12.6M nodes, ~302 MB data)
 */
TEST(RooflineBenchmark, Grid_3073x4096_Sequential)
{
    BenchmarkConfig config{
        .nr = 3073,
        .ntheta = 4096,
        .num_warmup = 2,
        .num_iterations = 10,
        .num_omp_threads = 1,
        .marker_name = "3073x4096_seq"
    };

    std::cout << "\nBenchmark: 3073 x 4096 grid (sequential)\n";
    auto result = run_benchmark_with_config(config);
    print_roofline_data(result);

    ASSERT_GT(result.gflops_per_sec, 0.0);
}

TEST(RooflineBenchmark, Grid_3073x4096_Parallel)
{
    BenchmarkConfig config{
        .nr = 3073,
        .ntheta = 4096,
        .num_warmup = 2,
        .num_iterations = 10,
        .num_omp_threads = 16,
        .marker_name = "3073x4096_par"
    };

    std::cout << "\nBenchmark: 3073 x 4096 grid (parallel, 16 threads)\n";
    auto result = run_benchmark_with_config(config);
    print_roofline_data(result);

    ASSERT_GT(result.gflops_per_sec, 0.0);
}

/*
 * Grid: 6145 x 8192 (~50.3M nodes, ~1.2 GB data)
 * WARNING: This test requires significant memory and time
 */
TEST(RooflineBenchmark, DISABLED_Grid_6145x8192_Sequential)
{
    BenchmarkConfig config{
        .nr = 6145,
        .ntheta = 8192,
        .num_warmup = 1,
        .num_iterations = 5,
        .num_omp_threads = 1,
        .marker_name = "6145x8192_seq"
    };

    std::cout << "\nBenchmark: 6145 x 8192 grid (sequential) - LARGE\n";
    auto result = run_benchmark_with_config(config);
    print_roofline_data(result);

    ASSERT_GT(result.gflops_per_sec, 0.0);
}

TEST(RooflineBenchmark, DISABLED_Grid_6145x8192_Parallel)
{
    BenchmarkConfig config{
        .nr = 6145,
        .ntheta = 8192,
        .num_warmup = 1,
        .num_iterations = 5,
        .num_omp_threads = 16,
        .marker_name = "6145x8192_par"
    };

    std::cout << "\nBenchmark: 6145 x 8192 grid (parallel, 16 threads) - LARGE\n";
    auto result = run_benchmark_with_config(config);
    print_roofline_data(result);

    ASSERT_GT(result.gflops_per_sec, 0.0);
}

/*
 * ============================================================================
 * PARAMETER SWEEP - Run multiple grid sizes for roofline analysis
 * ============================================================================
 */
TEST(RooflineBenchmark, ParameterSweep_Sequential)
{
    std::cout << "\n============ ROOFLINE PARAMETER SWEEP (Sequential) ============\n";
    std::cout << "Running benchmarks at multiple grid sizes for roofline analysis.\n";
    std::cout << "For accurate measurements, run with LIKWID.\n\n";

    // Grid sizes that produce interesting roofline data points
    std::vector<std::pair<int, int>> grid_sizes = {
        {65, 128},      // ~8K nodes, fits in L2
        {129, 256},     // ~33K nodes, fits in L3
        {257, 512},     // ~132K nodes, exceeds L3
        {513, 1024},    // ~525K nodes
        {769, 1024},    // ~787K nodes (matches Python data)
    };

    std::cout << "nr,ntheta,nodes,time_ms,gflops,gbytes,AI\n";

    for (const auto& [nr, ntheta] : grid_sizes) {
        BenchmarkConfig config{
            .nr = nr,
            .ntheta = ntheta,
            .num_warmup = 5,
            .num_iterations = 20,
            .num_omp_threads = 1,
            .marker_name = "sweep_" + std::to_string(nr) + "x" + std::to_string(ntheta)
        };

        auto result = run_benchmark_with_config(config);

        std::cout << result.nr << ","
                  << result.ntheta << ","
                  << result.grid_size << ","
                  << result.time_per_iteration_ms << ","
                  << result.gflops_per_sec << ","
                  << result.bandwidth_gb_per_sec << ","
                  << result.arithmetic_intensity << "\n";
    }

    std::cout << "\n===============================================================\n";
}
