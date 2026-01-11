#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

#include "../../include/GMGPolar/gmgpolar.h"
#include "../../include/Smoother/SmootherGive/smootherGive.h"
#include "../../include/InputFunctions/domainGeometry.h"
#include "../../include/InputFunctions/densityProfileCoefficients.h"

/* Test Case Includes */
#include "../include/InputFunctions/DomainGeometry/czarnyGeometry.h"
#include "../include/InputFunctions/DensityProfileCoefficients/zoniShiftedCoefficients.h"

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
};

void print_roofline_data(const BenchmarkResult& result) {
    std::cout << "\n========== ROOFLINE BENCHMARK RESULTS ==========\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Grid Size:              " << result.grid_size << " nodes\n";
    std::cout << "Iterations:             " << result.num_iterations << "\n";
    std::cout << "Time per iteration:     " << result.time_per_iteration_ms << " ms\n";
    std::cout << "Performance:            " << result.gflops_per_sec << " GFLOP/s\n";
    std::cout << "Memory Bandwidth:       " << result.bandwidth_gb_per_sec << " GB/s\n";
    std::cout << "Arithmetic Intensity:   " << result.arithmetic_intensity << " FLOP/byte\n";
    std::cout << "================================================\n\n";
    
    // Output for easy plotting
    std::cout << "ROOFLINE_DATA," 
              << result.arithmetic_intensity << "," 
              << result.gflops_per_sec << "\n";
}

BenchmarkResult run_smoother_benchmark(
    SmootherGive& smoother_op,
    const PolarGrid& grid,
    int num_warmup,
    int num_iterations,
    int flops_per_point,
    int bytes_per_point)
{
    BenchmarkResult result;
    result.grid_size = grid.numberOfNodes();
    result.num_iterations = num_iterations;
    
    // Create data
    Vector<double> solution = generate_random_data(grid, 42);
    Vector<double> rhs = generate_random_data(grid, 69);
    Vector<double> temp = generate_random_data(grid, 8);
    
    // Warmup iterations
    for (int i = 0; i < num_warmup; ++i) {
        smoother_op.smoothing(solution, rhs, temp);
    }
    
    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        smoother_op.smoothing(solution, rhs, temp);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Calculate metrics
    double total_time_sec = elapsed.count();
    result.time_per_iteration_ms = (total_time_sec / num_iterations) * 1000.0;
    
    // Total FLOPs and bytes per iteration
    size_t total_flops = grid.numberOfNodes() * flops_per_point;
    size_t total_bytes = grid.numberOfNodes() * bytes_per_point;
    
    // Performance metrics
    double total_flops_all_iters = total_flops * num_iterations;
    double total_bytes_all_iters = total_bytes * num_iterations;
    
    result.gflops_per_sec = (total_flops_all_iters / total_time_sec) / 1e9;
    result.bandwidth_gb_per_sec = (total_bytes_all_iters / total_time_sec) / 1e9;
    result.arithmetic_intensity = static_cast<double>(flops_per_point) / bytes_per_point;
    
    return result;
}

} // namespace RooflineBenchmark

using namespace RooflineBenchmark;

/* 
 * Roofline Benchmark Test - Small Grid
 * For quick testing and validation
 */
TEST(RooflineBenchmark, SmallGrid_Sequential)
{
    // Small grid for testing
    std::vector<double> radii  = {1e-5, 0.2, 0.5, 0.8, 1.0, 1.2, 1.3};
    // Need even number of angles (divisible by 2)
    std::vector<double> angles;
    for (int i = 0; i <= 8; ++i) {
        angles.push_back(i * 2 * M_PI / 8);
    }
    
    double Rmax = radii.back();
    double kappa_eps = 0.3;
    double delta_e = 1.4;
    
    CzarnyGeometry domain_geometry(Rmax, kappa_eps, delta_e);
    
    double alpha_jump = 0.678 * Rmax;
    std::unique_ptr<DensityProfileCoefficients> coefficients =
        std::make_unique<ZoniShiftedCoefficients>(Rmax, alpha_jump);
    
    bool DirBC_Interior = true;
    int maxOpenMPThreads = 1;
    bool cache_density_profile = true;
    bool cache_domain_geom = false;
    
    auto grid = std::make_unique<PolarGrid>(radii, angles);
    auto levelCache = std::make_unique<LevelCache>(*grid, *coefficients, domain_geometry,
                                                   cache_density_profile, cache_domain_geom);
    Level level(0, std::move(grid), std::move(levelCache), ExtrapolationType::NONE, 0);
    
    SmootherGive smoother_op(level.grid(), level.levelCache(), domain_geometry, 
                            *coefficients, DirBC_Interior, maxOpenMPThreads);
    
    // FLOP count measured from LIKWID:
    // - Actual measurement: ~231 FLOPs per point
    // - This includes all stencil operations, boundary conditions, etc.
    // Memory: estimated based on stencil reads/writes
    int flops_per_point = 231;
    int bytes_per_point = 48;
    
    int num_warmup = 10;
    int num_iterations = 100;
    
    auto result = run_smoother_benchmark(smoother_op, level.grid(), 
                                        num_warmup, num_iterations,
                                        flops_per_point, bytes_per_point);
    
    print_roofline_data(result);
    
    // Basic sanity checks
    ASSERT_GT(result.gflops_per_sec, 0.0);
    ASSERT_GT(result.bandwidth_gb_per_sec, 0.0);
}

/* 
 * Roofline Benchmark Test - Medium Grid
 * For more realistic measurements
 */
TEST(RooflineBenchmark, MediumGrid_Sequential)
{
    // Medium-sized grid
    std::vector<double> radii  = {1e-5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    std::vector<double> angles;
    for (int i = 0; i <= 16; ++i) {
        angles.push_back(i * 2 * M_PI / 16);
    }
    
    double Rmax = radii.back();
    double kappa_eps = 0.3;
    double delta_e = 1.4;
    
    CzarnyGeometry domain_geometry(Rmax, kappa_eps, delta_e);
    
    double alpha_jump = 0.678 * Rmax;
    std::unique_ptr<DensityProfileCoefficients> coefficients =
        std::make_unique<ZoniShiftedCoefficients>(Rmax, alpha_jump);
    
    bool DirBC_Interior = true;
    int maxOpenMPThreads = 1;
    bool cache_density_profile = true;
    bool cache_domain_geom = false;
    
    auto grid = std::make_unique<PolarGrid>(radii, angles);
    auto levelCache = std::make_unique<LevelCache>(*grid, *coefficients, domain_geometry,
                                                   cache_density_profile, cache_domain_geom);
    Level level(0, std::move(grid), std::move(levelCache), ExtrapolationType::NONE, 0);
    
    SmootherGive smoother_op(level.grid(), level.levelCache(), domain_geometry, 
                            *coefficients, DirBC_Interior, maxOpenMPThreads);
    
    int flops_per_point = 231;
    int bytes_per_point = 48;
    
    int num_warmup = 20;
    int num_iterations = 200;
    
    auto result = run_smoother_benchmark(smoother_op, level.grid(), 
                                        num_warmup, num_iterations,
                                        flops_per_point, bytes_per_point);
    
    print_roofline_data(result);
    
    ASSERT_GT(result.gflops_per_sec, 0.0);
    ASSERT_GT(result.bandwidth_gb_per_sec, 0.0);
}

/* 
 * Roofline Benchmark Test - Parallel
 */
TEST(RooflineBenchmark, MediumGrid_Parallel)
{
    std::vector<double> radii  = {1e-5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    std::vector<double> angles;
    for (int i = 0; i <= 16; ++i) {
        angles.push_back(i * 2 * M_PI / 16);
    }
    
    double Rmax = radii.back();
    double kappa_eps = 0.3;
    double delta_e = 1.4;
    
    CzarnyGeometry domain_geometry(Rmax, kappa_eps, delta_e);
    
    double alpha_jump = 0.678 * Rmax;
    std::unique_ptr<DensityProfileCoefficients> coefficients =
        std::make_unique<ZoniShiftedCoefficients>(Rmax, alpha_jump);
    
    bool DirBC_Interior = true;
    int maxOpenMPThreads = 16;  // Parallel execution
    bool cache_density_profile = true;
    bool cache_domain_geom = false;
    
    auto grid = std::make_unique<PolarGrid>(radii, angles);
    auto levelCache = std::make_unique<LevelCache>(*grid, *coefficients, domain_geometry,
                                                   cache_density_profile, cache_domain_geom);
    Level level(0, std::move(grid), std::move(levelCache), ExtrapolationType::NONE, 0);
    
    SmootherGive smoother_op(level.grid(), level.levelCache(), domain_geometry, 
                            *coefficients, DirBC_Interior, maxOpenMPThreads);
    
    int flops_per_point = 231;
    int bytes_per_point = 48;
    
    int num_warmup = 20;
    int num_iterations = 200;
    
    auto result = run_smoother_benchmark(smoother_op, level.grid(), 
                                        num_warmup, num_iterations,
                                        flops_per_point, bytes_per_point);
    
    print_roofline_data(result);
    
    ASSERT_GT(result.gflops_per_sec, 0.0);
    ASSERT_GT(result.bandwidth_gb_per_sec, 0.0);
}

/* 
 * Roofline Benchmark Test - Large Grid
 * Use this for publication-quality measurements
 * Run with LIKWID for accurate hardware counter measurements
 */
TEST(RooflineBenchmark, LargeGrid_ForLIKWID)
{
    // Large grid - good for cache effects and realistic measurements
    std::vector<double> radii;
    for (int i = 0; i <= 32; ++i) {
        radii.push_back(1e-5 + i * (1.3 - 1e-5) / 32);
    }
    
    std::vector<double> angles;
    for (int i = 0; i <= 64; ++i) {
        angles.push_back(i * 2 * M_PI / 64);
    }
    
    double Rmax = radii.back();
    double kappa_eps = 0.3;
    double delta_e = 1.4;
    
    CzarnyGeometry domain_geometry(Rmax, kappa_eps, delta_e);
    
    double alpha_jump = 0.678 * Rmax;
    std::unique_ptr<DensityProfileCoefficients> coefficients =
        std::make_unique<ZoniShiftedCoefficients>(Rmax, alpha_jump);
    
    bool DirBC_Interior = true;
    int maxOpenMPThreads = 1;  // Start with sequential for clearer measurements
    bool cache_density_profile = true;
    bool cache_domain_geom = false;
    
    auto grid = std::make_unique<PolarGrid>(radii, angles);
    auto levelCache = std::make_unique<LevelCache>(*grid, *coefficients, domain_geometry,
                                                   cache_density_profile, cache_domain_geom);
    Level level(0, std::move(grid), std::move(levelCache), ExtrapolationType::NONE, 0);
    
    SmootherGive smoother_op(level.grid(), level.levelCache(), domain_geometry, 
                            *coefficients, DirBC_Interior, maxOpenMPThreads);
    
    int flops_per_point = 231;
    int bytes_per_point = 48;
    
    int num_warmup = 50;
    int num_iterations = 500;
    
    std::cout << "\nStarting large grid benchmark...\n";
    std::cout << "Grid size: " << level.grid().numberOfNodes() << " nodes\n";
    std::cout << "This may take a while...\n";
    
    auto result = run_smoother_benchmark(smoother_op, level.grid(), 
                                        num_warmup, num_iterations,
                                        flops_per_point, bytes_per_point);
    
    print_roofline_data(result);
    
    ASSERT_GT(result.gflops_per_sec, 0.0);
    ASSERT_GT(result.bandwidth_gb_per_sec, 0.0);
}
