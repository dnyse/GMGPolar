/*
 * =============================================================================
 * GOLDEN EQUIVALENCE TEST FOR SmootherGive::smoothing()
 * =============================================================================
 *
 * Locks the numerical output of a fixed sequence of SmootherGive smoothing
 * sweeps so that performance refactors can be verified to still compute the
 * correct result. The reference checksums below were captured from the
 * original (pre-optimization) implementation.
 *
 * Two invariants are checked, for both boundary variants (DirBC interior and
 * across-origin):
 *   1) The sequential path (1 thread) reproduces the golden checksum.
 *   2) The parallel path (fixed thread count) reproduces the same result as
 *      the sequential path (and hence the golden checksum).
 *
 * Any incorrect refactor of applyAscOrtho*, the line solvers, the coloring,
 * or the data layout changes these checksums and fails the test.
 * =============================================================================
 */

#include <gtest/gtest.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "../../include/GMGPolar/gmgpolar.h"
#include "../../include/Smoother/SmootherGive/smootherGive.h"
#include "../../include/InputFunctions/domainGeometry.h"
#include "../../include/InputFunctions/densityProfileCoefficients.h"

#include "../include/InputFunctions/DomainGeometry/czarnyGeometry.h"
#include "../include/InputFunctions/DensityProfileCoefficients/zoniShiftedCoefficients.h"

namespace SmootherGiveGolden
{

struct Checksum {
    double sum;   // sum_i x_i
    double sumsq; // sum_i x_i^2
    double wsum;  // sum_i sin(0.001*(i+1)) * x_i   (order-sensitive mixing)
};

static Vector<double> make_data(const PolarGrid& grid, unsigned seed)
{
    Vector<double> x("x", grid.numberOfNodes());
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < x.size(); ++i)
        x(i) = dist(gen);
    return x;
}

static Checksum checksum(const Vector<double>& x)
{
    Checksum c{0.0, 0.0, 0.0};
    for (int i = 0; i < x.size(); ++i) {
        const double v = x(i);
        c.sum += v;
        c.sumsq += v * v;
        c.wsum += std::sin(0.001 * (i + 1)) * v;
    }
    return c;
}

// Run `iters` smoothing sweeps on a fixed problem and return the checksum of x.
static Checksum run_case(int nr, int ntheta, bool DirBC_Interior, int num_threads, int iters)
{
    const double Rmax      = 1.3;
    const double r_min     = 1e-5;
    const double kappa_eps = 0.3;
    const double delta_e   = 1.4;

    std::vector<double> radii(nr);
    for (int i = 0; i < nr; ++i)
        radii[i] = r_min + i * (Rmax - r_min) / (nr - 1);
    std::vector<double> angles(ntheta + 1);
    for (int i = 0; i <= ntheta; ++i)
        angles[i] = i * 2 * M_PI / ntheta;

    CzarnyGeometry domain_geometry(Rmax, kappa_eps, delta_e);
    double alpha_jump = 0.678 * Rmax;
    std::unique_ptr<DensityProfileCoefficients> coefficients =
        std::make_unique<ZoniShiftedCoefficients>(Rmax, alpha_jump);

    auto grid       = std::make_unique<PolarGrid>(radii, angles);
    auto levelCache = std::make_unique<LevelCache>(*grid, *coefficients, domain_geometry, true, false);
    Level level(0, std::move(grid), std::move(levelCache), ExtrapolationType::NONE, 0);

    SmootherGive smoother_op(level.grid(), level.levelCache(), domain_geometry, *coefficients, DirBC_Interior,
                             num_threads);

    Vector<double> x    = make_data(level.grid(), 42);
    Vector<double> rhs  = make_data(level.grid(), 69);
    Vector<double> temp = make_data(level.grid(), 8);

    for (int it = 0; it < iters; ++it)
        smoother_op.smoothing(x, rhs, temp);

    return checksum(x);
}

// ---- Golden reference values (captured from the original implementation) ----
// nr=129, ntheta=128, iters=30
struct Golden {
    double sum, sumsq, wsum;
};
constexpr Golden GOLDEN_DIRBC  = {-140969.52415082563, 253902476.94138741, -340672.51331936056};
constexpr Golden GOLDEN_ACROSS = {-201303.78592188953, 262321939.55585948, -358803.74526616267};

// Coarse tripwire tolerance: tolerant of FP reordering from a legitimate
// parallel/data-layout rewrite, but any genuine algorithmic error (wrong
// coefficient, wrong coloring, missing contribution) deviates by orders of
// magnitude more. The strict correctness guarantee lives in the SmootherTest
// convergence suite (converges to the direct solve within 1e-12).
constexpr double REL_TOL = 1e-6;

static void expect_close(const Checksum& c, const Golden& g)
{
    EXPECT_NEAR(c.sum,   g.sum,   REL_TOL * std::abs(g.sum)   + 1e-6);
    EXPECT_NEAR(c.sumsq, g.sumsq, REL_TOL * std::abs(g.sumsq) + 1e-6);
    EXPECT_NEAR(c.wsum,  g.wsum,  REL_TOL * std::abs(g.wsum)  + 1e-6);
}

static void expect_equal(const Checksum& a, const Checksum& b)
{
    EXPECT_NEAR(a.sum,   b.sum,   1e-7 * std::abs(b.sum)   + 1e-6);
    EXPECT_NEAR(a.sumsq, b.sumsq, 1e-7 * std::abs(b.sumsq) + 1e-6);
    EXPECT_NEAR(a.wsum,  b.wsum,  1e-7 * std::abs(b.wsum)  + 1e-6);
}

// Set GOLDEN_CAPTURE=1 in the environment to print checksums instead of asserting.
static bool capture_mode()
{
    const char* e = std::getenv("GOLDEN_CAPTURE");
    return e && std::atoi(e) != 0;
}

static void report(const char* tag, const Checksum& c)
{
    std::cout << std::setprecision(17) << "GOLDEN " << tag << "  sum=" << c.sum << "  sumsq=" << c.sumsq
              << "  wsum=" << c.wsum << "\n";
}

} // namespace SmootherGiveGolden

using namespace SmootherGiveGolden;

TEST(SmootherGiveGolden, DirBC_Interior_Sequential)
{
    Checksum c = run_case(129, 128, /*DirBC*/ true, /*threads*/ 1, /*iters*/ 30);
    if (capture_mode()) { report("DIRBC_SEQ", c); return; }
    expect_close(c, GOLDEN_DIRBC);
}

TEST(SmootherGiveGolden, DirBC_Interior_Parallel)
{
    Checksum c = run_case(129, 128, /*DirBC*/ true, /*threads*/ 8, /*iters*/ 30);
    if (capture_mode()) { report("DIRBC_PAR", c); return; }
    expect_close(c, GOLDEN_DIRBC);
}

TEST(SmootherGiveGolden, AcrossOrigin_Sequential)
{
    Checksum c = run_case(129, 128, /*DirBC*/ false, /*threads*/ 1, /*iters*/ 30);
    if (capture_mode()) { report("ACROSS_SEQ", c); return; }
    expect_close(c, GOLDEN_ACROSS);
}

TEST(SmootherGiveGolden, AcrossOrigin_Parallel)
{
    Checksum c = run_case(129, 128, /*DirBC*/ false, /*threads*/ 8, /*iters*/ 30);
    if (capture_mode()) { report("ACROSS_PAR", c); return; }
    expect_close(c, GOLDEN_ACROSS);
}

// Direct sequential-vs-parallel equivalence (independent of the golden constants).
TEST(SmootherGiveGolden, SeqParEquivalence)
{
    expect_equal(run_case(129, 128, true,  8, 30), run_case(129, 128, true,  1, 30));
    expect_equal(run_case(129, 128, false, 8, 30), run_case(129, 128, false, 1, 30));
}
