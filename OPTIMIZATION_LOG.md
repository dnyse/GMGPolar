# SmootherGive Optimization Log

Tracking the iterative optimization of `SmootherGive::smoothing()` on Fritz
(2-socket Intel Xeon Platinum 8360Y, 72 cores, AlmaLinux 9). Branch: `smoother-opt`.

Metric: median/best-of-N wall-clock **ms per smoothing iteration** (lower is better),
from `./scripts/bench_smoother.sh bench`. Primary target: **72-thread** times.

Correctness gate for every kept change: `./scripts/bench_smoother.sh correctness`
must print `[  PASSED  ] 11 tests.` (SmootherGiveGolden.* + SmootherTest Give suite).

## Baseline (best-of-3, pristine code, commit `991bc2f`)

| Grid       | 1 thread | 36 threads | 72 threads |
|------------|---------:|-----------:|-----------:|
| 769×1024   |  80.9 ms |    88.3 ms |    97.7 ms |
| 1537×2048  | 322.8 ms |        —   |   363.5 ms |

Symptom: **negative scaling** — 72 threads slower than 1.

## Attempts

Newest first. One row per experiment. "Kept?" = committed on `smoother-opt` (Y) or reverted (N).

| # | Change | 769² @72t | 1537² @72t | 769² @1t | Correct? | Kept? | Commit | Notes |
|---|--------|----------:|-----------:|---------:|:--------:|:-----:|--------|-------|
| 2 | Fuse `deep_copy(temp,rhs)` into the parallel region as an `omp for` | **1.13** | **4.27** | 48.8 | ✅ 11/11 | Y | 558fd19 | Removes one Kokkos fork/join per sweep. Big help at 36t (2.76→1.79); 72t improves too (best-of-7). Neutral at 1t (seq path unchanged). |
| 1 | Hot path on raw `double*` (no View copies) + hoist level-cache accessors to raw ptrs | 1.46 | 5.36 | 48.8 | ✅ 11/11 (+275 full) | Y | e550856 | Killed the 63% `SharedAllocationRecord::increment` contention. Per-node `level_cache_.coeff_beta()[i]` / `coeff_alpha()[i]` each minted a temporary `Kokkos::View` (atomic refcount on shared cacheline); hoisting to `.data()` ptrs before the loop was the decisive change. Now scales positively: 769² 1t 48.8→72t 1.46 = 33×. |
| 0 | Baseline | 97.7 | 363.5 | 80.9 | ✅ 11/11 | — | 991bc2f | reference |

## Profiling notes

### 72-thread flat profile (perf, cycles, 769x1024, baseline commit 991bc2f)
- **60.61%** `Kokkos::Impl::SharedAllocationRecord<void,void>::increment` (atomic ref-count bump)
- **3.40%** `SharedAllocationRecord::decrement`
- **~33%** `libgomp` (OpenMP barrier spin / for-loop dispatch — ~15 barriers/sweep)
- **<1%** actual stencil math (`applyAscOrtho*`, `CzarnyGeometry`, tridiagonal solvers)

### 72-thread flat profile AFTER fixes (perf, cycles, 1537x2048, commit e550856)
- **~62%** `libgomp` (OpenMP barrier sync / parallel-for dispatch)
- **~19%** stencil apply (`applyAscOrthoRadialSection` 14.7% + Circle 3.9%)
- **~13%** `CzarnyGeometry::dF*` (per-node Jacobian; benchmark runs with
  `cache_domain_geometry=false`, so this transcendental math is done on the fly)
- **~3%** tridiagonal solvers, **~3%** temp=rhs copy
- `SharedAllocationRecord::increment` now **0.7%** (was 63%).

Remaining bottleneck is OpenMP synchronization: ~15 `#pragma omp for` barriers per
sweep, mandated by the red/black colouring dependencies. Scaling 36t→72t is still
~1.6× (769²), so this is not pure overhead — the box is near its useful limit for
this algorithm. Next step (if pursued): reduce barrier count by fusing colour phases,
or restructure to fewer parallel regions; both are correctness-risky given the
colouring order.

Root cause of negative scaling: `applyAscOrthoCircleSection/RadialSection` and
`solveCircleSection/RadialSection` take `Vector<double>` (= `Kokkos::View`) **by value**.
Each of the thousands of per-line calls copy-constructs the *shared* `x`/`rhs`/`temp`
views, and Kokkos atomically increments/decrements their reference counts. All 72
threads hammer the same 3 ref-count cache lines → cache-line ping-pong dominates.
Fix: thread the hot path on raw `double*` pointers (no View copies). Secondary:
per-call `Kokkos::View` scratch allocations + `deep_copy` of tiny subviews also mint
SharedAllocationRecords under a global lock — replace with plain buffers / `std::copy`.

## Summary of what worked / didn't

### Final results (best-of-7, commit `558fd19`)

| Grid       | 1 thread | 36 threads | 72 threads | 72t speedup vs baseline |
|------------|---------:|-----------:|-----------:|------------------------:|
| 769×1024   | 48.8 ms  |  1.79 ms   |  1.13 ms   | **86×** (97.7→1.13)     |
| 1537×2048  | 194.2 ms |     —      |  4.27 ms   | **85×** (363.5→4.27)    |

Scaling is now strongly positive: 769² goes 48.8 ms @1t → 1.13 ms @72t (**43×**),
where the pristine code went the *wrong* way (80.9 → 97.7).

### What mattered (ranked)
1. **Hoisting per-node `Kokkos::View` accessors to raw pointers** (`coeff_beta()[i]`,
   `coeff_alpha()[i]`, etc.) — decisive. These minted a temporary View per node and
   atomically bumped a shared reference count, so 72 threads ping-ponged the same
   cache line. This single change removed the 63% hotspot and flipped scaling positive.
2. **Passing the hot path on raw `double*` instead of `Vector<double>` by value** —
   same class of problem (View copy-construct per call) for the `x`/`rhs`/`temp`
   arguments; part of the same commit. Also let `solve*` use `std::copy` instead of
   `Kokkos::deep_copy` on tiny subviews.
3. **Fusing `deep_copy(temp,rhs)` into the parallel region** — one fewer fork/join
   per sweep; clear win at 36t, modest at 72t.

### Not needed / not pursued
- Per-thread scratch was switched from `Kokkos::View` to `std::vector` as part of #1
  (avoids the global allocation lock), but this was minor next to the accessor fix.
- Reducing the ~15 `omp for` barriers per sweep: the remaining profile is barrier-sync
  dominated, but 36t→72t still scales ~1.6×, and reordering the red/black colour phases
  is correctness-risky for small expected gain. Left as documented next step.

## LIKWID roofline (baseline vs optimized)

Grid 3073×4096 (DRAM-bound, working set ≫ cache), whole-process measurement
(`likwid-perfctr -g FLOPS_DP / MEM_DP`, no `-m`, summed across cores), high
`SMOOTHER_ITERS` so the smoother dominates. AI computed manually as
`(DP_MFLOPs_sum × runtime) / memory_data_volume` (LIKWID's per-socket AI is
unreliable on 2-socket). Peak ceilings reused from the reference `likwid-bench`.

| cores | baseline GFLOP/s | optimized GFLOP/s | speedup | opt AI (F/B) | opt % of compute peak | opt % of mem roofline |
|------:|-----------------:|------------------:|--------:|-------------:|----------------------:|----------------------:|
| 1     | 1.92 | 3.19   | 1.7× | 1.76 | 3.0% | 9%  |
| 36    | 1.83 | 72.85  | 40×  | 2.30 | 2.5% | 17% |
| 72    | 1.75 | 104.71 | 60×  | 2.84 | 1.8% | 10% |

Both versions sit in the memory-bound region (AI ≈ 1.5–2.8, ridge ≈ 15 F/B), but
the baseline achieved only ~0.3% of the memory roofline (buried by ref-count
contention + barrier overhead); the optimized version reaches ~10–17% — it moved
~60× up toward the roofline at full node. Still ~6–10× below the memory ceiling,
consistent with the residual OpenMP-barrier-sync bottleneck (see profiling notes).

PNGs (baseline vs optimized on each roofline, reference PNGs left untouched):
`HPC-Project/roofline_{1c,36c,72c}_optimized.png` + their `.gnuplot` scripts;
raw LIKWID output in `HPC-Project/roofline_optimized_data/`.
