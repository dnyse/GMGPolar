#!/bin/bash
# Reproducible environment + benchmark harness for the SmootherGive optimization work.
# Usage:
#   source scripts/bench_smoother.sh           # just load the environment (modules/env vars)
#   ./scripts/bench_smoother.sh build          # (re)compile the project
#   ./scripts/bench_smoother.sh correctness    # run the correctness oracles (must stay green)
#   ./scripts/bench_smoother.sh bench [reps]    # run the smoother benchmarks, best-of-<reps> (default 3)
#   ./scripts/bench_smoother.sh all [reps]      # build + correctness + bench
#
# Primary metric: 72-core time/iter on the Parallel benchmarks (lower is better).

set -uo pipefail
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# ---- Environment (Fritz, AlmaLinux 9) ----
source /apps/modules/5.6.0/init/bash
export MODULEPATH=/apps/modules/data
module load compiler/gcc/11.5.0 tools/cmake/3.31.8 tools/likwid/5.4.1 >/dev/null 2>&1
export LD_LIBRARY_PATH="${LIKWID_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export OMP_PROC_BIND=spread OMP_PLACES=cores

TESTBIN="$ROOT/build/tests/gmgpolar_tests"

do_build() {
    echo ">>> Building..."
    cmake --build "$ROOT/build" -j 16 --target gmgpolar_tests || return 1
    echo ">>> Build OK"
}

do_correctness() {
    echo ">>> Correctness oracle (SmootherTest Give + SmootherGiveGolden) ..."
    "$TESTBIN" --gtest_filter='SmootherGiveGolden.*:SmootherTest.SequentialSmootherDirBC_Interior:SmootherTest.ParallelSmootherDirBC_Interior:SmootherTest.SequentialSmootherAcrossOrigin:SmootherTest.ParallelSmootherAcrossOrigin:SmootherTest.smoother_DirBC_Interior:SmootherTest.smoother_AcrossOrigin' 2>&1 | grep -E "PASSED|FAILED|OK \]|\[  FAILED"
}

# best-of-N time/iter for one gtest filter at a given thread count
bench_one() {
    local test="$1" threads="$2" reps="$3"
    local best=""
    for r in $(seq 1 "$reps"); do
        local t
        t=$(SMOOTHER_THREADS="$threads" "$TESTBIN" --gtest_filter="RooflineBenchmark.$test" 2>/dev/null \
            | grep TIMING_CSV | tail -1 | awk -F, '{print $7}')
        if [ -n "$t" ]; then
            if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
        fi
    done
    printf "  %-28s threads=%-3s  best/%s: %s ms/iter\n" "$test" "$threads" "$reps" "$best"
}

do_bench() {
    local reps="${1:-3}"
    echo ">>> Benchmark best-of-$reps (ms/iter, lower is better)"
    for t in 1 36 72; do
        bench_one Grid_769x1024_Parallel  "$t" "$reps"
    done
    for t in 1 72; do
        bench_one Grid_1537x2048_Parallel "$t" "$reps"
    done
}

cmd="${1:-env}"
case "$cmd" in
    env)         echo "Environment loaded (gcc 11.5, cmake, likwid). OMP_PROC_BIND=$OMP_PROC_BIND";;
    build)       do_build;;
    correctness) do_correctness;;
    bench)       do_bench "${2:-3}";;
    all)         do_build && do_correctness && do_bench "${2:-3}";;
    *) echo "unknown command: $cmd"; exit 1;;
esac
