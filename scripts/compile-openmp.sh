#!/bin/bash

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Kokkos configuration for OpenMP
# Local Kokkos 4.6.02 OpenMP build (see kokkos/install-openmp)
export Kokkos_DIR="${Kokkos_DIR:-$PROJECT_ROOT/kokkos/install-openmp/lib64/cmake/Kokkos}"

# MUMPS directory (if you enable it with -DGMGPOLAR_USE_MUMPS=ON)
export MUMPS_DIR="${MUMPS_DIR:-$HOME/source/mumps/build/local}"

# LIKWID configuration for roofline benchmarking
# Prefer the module-provided LIKWID (LIKWID_ROOT set by `module load tools/likwid`)
export LIKWID_DIR="${LIKWID_DIR:-${LIKWID_ROOT:-/apps/likwid/5.4.1}}"
LIKWID_ENABLED=ON

# Function to display usage information
usage() {
    echo "Usage: $0 [Debug|Release]"
    echo "Defaults to Release if not specified"
    exit 1
}

# Check if build directory exists
if [ -d "$PROJECT_ROOT/build" ]; then
    build_exists=true
else
    build_exists=false
fi

# Check if build directory exists and delete if it does (only if an argument is provided)
if [ -n "$1" ] && [ -d "$PROJECT_ROOT/build" ]; then
    echo "Removing existing build directory..."
    rm -rf "$PROJECT_ROOT/build" || { echo "Failed to remove build directory"; exit 1; }
    build_exists=false
fi

# Create build directory if it doesn't exist
if ! $build_exists; then
    echo "Creating build directory..."
    mkdir -p "$PROJECT_ROOT/build" || { echo "Failed to create build directory"; exit 1; }
    build_exists=true
fi

# Determine build type - default to Release if not specified
if [ -n "$1" ]; then
    case "$1" in
        Debug)
            build_type="Debug"
            ;;
        Release)
            build_type="Release"
            ;;
        *)
            echo "Invalid build type. Please specify Debug or Release."
            usage
            ;;
    esac
elif [ "$build_exists" != true ] || [ ! -f "$PROJECT_ROOT/build/CMakeCache.txt" ]; then
    # Default to Release if build doesn't exist or isn't configured
    build_type="Release"
    echo "No build type specified, defaulting to Release"
fi

if [ -n "$build_type" ]; then
    echo "Configuring with $build_type build type (OpenMP backend)..."
    echo "Using Kokkos from: $Kokkos_DIR"
    echo "Using GCC compiler"

    # Optimization flags for Ice Lake (AVX-512 support)
    # -march=native: Use all CPU features available on this machine
    # -funroll-loops: Unroll loops for better pipelining
    # -ffast-math: Aggressive FP optimizations (may slightly change results)
    # Remove -ffast-math if strict IEEE compliance is needed
    OPT_FLAGS="-march=native -funroll-loops"

    # Build LIKWID flags if enabled
    LIKWID_CXX_FLAGS=""
    LIKWID_LINK_FLAGS=""
    if [ "$LIKWID_ENABLED" = "ON" ]; then
        echo "LIKWID markers enabled for roofline benchmarking"
        LIKWID_CXX_FLAGS="-DLIKWID_PERFMON"
        LIKWID_LINK_FLAGS="-llikwid"
        # Add LIKWID paths if LIKWID_DIR is set
        if [ -n "$LIKWID_DIR" ]; then
            LIKWID_CXX_FLAGS="$LIKWID_CXX_FLAGS -I$LIKWID_DIR/include"
            LIKWID_LINK_FLAGS="-L$LIKWID_DIR/lib $LIKWID_LINK_FLAGS"
        fi
    fi

    # Combine all CXX flags
    ALL_CXX_FLAGS="$OPT_FLAGS $LIKWID_CXX_FLAGS"

    echo "Optimization flags: $OPT_FLAGS"

    cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build" \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DCMAKE_CXX_FLAGS="$ALL_CXX_FLAGS" \
        -DCMAKE_EXE_LINKER_FLAGS="$LIKWID_LINK_FLAGS" \
        -DKokkos_DIR="$Kokkos_DIR" \
        -DGMGPOLAR_USE_MUMPS=OFF \
        -DGMGPOLAR_BUILD_TESTS=ON || { echo "CMake configuration failed"; exit 1; }
fi

echo "Building project..."
cmake --build "$PROJECT_ROOT/build" -j 16 || { echo "Build failed"; exit 1; }

echo "Build completed successfully!"
