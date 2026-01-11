#!/bin/bash

# Kokkos configuration for OpenMP
export Kokkos_DIR="$HOME/source/kokkos/build-openmp/cmake_packages/Kokkos"

# MUMPS directory (if you enable it with -DGMGPOLAR_USE_MUMPS=ON)
export MUMPS_DIR="$HOME/source/mumps/build/local"

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

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

    cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build" \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DKokkos_DIR="$Kokkos_DIR" \
        -DGMGPOLAR_USE_MUMPS=OFF \
        -DGMGPOLAR_BUILD_TESTS=ON || { echo "CMake configuration failed"; exit 1; }
fi

echo "Building project..."
cmake --build "$PROJECT_ROOT/build" -j 16 || { echo "Build failed"; exit 1; }

echo "Build completed successfully!"
