#=========================================================================================
# (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for
# the U.S. Department of Energy/National Nuclear Security Administration. All rights in
# the program are reserved by Triad National Security, LLC, and the U.S. Department of
# Energy/National Nuclear Security Administration. The Government is granted for itself
# and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
# in this material to reproduce, prepare. derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#=========================================================================================

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set( jb_descript "The Jaybenne thermal radiative transfer code" )
project( Jaybenne
  VERSION 0.1
  DESCRIPTION ${jb_descript}
  LANGUAGES C CXX )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

option(JAYBENNE_ENABLE_CUDA "Enable CUDA for Jaybenne and all dependencies" OFF)
option(JAYBENNE_ENABLE_HDF5 "Enable HDF5 for Jaybenne and all dependencies" ON)
option(JAYBENNE_ENABLE_MPI "Enable MPI for Jaybenne and all dependencies" ON)
option(JAYBENNE_ENABLE_OPENMP "Enable OpenbMP for Jaybenne and all dependencies" OFF)
option(JAYBENNE_STANDALONE_MODE "Enable McBlock driver executable" ON)

# Prevent building in the source tree
file( TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if( EXISTS "${LOC_PATH}")
  message( FATAL_ERROR "You cannot build in the source tree. Please make a \
                        build directory elsewhere." )
endif()

# Default to RelWithDebInfo
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# If this is a debug build, set kokkos debug on
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Enabling Kokkos debug mode")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
     "Bounds checking on Kokkos views")
  set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
      "Sanity checks on Kokkos DualView")
endif()

# Always use Kokkos
set(PORTABILITY_STRATEGY_KOKKOS ON CACHE BOOL "" FORCE)

# Aggressive vectorization
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
    "Kokkos aggressive vectorization" FORCE)

# CUDA
if(JAYBENNE_ENABLE_CUDA)
  set(SINGULARITY_USE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "" FORCE)
endif()

# HDF5
if(JAYBENNE_ENABLE_HDF5)
  set(HDF5_PREFER_PARALLEL ${JAYBENNE_ENABLE_MPI})
  find_package(HDF5 COMPONENTS C HL)
  if (NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 is required but couldn't be found. "
      "If you want to build artemis without HDF5, please rerun "
      "CMake with -DJAYBENNE_ENABLE_HDF5=OFF")
  endif()
  if (JAYBENNE_ENABLE_MPI AND (NOT HDF5_IS_PARALLEL))
    message(FATAL_ERROR "Both MPI and HDF5 are enabled "
      "but only a serial version of HDF5 was found. Please install "
      "a parallel version of HDF5 (or point CMake to it by adding its path "
      "to the CMAKE_PREFIX_PATH environment variable), or disable either MPI "
      "or HDF5 by rerunning CMake with -DJAYBENNE_ENABLE_MPI=OFF or "
      "-DJAYBENNE_ENABLE_HDF5=OFF")
  endif()
  set(SINGULARITY_USE_HDF5 ON CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_HDF5 OFF CACHE BOOL "" FORCE)
else()
  set(SINGULARITY_USE_HDF5 OFF CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_HDF5 ON CACHE BOOL "" FORCE)
endif()

if(JAYBENNE_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
else()
  set(PARTHENON_DISABLE_MPI ON CACHE BOOL "" FORCE)
endif()

if(JAYBENNE_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  set(PAR_LOOP_LAYOUT MDRANGE_LOOP CACHE STRING "" FORCE)
else()
  set(PARTHENON_DISABLE_OPENMP ON CACHE BOOL "" FORCE)
endif()

# Configure parthenon if the target does not already exist in the build
if (NOT TARGET parthenon)
  message(STATUS "Configuring Parthenon")
  set(PARTHENON_ENABLE_INIT_PACKING ON CACHE BOOL "" FORCE)
  set(PARTHENON_LINT_DEFAULT OFF CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  add_subdirectory(external/parthenon parthenon)
else()
  message(STATUS "Parthenon already configured in this build, skipping.")
endif()

# singularity common
message("Configuring singularity")
set(SINGULARITY_USE_KOKKOS ON CACHE BOOL "" FORCE)
set(SINGULARITY_USE_FORTRAN OFF CACHE BOOL "" FORCE)
set(SINGULARITY_SUBMODULE_MODE ON CACHE BOOL "" FORCE)
set(SINGULARITY_USE_KOKKOSKERNELS OFF CACHE BOOL "" FORCE)
set(SINGULARITY_BUILD_CLOSURE OFF CACHE BOOL "" FORCE)

# Disable cmake's package registry because it can interfere
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON CACHE BOOL "" FORCE)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON CACHE BOOL "" FORCE)

# singularity-eos (patch applies to singularity-opac as well)
message(STATUS "Patching mpark::variant to support GPUs")
execute_process(COMMAND patch -N -s -V never
  ${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-eos/utils/variant/include/mpark/variant.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-eos/utils/cuda_compatibility.patch
)

# TODO(BRR) The below may replace all manual includes... but there is some issue with
# using both singularity-eos and singularity-opac together

## singularity eos
# message("Configuring singularity-eos")
# add_subdirectory(external/singularity-eos singularity-eos)

## singularity opac
# message("Configuring singularity-opac")
# add_subdirectory(external/singularity-opac singularity-opac)

# Enroll source code for compilation
message("Configuring jaybenne source")
add_subdirectory(src)
