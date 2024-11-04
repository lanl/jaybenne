# ========================================================================================
# (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights
# reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for
# Los Alamos National Laboratory (LANL), which is operated by Triad National
# Security, LLC for the U.S. Department of Energy/National Nuclear Security
# Administration. All rights in the program are reserved by Triad National
# Security, LLC, and the U.S. Department of Energy/National Nuclear Security
# Administration. The Government is granted for itself and others acting on its
# behalf a nonexclusive, paid-up, irrevocable worldwide license in this material
# to reproduce, prepare derivative works, distribute copies to the public,
# perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# Set configuration values for jaybenne library to interface with host
# NOTE(@pdmullen): Default values provided here are mcblock-specific
if(JAYBENNE_STANDALONE_MODE)
  set(JAYBENNE_OPACITY_HEADER
      "../mcblock/opacity.hpp"
      CACHE STRING "Path to opacity header for library to use" FORCE)
  set(JAYBENNE_EOS_HEADER
      "../mcblock/eos.hpp"
      CACHE STRING "Path to equation of state header for library to use" FORCE)
  set(JAYBENNE_EOS_TYPE
      "mcblock::EOS"
      CACHE STRING "Namespace for equation of state Singularity type" FORCE)
  set(JAYBENNE_ABSORPTION_OPACITY_TYPE
      "mcblock::Opacity"
      CACHE STRING "Namespace for absorption opacity Singularity type" FORCE)
  set(JAYBENNE_SCATTERING_OPACITY_TYPE
      "mcblock::Scattering"
      CACHE STRING "Namespace for scattering opacity Singularity type" FORCE)
  set(JAYBENNE_HOST_VARIABLE_HEADER
      "../mcblock/mcblock_variables.hpp"
      CACHE STRING "Path to host code variables header file" FORCE)
  set(JAYBENNE_HOST_DENSITY_VARIABLE
      "field::material::density"
      CACHE STRING "Variable type name for host density" FORCE)
  set(JAYBENNE_HOST_SPECIFIC_INTERNAL_ENERGY_VARIABLE
      "field::material::sie"
      CACHE STRING "Variable type name for host internal energy density" FORCE)
  set(JAYBENNE_HOST_UPDATE_ENERGY_VARIABLE
      "field::material::internal_energy"
      CACHE STRING "Variable type name for host internal energy density" FORCE)

  # Set locations of singularity include files for jaybenne library to use
  # during its configuration step
  set(JAYBENNE_SINGULARITY_INCLUDE_PATHS
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos"
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-opac"
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos/utils"
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos/utils/ports-of-call"
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos/utils/spiner"
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos/utils/variant/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-opac/utils")
endif()

# Build jaybenne library
add_subdirectory(jaybenne)

# Build mcblock driver
if(JAYBENNE_STANDALONE_MODE)
  add_subdirectory(mcblock)
endif()
