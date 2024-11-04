# ========================================================================================
# (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

set(driver_sources mcblock.cpp mcblock_driver.cpp)
file(GLOB driver_headers "*.hpp")
add_library(lib_jaybenne_driver ${driver_sources} ${driver_headers})
target_compile_definitions(lib_jaybenne_driver
                           PRIVATE PORTABILITY_STRATEGY_KOKKOS)
target_link_libraries(lib_jaybenne_driver PUBLIC Parthenon::parthenon
                                                 lib_jaybenne_package)
target_include_directories(
  lib_jaybenne_driver
  PUBLIC Parthenon::parthenon "${CMAKE_CURRENT_BINARY_DIR}"
         "${CMAKE_CURRENT_BINARY_DIR}/../jaybenne"
         ${JAYBENNE_SINGULARITY_INCLUDE_PATHS})
target_include_directories(
  lib_jaybenne_driver PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

add_executable(mcblock "main.cpp")
target_compile_definitions(mcblock PRIVATE PORTABILITY_STRATEGY_KOKKOS)

target_include_directories(
  mcblock
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
         "${CMAKE_CURRENT_BINARY_DIR}"
         "${CMAKE_CURRENT_BINARY_DIR}/../jaybenne"
         ${JAYBENNE_SINGULARITY_INCLUDE_PATHS})

target_link_libraries(mcblock PUBLIC Parthenon::parthenon lib_jaybenne_driver)

add_custom_command(
  TARGET mcblock
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mcblock>
          ${CMAKE_BINARY_DIR}/$<TARGET_FILE_NAME:mcblock>
  COMMAND ${CMAKE_COMMAND} -E remove $<TARGET_FILE:mcblock>)

install(
  TARGETS mcblock
  EXPORT jaybenne-targets
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
