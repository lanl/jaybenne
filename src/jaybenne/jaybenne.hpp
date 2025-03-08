//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#ifndef JAYBENNE_JAYBENNE_HPP_
#define JAYBENNE_JAYBENNE_HPP_

// C/C++ includes
#include <limits>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// Kokkos includes
#include "Kokkos_Random.hpp"
// NOTE(PDM): these typedefs must appear before Jaybenne includes for compilation
typedef Kokkos::Random_XorShift64_Pool<> RngPool;
typedef RngPool::generator_type RngGen;

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon;
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

// Host configuration file
#include "jaybenne_config.hpp"

// Jaybenne includes
// NOTE(BRR) what we want exposed to other packages including mcblock
// TODO(BRR) don't include these here?
#include "boundaries.hpp"
#include "jaybenne_variables.hpp"
#include "planck.hpp"
#include "scattering.hpp"

namespace jaybenne {

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin,
                                                       Opacity &opacity,
                                                       Scattering &scattering, EOS &eos);
std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin,
                                                       MeanOpacity &mopacity,
                                                       MeanScattering &mscattering,
                                                       EOS &eos);

// Model enums
enum class SourceStrategy { uniform, energy };
enum class SourceType { thermal, emission };
enum class FrequencyType { gray, multigroup };

// Initialization nulls
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto JaybenneNull() {
  return std::numeric_limits<T>::quiet_NaN();
}
template <>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto JaybenneNull<int>() {
  return std::numeric_limits<int>::max();
}

// Tasks
template <FrequencyType FT>
TaskStatus TransportPhotons(MeshData<Real> *md, const Real t_start, const Real dt);
template <FrequencyType FT>
TaskStatus TransportPhotons_DDMC(MeshData<Real> *md, const Real t_start, const Real dt);
TaskStatus SampleDDMCBlockFace(MeshData<Real> *md);
TaskStatus CheckCompletion(MeshData<Real> *md, const Real t_end);
template <typename T, SourceType ST, FrequencyType FT>
TaskStatus SourcePhotons(T *md, const Real t_start, const Real dt);
TaskStatus DefragParticles(MeshBlock *pmb);
TaskStatus UpdateDerivedTransportFields(MeshData<Real> *md, const Real dt);
template <typename T>
TaskStatus EvaluateRadiationEnergy(T *md);
TaskStatus UpdateFluid(MeshData<Real> *md);
TaskStatus ControlPopulation(MeshData<Real> *md);

// TaskCollection for radiation step
TaskCollection RadiationStep(Mesh *pmesh, const Real t_start, const Real dt);

// Functions
Real EstimateTimestepMesh(MeshData<Real> *md);
void InitializeRadiation(MeshBlockData<Real> *mbd, const bool is_thermal);

} // namespace jaybenne

#endif // JAYBENNE_JAYBENNE_HPP_
